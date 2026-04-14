#ifndef OPTIMIZATION_AdaDelta_HPP_
#define OPTIMIZATION_AdaDelta_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

// 최적화 결과 반환용 구조체 일관성 유지 (중복 정의 방지)
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 담는 구조체
 * @tparam N 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt;  ///< 최적해 (Optimal Point)
    double f_opt;                 ///< 최적해에서의 목적 함수 값
    size_t iterations;            ///< 최적해를 찾는 데 소요된 반복 횟수
    long long elapsed_ns;         ///< 최적화에 소요된 시간 (나노초 단위)
};
}  // namespace Optimization
#endif

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief AdaDelta 최적화 알고리즘을 구현한 정적 클래스
 *
 * 기존의 AdaGrad 알고리즘은 과거의 모든 기울기 제곱을 누적하므로 반복이 진행될수록 
 * 분모가 무한히 커지고, 결과적으로 학습률(Step Size)이 0에 수렴하여 학습이 조기 종료(Premature Convergence)되는 단점이 있습니다.
 *
 * AdaDelta는 이를 극복하기 위해 제안된 알고리즘으로, 두 가지 핵심 개선점을 가집니다:
 * 1. 전체 과거 기울기가 아닌, 최근 기울기 정보에 가중치를 두는 '지수 이동 평균(Exponential Moving Average, EMA)'을 사용하여 기울기 제곱을 추적합니다.
 * 2. 변수 업데이트량(Δx)의 지수 이동 평균도 함께 추적합니다. 이를 통해 업데이트의 물리적 단위(Unit)를 맞추어 
 *    사용자가 직접 지정해야 하는 글로벌 '학습률(Learning Rate)' 하이퍼파라미터를 완전히 제거했습니다.
 *
 * @note 본 클래스는 MISRA C++ 코딩 표준을 지향하여 동적 메모리 할당을 배제하고, O(1) 정적 메모리와 
 * FMA(Fused Multiply-Add) 명령어 지원, SIMD 벡터 최적화 구문을 통해 성능을 극대화하도록 설계되었습니다.
 */
class AdaDelta {
   public:
    // 유틸리티 클래스로만 사용되도록 인스턴스 생성을 막음
    AdaDelta() = delete;

    /**
     * @brief AdaDelta 메인 최적화 함수
     *
     * @tparam N 최적화 변수의 차원 (N차원 배열)
     * @tparam Func 목적 함수 타입 (람다식, 펑터 등)
     * @param f 최소화하고자 하는 목적 함수
     * @param x_init 최적화 시작 지점 (초기 파라미터 추정값 배열)
     * @param gamma 감쇠율 (Decay Rate, 보통 0.9 ~ 0.99 사용). 과거 정보가 잊혀지는 속도를 결정하는 가중치 파라미터.
     * @param epsilon 0으로 나누어지는 것(Division by zero)을 방지하기 위한 수치적 안정성 상수 (보통 1e-8 등 매우 작은 양수)
     * @param max_iter 무한 루프를 방지하기 위한 최대 허용 반복 횟수
     * @param tol 수렴 판정을 위한 허용 오차. 기울기(Gradient) 벡터의 L2-Norm(크기)이 이 값보다 작아지면 극값(최적점)에 도달했다고 판정합니다.
     * @param verbose 콘솔에 최적화 진행 상황(반복 횟수, 함수값, 기울기 크기 등)을 출력할지 여부
     * @return OptimizationResultND<N> 최적화가 완료된 해(x_opt), 최소 목적 함수 값(f_opt), 실제 수행된 반복 횟수, 소요 시간 정보
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          double gamma = 0.9, double epsilon = 1e-8,
                                                          size_t max_iter = 15000,
                                                          double tol = 1e-4,
                                                          bool verbose = false) noexcept {
        // 변수의 차원 N은 최소 1차원 이상이어야 합니다.
        static_assert(N > 0, "Dimension N must be greater than 0");
        
        // 최적화 시작 시간을 기록 (나노초 단위 측정을 위함)
        auto start_clock = std::chrono::high_resolution_clock::now();

        // 캐시 라인 정렬(64바이트)을 통해 SIMD 벡터 연산 및 캐시 히트 성능을 극대화한 변수 배열 선언
        alignas(64) std::array<double, N> x = x_init;

        // s: 과거 기울기(Gradient) 제곱의 지수 이동 평균을 저장하는 배열 (E[g^2])
        // 분모에 위치하여 각 차원별 스텝 크기를 스케일링하는 역할을 합니다.
        alignas(64) std::array<double, N> s = {0.0};

        // u: 과거 변수 위치 변화량(Δx) 제곱의 지수 이동 평균을 저장하는 배열 (E[Δx^2])
        // 분자에 위치하며, 글로벌 학습률을 대체하여 파라미터 업데이트 시 단위(Unit)의 일관성을 맞추는 핵심 요소입니다.
        alignas(64) std::array<double, N> u = {0.0};

        // 수렴 판정 시 제곱근(sqrt) 연산을 피하기 위해 허용 오차를 미리 제곱해 둡니다.
        const double tol_sq = tol * tol;  
        double f_x = 0.0;
        size_t iter = 0;

        // 상세 출력(verbose) 모드가 활성화된 경우 시작 메시지 출력
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 📐 AdaDelta Optimizer Started (gamma=" << gamma << ")\n";
            std::cout << "========================================================\n";
        }

        // 최적화 메인 반복 루프 시작
        for (iter = 1; iter <= max_iter; ++iter) {
            // 현재 스텝에서의 기울기(Gradient) 벡터를 저장할 배열을 0으로 초기화 및 정렬
            alignas(64) std::array<double, N> g = {0.0};

            // 1. 순방향/역방향 자동 미분(Auto Differentiation) 호출
            // 현재 파라미터 위치 `x`에서의 목적 함수 값(`f_x`)과 기울기 벡터(`g`)를 계산합니다.
            // 정적 배열을 활용한 AutoDiff 기능으로 동적 메모리 할당 없이 O(1)의 안정성을 보장합니다.
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 현재 위치에서 기울기 벡터의 제곱합(L2-Norm의 제곱)을 계산합니다.
            double g_norm_sq = 0.0;
#pragma omp simd  // 컴파일러에게 이 루프를 SIMD(Single Instruction Multiple Data) 병렬 명령어로 처리하도록 지시
            for (size_t i = 0; i < N; ++i) {
                // std::fma(a, b, c) = (a * b) + c 
                // 단일 사이클 하드웨어 연산으로 처리하여 정밀도를 높이고 부동소수점 반올림 오차를 줄입니다.
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq);
            }

            // 2. 수렴 판정 (Convergence Check)
            // 기울기 벡터의 크기(의 제곱)가 지정된 허용 오차 제곱보다 작아지면 더 이상 
            // 업데이트할 필요가 없는 극소점(수렴)에 도달했다고 판단하고 루프를 탈출합니다.
            if (g_norm_sq < tol_sq) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. AdaDelta 핵심 파라미터 업데이트 로직
#pragma omp simd // 벡터화(Vectorization) 강제 지시어
            for (size_t i = 0; i < N; ++i) {
                // [Step 3-1] 기울기 제곱의 지수 이동 평균 (E[g^2]_t) 갱신
                // 수식: E[g^2]_t = gamma * E[g^2]_{t-1} + (1 - gamma) * g_t^2
                // 이전까지의 누적 정보 s[i]에 gamma를 곱하여 서서히 잊게 만들고, 현재 기울기의 제곱 비중을 (1-gamma)만큼 반영합니다.
                s[i] = std::fma(gamma, s[i], (1.0 - gamma) * g[i] * g[i]);

                // [Step 3-2] 업데이트 변위 산출을 위한 RMS (Root Mean Square) 계산
                // u[i]와 s[i]에 epsilon을 더하여 0으로 나누어지는 것을 방지하고 RMS를 구합니다.
                // rms_u: 과거 변위량(Δx)들의 RMS 값
                // rms_s: 과거 기울기(g)들의 RMS 값
                double rms_u = std::sqrt(u[i] + epsilon);
                double rms_s = std::sqrt(s[i] + epsilon);

                // [Step 3-3] 이번 반복 스텝에서의 변위(Δx) 계산
                // 수식: Δx_t = - (RMS[Δx]_{t-1} / RMS[g]_t) * g_t
                // AdaDelta의 가장 큰 특징인 '파라미터별 맞춤형 스텝 폭' 결정 부분입니다. 
                // 사용자가 지정하는 Learning Rate가 없으며 (rms_u / rms_s) 비율 자체가 그 역할을 대신합니다.
                double delta_x = -(rms_u / rms_s) * g[i];

                // [Step 3-4] 변위(Δx) 제곱의 지수 이동 평균 (E[Δx^2]_t) 갱신
                // 수식: E[Δx^2]_t = gamma * E[Δx^2]_{t-1} + (1 - gamma) * Δx_t^2
                // 계산된 현재 스텝의 변위(delta_x)를 바탕으로, 다음 반복에서 사용할 E[Δx^2] 값을 업데이트합니다.
                u[i] = std::fma(gamma, u[i], (1.0 - gamma) * delta_x * delta_x);

                // [Step 3-5] 실제 파라미터(x) 업데이트 적용
                // 현재 파라미터에 계산된 변위량(delta_x)을 더하여 위치를 갱신합니다.
                x[i] += delta_x;
            }

            // 진행 상태 주기적 로깅 (1000번 반복마다 또는 첫 번째 반복 시 출력)
            if (verbose && (iter % 1000 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << std::sqrt(g_norm_sq)
                          << "\n";
            }
        }

        // 최적화 루프 종료 후 종료 시간 기록 및 소요 시간(나노초) 계산
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        // 최적화 최종 결과 및 파라미터 상태 출력
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        // 최종 최적 해, 마지막 목적 함수 값, 실제 소요된 반복 횟수, 경과 시간을 구조체로 묶어 반환
        return {x, f_x, iter, duration.count()};
    }
};

}  // namespace Optimization
#endif  // OPTIMIZATION_AdaDelta_HPP_