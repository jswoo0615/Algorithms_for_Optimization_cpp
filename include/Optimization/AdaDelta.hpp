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
    std::array<double, N> x_opt; ///< 최적해 (Optimal Point)
    double f_opt;                ///< 최적해에서의 목적 함수 값
    size_t iterations;           ///< 최적해를 찾는 데 소요된 반복 횟수
    long long elapsed_ns;        ///< 최적화에 소요된 시간 (나노초 단위)
};
}  // namespace Optimization
#endif

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief AdaDelta 최적화 알고리즘
 * 
 * AdaGrad 알고리즘은 과거의 기울기 제곱을 계속 누적하기 때문에 반복이 진행될수록 
 * 학습률(Step Size)이 급격히 작아져 결국 학습이 멈추는(Premature Convergence) 단점이 있습니다.
 * 
 * AdaDelta는 이를 극복하기 위해 제안된 알고리즘으로, 다음 두 가지 핵심 개선 사항을 가집니다:
 * 1. 기울기 누적합 대신 지수 이동 평균(Exponential Moving Average, EMA)을 사용하여 최근 기울기 정보에 가중치를 부여합니다.
 * 2. 변수 업데이트량(Δx)의 지수 이동 평균을 함께 추적하여, 사용자가 직접 지정해야 하는 글로벌 '학습률(Learning Rate)' 파라미터를 완전히 제거했습니다.
 *    (업데이트의 물리적 단위가 일치하도록 차원 분석을 통해 자체적인 Step Size를 생성합니다.)
 * 
 * @note MISRA C++ 코딩 표준을 지향하여 동적 메모리 할당을 배제하고, O(1) 정적 메모리와 FMA(Fused Multiply-Add) 명령어를 통해 최적화 속도를 가속합니다.
 */
class AdaDelta {
   public:
    // 정적 유틸리티 클래스로 활용하기 위해 생성자 삭제
    AdaDelta() = delete;

    /**
     * @brief AdaDelta 메인 최적화 함수
     * 
     * @tparam N 최적화 변수의 차원
     * @tparam Func 목적 함수 타입 (람다식, 펑터 등)
     * @param f 최소화할 목적 함수
     * @param x_init 최적화 시작 지점 (초기 추정값)
     * @param gamma 감쇠율 (Decay Rate, 보통 0.9 ~ 0.99 사용). 과거 정보가 잊혀지는 속도를 결정합니다.
     * @param epsilon 0으로 나누어지는 것을 방지하는 수치적 안정성 상수 (작은 양수)
     * @param max_iter 무한 루프를 방지하기 위한 최대 허용 반복 횟수
     * @param tol 수렴 판정을 위한 허용 오차 (Gradient의 노름(Norm)이 이 값보다 작아지면 수렴으로 판정)
     * @param verbose 콘솔에 최적화 진행 상황을 출력할지 여부
     * @return OptimizationResultND<N> 최적화가 완료된 해, 목적 함수 값, 반복 횟수, 수행 시간 정보
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          double gamma = 0.9, double epsilon = 1e-8,
                                                          size_t max_iter = 15000,
                                                          double tol = 1e-4,
                                                          bool verbose = false) noexcept {
        // N은 최소 1차원 이상이어야 합니다.
        static_assert(N > 0, "Dimension N must be greater than 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        // 캐시 라인 정렬(64바이트)을 통해 SIMD 벡터 연산 성능 극대화
        alignas(64) std::array<double, N> x = x_init;

        // s: 과거 기울기 제곱의 지수 이동 평균 배열 (E[g^2])
        alignas(64) std::array<double, N> s = {0.0};
        
        // u: 과거 변수 위치 변화량(Δx) 제곱의 지수 이동 평균 배열 (E[Δx^2])
        // 이 배열 덕분에 글로벌 학습률(Learning Rate)이 필요하지 않게 됩니다.
        alignas(64) std::array<double, N> u = {0.0};

        const double tol_sq = tol * tol; // 수렴 판정을 위해 허용 오차를 제곱해 둠 (루트 연산 방지)
        double f_x = 0.0;
        size_t iter = 0;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 📐 AdaDelta Optimizer Started (gamma=" << gamma << ")\n";
            std::cout << "========================================================\n";
        }

        // 최적화 루프 시작
        for (iter = 1; iter <= max_iter; ++iter) {
            alignas(64) std::array<double, N> g = {0.0};

            // 1. Auto Diff 호출을 통해 현재 위치 x에서의 함수 값(f_x)과 기울기 벡터(g) 계산
            // (동적 메모리 할당 없는 O(1) 정적 미분기 사용)
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 현재 기울기 벡터의 제곱합(Norm^2)을 계산 (수렴 판정용)
            double g_norm_sq = 0.0;
#pragma omp simd // 컴파일러에게 SIMD 병렬 처리를 지시
            for (size_t i = 0; i < N; ++i) {
                // std::fma(a, b, c) = a * b + c 연산을 하드웨어 단에서 한 번에 처리하여 부동소수점 오차 축소 및 속도 향상
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq);
            }

            // 2. 수렴 판정 (Convergence Check)
            // 기울기의 크기가 지정된 허용 오차 이하로 떨어지면 최적점에 도달했다고 판단
            if (g_norm_sq < tol_sq) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. AdaDelta 파라미터 업데이트 과정
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                // [Step 3-1] 기울기 제곱의 지수 이동 평균 (E[g^2]_t) 업데이트
                // E[g^2]_t = gamma * E[g^2]_{t-1} + (1 - gamma) * g_t^2
                // 이전 반복까지의 누적값(s[i])에 감쇠율(gamma)을 곱하고, 현재 기울기 제곱에 (1 - gamma)를 곱해 더함.
                s[i] = std::fma(gamma, s[i], (1.0 - gamma) * g[i] * g[i]);

                // [Step 3-2] 변위량 산출 전 제곱근 계수(RMS, Root Mean Square) 준비
                // rms_u: 업데이트량(Δx) 제곱의 루트 평균 (단위가 파라미터 x와 동일함)
                // rms_s: 기울기(g) 제곱의 루트 평균 (단위가 기울기 g와 동일함)
                double rms_u = std::sqrt(u[i] + epsilon);
                double rms_s = std::sqrt(s[i] + epsilon);
                
                // [Step 3-3] 이번 스텝의 변위(Δx) 계산
                // Δx_t = - (RMS[Δx]_{t-1} / RMS[g]_t) * g_t
                // 학습률 대신 (rms_u / rms_s) 비율을 사용하여 각 파라미터별로 맞춤형 스텝 폭을 결정합니다.
                double delta_x = -(rms_u / rms_s) * g[i];

                // [Step 3-4] 변위(Δx) 제곱의 지수 이동 평균 (E[Δx^2]_t) 업데이트 (다음 반복을 위함)
                // E[Δx^2]_t = gamma * E[Δx^2]_{t-1} + (1 - gamma) * Δx_t^2
                u[i] = std::fma(gamma, u[i], (1.0 - gamma) * delta_x * delta_x);

                // [Step 3-5] 변수 업데이트
                x[i] += delta_x;
            }

            // 진행 상태 로깅
            if (verbose && (iter % 1000 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << std::sqrt(g_norm_sq)
                          << "\n";
            }
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        
        // 최종 결과 반환
        return {x, f_x, iter, duration.count()};
    }
};

}  // namespace Optimization
#endif  // OPTIMIZATION_AdaDelta_HPP_