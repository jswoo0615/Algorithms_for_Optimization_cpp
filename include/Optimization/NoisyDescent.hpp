#ifndef OPTIMIZATION_NOISY_DESCENT_HPP_
#define OPTIMIZATION_NOISY_DESCENT_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "Optimization/AutoDiff.hpp"

// 최적화 결과를 담는 구조체 중복 정의 방지용 매크로
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 저장하는 구조체
 * @tparam N 최적화 대상 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt; ///< 최적해 (Optimal solution)
    double f_opt;                ///< 최적 함수값 (Optimal function value)
    size_t iterations;           ///< 총 반복 횟수 (Total iterations)
    long long elapsed_ns;        ///< 소요 시간 (나노초 단위)
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @class NoisyDescent
 * @brief 노이즈 하강법(Noisy Descent) 알고리즘 정적 클래스
 * 
 * @details 경사 하강법(Gradient Descent)은 지형의 기울기(Gradient)만을 따라 내려가므로 
 * 평탄한 지역(Plateau), 안장점(Saddle Point), 혹은 얕은 지역 최솟값(Local Minima)에 갇히기 쉽습니다.
 * 노이즈 하강법은 기존의 1차 미분(Gradient) 업데이트 수식에 **무작위 가우시안 노이즈(Gaussian Noise)**를 추가로 주입하여 
 * 이러한 함정들을 확률적으로 탈출할 수 있도록 돕는 알고리즘입니다.
 * 
 * 핵심은 탐색 초반에는 노이즈를 크게 주어 넓은 영역을 탐색하게 하고,
 * 최적해에 가까워질수록(반복 횟수가 증가할수록) 노이즈의 분산(sigma)을 점진적으로 줄여 정밀하게 수렴하도록 유도하는 것입니다.
 * 
 * 수식:
 * x_{t+1} = x_t - alpha * g_t + N(0, sigma_t^2)
 * (여기서 N(0, sigma_t^2)는 평균이 0, 표준 편차가 sigma_t인 가우시안 노이즈)
 * 
 * @note 본 구현체는 다음과 같은 성능 최적화를 포함합니다:
 * 1. MISRA C++ 준수 및 동적 할당 완전 배제.
 * 2. 난수 생성기(mt19937)를 `thread_local`로 선언하여 매 호출 시 발생하는 초기화 오버헤드 원천 차단.
 * 3. 벡터 연산 시 FMA(Fused Multiply-Add)를 적용하여 정밀도 손실 없이 단일 명령어로 연산 가속.
 */
class NoisyDescent {
   public:
    NoisyDescent() = delete;  // 정적 메서드만 제공하므로 인스턴스화 방지

    /**
     * @brief 노이즈 하강법을 사용하여 목적 함수의 최적해를 찾습니다.
     * 
     * @tparam N 최적화하려는 변수의 차원 수
     * @tparam Func 목적 함수 타입
     * @tparam SigmaFunc 노이즈의 표준 편차(sigma)를 제어하는 스케줄링 콜백 함수 타입
     * @param f 최적화할 목적 함수 (AutoDiff 호환 가능해야 함)
     * @param x_start 탐색을 시작할 초기 위치 (N차원 벡터)
     * @param sigma_func 현재 반복 횟수(iter)를 인자로 받아 현재 세대의 표준 편차(sigma)를 반환하는 콜백 함수.
     *                   (예: `[](size_t iter) { return 1.0 / iter; }` 등 점진적 감소 형태 권장)
     * @param alpha 학습률 (Learning Rate). 경사 하강 보폭 크기 (기본값: 0.01)
     * @param tol 허용 오차. 기울기 크기와 노이즈 크기가 동시에 이 값 미만이 될 때 수렴으로 판단 (기본값: 1e-5)
     * @param max_iter 최대 반복 횟수 (기본값: 10000)
     * @param verbose 콘솔에 진행 과정을 출력할지 여부
     * @return 최적해(x_opt), 최적값(f_opt), 반복 횟수, 연산 시간을 포함하는 구조체
     */
    template <size_t N, typename Func, typename SigmaFunc>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_start,
                                                          SigmaFunc sigma_func, double alpha = 0.01,
                                                          double tol = 1e-5,
                                                          size_t max_iter = 10000,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0."); // 컴파일 타임 차원 검사
        auto start_clock = std::chrono::high_resolution_clock::now(); // 소요 시간 측정 시작

        // 반복문 내부에서 불필요한 제곱근 연산을 피하기 위해 오차 허용치의 제곱을 미리 계산
        const double tol_sq = tol * tol;
        
        // 캐시 라인(64 byte) 정렬을 통해 메모리 접근 속도 최적화
        alignas(64) std::array<double, N> x = x_start;

        // [핵심 최적화] 하드웨어 엔트로피 획득 및 난수 엔진 초기화는 스레드당 1회만 수행
        // 매 이터레이션(반복) 혹은 함수 호출 시마다 엔진을 새로 생성하면 병목(Bottleneck)이 발생하므로 이를 완전 제거
        static thread_local std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dist(0.0, 1.0); // 평균 0, 표준 편차 1인 표준 정규 분포

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🎲 Noisy Descent Started (alpha=" << alpha << ")\n";
            std::cout << "========================================================\n";
        }

        double f_x = 0.0;
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            // 캐시 정렬된 기울기 벡터 초기화
            alignas(64) std::array<double, N> g{0.0};  

            // -------------------------------------------------------------
            // 1. 목적 함수값 및 기울기 평가
            // AutoDiff(자동 미분) 엔진을 호출하여 수치적 오차 없는 정확한 함수값과 기울기(g)를 동시에 획득
            // -------------------------------------------------------------
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 기울기 노름의 제곱(Gradient Norm Squared) 계산
            double g_norm_sq = 0.0;
// OpenMP SIMD 지시어를 사용하여 벡터 누적 연산을 가속 (컴파일러 지원 시)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                // g_norm_sq += g[i] * g[i] 와 동일한 FMA 연산
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq); 
            }

            // -------------------------------------------------------------
            // 2. 노이즈 스케줄링(Noise Scheduling) 적용
            // 외부에서 주입된 콜백 함수(sigma_func)를 호출하여 현재 반복 횟수(iter)에 맞는 
            // 가우시안 노이즈의 표준 편차(sigma) 크기를 결정합니다. (일반적으로 점진적 감소)
            // -------------------------------------------------------------
            double sigma = sigma_func(iter);

            // -------------------------------------------------------------
            // 3. 조기 종료(Convergence) 조건 검사
            // 단순 경사 하강법과 달리, 1) 기울기가 0에 수렴해야 하고 (g_norm_sq < tol_sq) 
            // 2) 주입되는 노이즈 또한 허용치(tol) 이하로 소멸되어야만 최종 수렴한 것으로 판단합니다.
            // -------------------------------------------------------------
            if (g_norm_sq < tol_sq && sigma < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // -------------------------------------------------------------
            // 4. 파라미터 업데이트 (경사 하강 방향 이동 + 가우시안 노이즈 주입)
            // -------------------------------------------------------------
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                // 표준 정규 분포에서 뽑은 난수에 sigma를 곱해 N(0, sigma^2) 노이즈 생성
                double noise_term = sigma * dist(gen);
                
                // 기존 수식: x[i] = x[i] - alpha * g[i] + noise_term;
                // FMA(Fused Multiply-Add)를 적용하여 곱셈과 덧셈을 한 번에 정확하게 처리
                // std::fma(-alpha, g[i], x[i]) 가 (x[i] - alpha * g[i]) 와 동일함
                x[i] = std::fma(-alpha, g[i], x[i]) + noise_term;  
            }

            // 진행 상태 출력: 첫 세대이거나 매 100세대마다 한 번씩 출력
            if (verbose && (iter % 100 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(4) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << std::sqrt(g_norm_sq)
                          << " | sigma: " << sigma << "\n";
            }
        }
        
        // 탐색 종료 후 시간 측정 완료
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

        // 최적해와 부가 정보를 결과 구조체에 담아 반환
        return {x, f_x, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_NOISY_DESCENT_HPP_