#ifndef OPTIMIZATION_NOISY_DESCENT_HPP_
#define OPTIMIZATION_NOISY_DESCENT_HPP_

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include "Optimization/AutoDiff.hpp"

#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
    template <size_t N>
    struct OptimizationResultND {
        std::array<double, N> x_opt;
        double f_opt;
        size_t iterations;
        long long elapsed_ns;
    };
}
#endif // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {
    /**
     * @brief Noisy Descent (노이즈 하강법) 알고리즘
     * @note AutoDiff 엔진의 정확한 1차 미분 (Gradient)에 점진적으로 감소하는 가우시안 노이즈를
     * 결합하여 안장점 (Saddle Point) 및 국소 최적해 (Local Minima)를 탈출합니다.
     * MISRA C++ 준수. 난수 생성기 정적 할당을 통한 지연 (Latency) 최소화 및 FMA 적용
     */

     class NoisyDescent {
        public:
            NoisyDescent() = delete;    // 인스턴스화 방지
            // SigmaFunc : 반복 횟수 (iter)를 받아 현재의 표준 편차 sigma를 반환하는 스케줄링 콜백 함수
            template <size_t N, typename Func, typename SigmaFunc>
            [[nodiscard]] static OptimizationResultND<N> optimize(
                Func f, std::array<double, N> x_start, SigmaFunc sigma_func, 
                double alpha = 0.01, double tol = 1e-5, size_t max_iter = 10000,
                bool verbose = false) noexcept {
                static_assert(N > 0, "Dimension N must be greater than 0.");
                auto start_clock = std::chrono::high_resolution_clock::now();

                const double tol_sq = tol * tol;
                alignas(64) std::array<double, N> x = x_start;

                // [핵심 최적화] 하드웨어 엔트로피 획득 및 난수 엔진 초기화는 스레드당 1회만 수행
                // 매 iteration 또는 함수 호출 시 발생하는 병목 완전 제거
                static thread_local std::mt19937 gen(std::random_device{}());
                std::normal_distribution<double> dist(0.0, 1.0);

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🎲 Noisy Descent Started (alpha=" << alpha << ")\n";
                    std::cout << "========================================================\n";
                }

                double f_x = 0.0;
                size_t iter = 0;

                for (iter = 1; iter <= max_iter; ++iter) {
                    alignas(64) std::array<double, N> g{0.0}; // Gradient 벡터 (캐시 라인 정렬)

                    // 1. Auto Diff를 통한 정확한 목적 함수 값 및 1차 미분 (Gradient) 획득
                    AutoDiff::value_and_gradient<N>(f, x, f_x, g);

                    double g_norm_sq = 0.0;
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        g_norm_sq = std::fma(g[i], g[i], g_norm_sq);
                    }

                    // 2. 외부 주입된 노이즈 스케줄링 콜백을 통해 현재 sigma 계산
                    double sigma = sigma_func(iter);

                    // 3. 종료 조건 검사 : 기울기가 0에 수렴하고, 노이즈가 허용치 이하로 소멸했을 때
                    if (g_norm_sq < tol_sq && sigma < tol) {
                        if (verbose) 
                            std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                        break;
                    }

                    // 4. 파라미터 업데이트 (경사 하강 + 가우시안 확률론적)
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        double noise_term = sigma * dist(gen);
                        // x[i] = x[i] - alpha * g[i] + noise_term; // 기존 업데이트
                        x[i] = std::fma(-alpha, g[i], x[i]) + noise_term; // FMA 적용하여 정확도 및 성능 향상
                    }

                    if (verbose && (iter % 100 == 0 || iter == 1)) {
                        std::cout << "[Iter " << std::setw(4) << iter 
                                << "] f(x): " << std::fixed << std::setprecision(6) << f_x 
                                << " | ||g||: " << std::sqrt(g_norm_sq) 
                                << " | sigma: " << sigma << "\n";
                    }
                }
                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🏁 Final Optimal Point: [" << x[0];
                    if constexpr (N > 1) std::cout << ", " << x[1];
                    if constexpr (N > 2) std::cout << ", ...";
                    std::cout << "]\n========================================================\n";
                }
                
                return {x, f_x, iter, duration.count()};
            }
     };
} // namespace Optimization

#endif // OPTIMIZATION_NOISY_DESCENT_HPP_