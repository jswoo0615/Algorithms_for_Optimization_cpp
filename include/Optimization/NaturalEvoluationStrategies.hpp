#ifndef OPTIMIZATION_NATURAL_EVOLUTION_STRATEGIES_HPP_
#define OPTIMIZATION_NATURAL_EVOLUTION_STRATEGIES_HPP_

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>

// 결과 반환 구조체 일관성 유지
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
#endif

namespace Optimization {

/**
 * @brief Natural Evolution Strategies (NES)
 * @note Fisher Information Matrix 역행렬을 적용한 'Natural Gradient'와 
 * Z-Score 'Fitness Normalization'을 결합하여, 스케일이 불균형한 함수에서도 
 * 오버슈팅 없이 전역 최적해를 안전하게 찾아냅니다.
 */
class NaturalEvolutionStrategies {
public:
    NaturalEvolutionStrategies() = delete;

    template <size_t N, size_t M = 100, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(
        Func f, std::array<double, N> mu_init, std::array<double, N> sigma_sq_init,
        double alpha = 0.05, size_t max_iter = 100, uint32_t seed = 12345, 
        bool verbose = false) noexcept 
    {
        static_assert(N > 0, "Dimension N must be > 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        alignas(64) std::array<double, N> mu = mu_init;
        alignas(64) std::array<double, N> sigma_sq = sigma_sq_init;

        static thread_local std::mt19937 gen(seed);
        std::normal_distribution<double> standard_normal(0.0, 1.0);

        // O(M) 메모리 정적 할당
        static thread_local std::array<std::array<double, N>, M> z_cache;
        static thread_local std::array<double, M> y_cache;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🧬 True Natural Evolution Strategies Started\n";
            std::cout << "========================================================\n";
        }

        double best_y_global = 1e99;
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            double sum_y = 0.0;
            double sum_y_sq = 0.0;
            double best_y_iter = 1e99;

            // 1. 샘플 생성 및 평가 (Pass 1)
            for (size_t p = 0; p < M; ++p) {
                alignas(64) std::array<double, N> x = {0.0};
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    z_cache[p][i] = standard_normal(gen);
                    x[i] = std::fma(std::sqrt(sigma_sq[i]), z_cache[p][i], mu[i]);
                }

                y_cache[p] = f(x); 
                sum_y += y_cache[p];
                sum_y_sq += y_cache[p] * y_cache[p];
                if (y_cache[p] < best_y_iter) best_y_iter = y_cache[p];
            }

            if (best_y_iter < best_y_global) best_y_global = best_y_iter;

            // [핵심 1] Fitness Normalization (Z-Score 정규화)
            // 목적 함수의 극단적 스케일 차이를 억제하여 오버슈팅을 막습니다.
            double mean_y = sum_y / static_cast<double>(M);
            double var_y = (sum_y_sq / static_cast<double>(M)) - (mean_y * mean_y);
            double std_y = std::sqrt(std::max(var_y, 1e-12));

            alignas(64) std::array<double, N> grad_mu = {0.0};
            alignas(64) std::array<double, N> grad_sigma_sq = {0.0};

            // 2. Natural Gradient 누적 (Pass 2)
            for (size_t p = 0; p < M; ++p) {
                // 정규화된 Utility: 함수값이 평균보다 작으면(좋으면) 음수!
                double normalized_y = (y_cache[p] - mean_y) / std_y; 

                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    double var = sigma_sq[i];
                    double std_dev = std::sqrt(var);
                    
                    // [핵심 2] Natural Gradient (F^{-1} 적용)
                    // 기존의 나눗셈 폭탄(Division by zero)이 사라지고 곱셈으로 우아하게 변합니다.
                    double nat_grad_mu = z_cache[p][i] * std_dev;
                    double nat_grad_sigma = (z_cache[p][i] * z_cache[p][i] - 1.0) * var;

                    grad_mu[i] = std::fma(normalized_y, nat_grad_mu, grad_mu[i]);
                    grad_sigma_sq[i] = std::fma(normalized_y, nat_grad_sigma, grad_sigma_sq[i]);
                }
            }

            // 3. 파라미터 업데이트 (최솟값을 찾기 위해 경사 하강)
            double inv_M = 1.0 / static_cast<double>(M);
            #pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                mu[i] = std::fma(-alpha, grad_mu[i] * inv_M, mu[i]);
                sigma_sq[i] = std::fma(-alpha, grad_sigma_sq[i] * inv_M, sigma_sq[i]);

                // 분산 붕괴 방지 하드 리미트
                if (sigma_sq[i] < 1e-4) sigma_sq[i] = 1e-4;
            }

            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter 
                          << "] Best f(x): " << std::fixed << std::setprecision(5) << best_y_global 
                          << " | mu: [" << mu[0] << ", " << mu[1] << "]\n";
            }
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << mu[0];
            if constexpr (N > 1) std::cout << ", " << mu[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        return {mu, best_y_global, iter, duration.count()};
    }
};

} // namespace Optimization
#endif // OPTIMIZATION_NATURAL_EVOLUTION_STRATEGIES_HPP_