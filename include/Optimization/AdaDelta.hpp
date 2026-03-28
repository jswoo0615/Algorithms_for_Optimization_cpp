#ifndef OPTIMIZATION_AdaDelta_HPP_
#define OPTIMIZATION_AdaDelta_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>

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

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief AdaDelta 최적화 알고리즘
 * @note 학습률(Learning Rate)을 완전히 제거
 * MISRA C++ 준수. O(1) 메모리 할당 및 FMA 가속 적용.
 */
class AdaDelta {
public:
    AdaDelta() = delete; 

    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(
        Func f, std::array<double, N> x_init,
        double gamma = 0.9, double epsilon = 1e-8,
        size_t max_iter = 15000, double tol = 1e-4,
        bool verbose = false) noexcept 
    {
        static_assert(N > 0, "Dimension N must be greater than 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        alignas(64) std::array<double, N> x = x_init;
        
        // 과거 기울기 제곱의 지수 이동 평균 (E[g^2])
        alignas(64) std::array<double, N> s = {0.0};
        // 과거 위치 변화량 제곱의 지수 이동 평균 (E[Δx^2])
        alignas(64) std::array<double, N> u = {0.0};

        const double tol_sq = tol * tol;
        double f_x = 0.0;
        size_t iter = 0;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 📐 AdaDelta Optimizer Started (gamma=" << gamma << ")\n";
            std::cout << "========================================================\n";
        }

        for (iter = 1; iter <= max_iter; ++iter) {
            alignas(64) std::array<double, N> g = {0.0};
            
            // 1. Auto Diff 호출 (O(1) 정적 할당)
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            double g_norm_sq = 0.0;
            #pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq);
            }

            // 2. 수렴 판정
            if (g_norm_sq < tol_sq) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. AdaDelta 파라미터 업데이트
            #pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                // E[g^2]_t = gamma * E[g^2]_{t-1} + (1 - gamma) * g^2
                s[i] = std::fma(gamma, s[i], (1.0 - gamma) * g[i] * g[i]);
                
                // Δx_t = - (RMS[Δx]_{t-1} / RMS[g]_t) * g_t
                double rms_u = std::sqrt(u[i] + epsilon);
                double rms_s = std::sqrt(s[i] + epsilon);
                double delta_x = -(rms_u / rms_s) * g[i];
                
                // E[Δx^2]_t = gamma * E[Δx^2]_{t-1} + (1 - gamma) * Δx^2
                u[i] = std::fma(gamma, u[i], (1.0 - gamma) * delta_x * delta_x);
                
                x[i] += delta_x;
            }

            if (verbose && (iter % 1000 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(5) << iter 
                          << "] f(x): " << std::fixed << std::setprecision(6) << f_x 
                          << " | ||g||: " << std::sqrt(g_norm_sq) << "\n";
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

}  // namespace Optimization
#endif  // OPTIMIZATION_AdaDelta_HPP_