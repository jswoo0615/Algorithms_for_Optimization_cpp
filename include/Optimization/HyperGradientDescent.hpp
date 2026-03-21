#ifndef OPTIMIZATION_HYPER_GRADIENT_DESCENT_HPP_
#define OPTIMIZATION_HYPER_GRADIENT_DESCENT_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief Hyper Gradient Descent Optimizer (Based on Algorithms for Optimization, Sec 5.9)
 * @note MISRA C++ 준수. O(1) 메모리 구조 및 실시간 발산 제어(Fail-safe) 적용.
 */
class HyperGradientDescent {
   public:
    HyperGradientDescent() = delete;

    template <size_t N, typename Func>
    [[nodiscard]] static constexpr std::array<double, N> optimize(
        Func f, std::array<double, N> x, double alpha_init = 0.001, double mu = 1e-6,
        double tol = 1e-5, size_t max_iter = 50000, bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0.");

        double alpha = alpha_init;
        std::array<double, N> g_prev = {0.0};
        std::array<double, N> x_valid = x;  // 발산 방어용 백업 상태

        const double tol_sq = tol * tol;

        // [안전성 보장] 수치적 발산을 막기 위한 학습률 범위
        constexpr double MIN_ALPHA = 1e-8;
        constexpr double MAX_ALPHA = 0.005;  // Rosenbrock과 같은 급경사를 위해 보수적 상한 적용
        constexpr double MAX_GRAD_NORM = 100.0;  // Gradient Clipping 임계값

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x = 0.0;
            std::array<double, N> g = {0.0};

            // 1. Auto Differentiation (O(1) 할당)
            AutoDiff::value_and_gradient(f, x, f_x, g);

            // [Fail-safe] 목적 함수값이 NaN 또는 Inf로 발산한 경우, 직전의 유효한 x 반환
            if (std::isnan(f_x) || std::isinf(f_x)) {
                if (verbose)
                    std::cout << " ⚠️ Warning: Numerical explosion detected. Rolling back.\n";
                return x_valid;
            }
            x_valid = x;  // 현재 상태가 유효하므로 백업

            double g_norm_sq = 0.0;
            double g_dot_gprev = 0.0;

// 2. 기울기 노름 및 내적 동시 계산 (FMA 가속)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq);
                g_dot_gprev = std::fma(g[i], g_prev[i], g_dot_gprev);
            }

            // 3. 종료 조건 검사
            if (g_norm_sq < tol_sq) {
                break;
            }

            // [안전성 보장] Gradient Clipping: 기울기가 비정상적으로 클 때 스케일링
            double g_norm = std::sqrt(g_norm_sq);
            double clip_scale = (g_norm > MAX_GRAD_NORM) ? (MAX_GRAD_NORM / g_norm) : 1.0;

            // 4. 하이퍼그레디언트 업데이트
            alpha += mu * (g_dot_gprev * clip_scale * clip_scale);
            alpha = std::clamp(alpha, MIN_ALPHA, MAX_ALPHA);

// 5. 파라미터 업데이트 (In-place)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                g[i] *= clip_scale;  // 클리핑 적용
                x[i] -= alpha * g[i];
                g_prev[i] = g[i];
            }

            if (verbose && (iter % 5000 == 0)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << g_norm
                          << " | alpha: " << alpha << "\n";
            }
        }
        return x_valid;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_HYPER_GRADIENT_DESCENT_HPP_