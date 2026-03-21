#ifndef OPTIMIZATION_NEWTON_METHOD_HPP_
#define OPTIMIZATION_NEWTON_METHOD_HPP_

#include <array>
#include <cmath>
#include <optional>
#include <chrono>
#include <algorithm>

// 리포지토리에 구현된 AutoDiff 엔진을 사용합니다.
#include "Optimization/AutoDiff.hpp"

namespace Optimization {

    /**
     * @brief 최적화 결과를 담는 구조체 (N차원 지원)
     */
    template <size_t N>
    struct OptimizationResult {
        std::array<double, N> x_opt;
        double f_opt;
        size_t iterations;
        long long elapsed_ns;       // 실제 소요 시간 (나노초)
    };

    class NewtonMethod {
    private:
        /**
         * @brief 선형 시스템 해 풀이 (Hessian Inverse * Gradient)
         * @note N=2인 경우 Cramer's Rule로 초고속 처리, N>2인 경우 정적 가우스 소거법(O(N^3))
         */
        template <size_t N>
        [[nodiscard]] static constexpr std::optional<std::array<double, N>> solve_newton_system(
            std::array<std::array<double, N>, N> H,
            std::array<double, N> g) noexcept
        {
            if constexpr (N == 2) {
                // [2차원 최적화] Cramer's rule 및 FMA (Fused Multiply-Add)
                const double det = std::fma(H[0][0], H[1][1], -H[0][1] * H[1][0]);

                if (std::abs(det) < 1e-15) {
                    return std::nullopt; // Singular Matrix (평탄한 지형) 대처
                }

                const double inv_det = 1.0 / det;
                return std::array<double, 2>{
                    -inv_det * std::fma(H[1][1], g[0], -H[0][1] * g[1]),
                    -inv_det * std::fma(H[0][0], g[1], -H[1][0] * g[0])
                };
            } else {
                // [N차원 최적화] 정적 가우스 소거법 (동적 할당 Zero)
                for (size_t i = 0; i < N; ++i) {
                    size_t pivot = i;
                    for (size_t j = i + 1; j < N; ++j) {
                        if (std::abs(H[j][i]) > std::abs(H[pivot][i])) pivot = j;
                    }
                    std::swap(H[i], H[pivot]);
                    std::swap(g[i], g[pivot]);

                    if (std::abs(H[i][i]) < 1e-15) return std::nullopt;

                    for (size_t j = i + 1; j < N; ++j) {
                        const double factor = H[j][i] / H[i][i];
                        for (size_t k = i; k < N; ++k) H[j][k] -= factor * H[i][k];
                        g[j] -= factor * g[i];
                    }
                }

                std::array<double, N> p = {0.0};
                for (size_t i = N; i-- > 0;) {
                    double sum = 0.0;
                    for (size_t j = i + 1; j < N; ++j) sum = std::fma(H[i][j], p[j], sum);
                    p[i] = -(g[i] - sum) / H[i][i]; // 방향 유지를 위해 부호 반전 (-H^-1 * g)
                }
                return p;
            }
        }

    public:
        NewtonMethod() = delete;

        template <size_t N, typename Func>
        [[nodiscard]] static OptimizationResult<N> optimize(
            Func f, std::array<double, N> x_init,
            double tol = 1e-6, size_t max_iter = 50) noexcept
        {
            auto start_clock = std::chrono::high_resolution_clock::now();

            std::array<double, N> x = x_init;
            const double tol_sq = tol * tol;
            size_t iter = 0;

            for (iter = 1; iter <= max_iter; ++iter) {
                double f_val = 0.0;
                std::array<double, N> g = {0.0};

                // 리포지토리의 AutoDiff.hpp 활용: 목적 함수값과 기울기(1차 미분)를 O(1) 할당으로 획득
                AutoDiff::value_and_gradient<N>(f, x, f_val, g);

                double g_norm_sq = 0.0;
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    g_norm_sq = std::fma(g[i], g[i], g_norm_sq);
                }

                if (g_norm_sq < tol_sq) {
                    break;
                }

                // 리포지토리의 AutoDiff.hpp 활용: Central Difference 기반 헤시안 행렬(2차 미분) 획득
                auto H = AutoDiff::hessian<N>(f, x);
                auto p = solve_newton_system<N>(H, g);

                if (!p) {
                    break; // 역행렬 실패 시 즉시 중단
                }

                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    x[i] += (*p)[i]; // p는 -H^-1 * g 로 계산되어 있으므로 더해줌
                }
            }

            auto end_clock = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

            return {x, AutoDiff::value<N>(f, x), iter, duration.count()};
        }
    };
} // namespace Optimization

#endif // OPTIMIZATION_NEWTON_METHOD_HPP_