#ifndef OPTIMIZATION_BFGS_HPP_
#define OPTIMIZATION_BFGS_HPP_

#include <array>
#include <cmath>
#include <chrono>
#include "Optimization/AutoDiff.hpp"
#include "Optimization/BacktrackingLineSearch.hpp"

namespace Optimization {
    template <size_t N>
    struct OptimizationResultND {
        std::array<double, N> x_opt;
        double f_opt;
        size_t iterations;
        long long elapsed_ns;
    };

    class BFGS {
        private:
            template <size_t N>
            [[nodiscard]] static constexpr double dot(const std::array<double, N>& a, 
                                                      const std::array<double, N>& b) noexcept {
                double sum = 0.0;
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    sum = std::fma(a[i], b[i], sum);
                }
                return sum;
            }

            template <size_t N>
            [[nodiscard]] static constexpr std::array<double, N> mat_vec_mul(
                const std::array<std::array<double, N>, N>& M,
                const std::array<double, N>& v) noexcept {
                
                std::array<double, N> res = {0.0};
                for (size_t i = 0; i < N; ++i) {
                    double sum = 0.0;
                    #pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        sum = std::fma(M[i][j], v[j], sum);
                    }
                    res[i] = sum;
                }
                return res;
            }

        public:
            BFGS() = delete;        // 인스턴스화 금지
            template <size_t N, typename Func>
            [[nodiscard]] static OptimizationResultND<N> optimize(
                Func f, std::array<double, N> x_init, double tol=1e-6, size_t max_iter=200) noexcept {
                
                auto start_clock = std::chrono::high_resolution_clock::now();

                std::array<double, N> x = x_init;
                const double tol_sq = tol * tol;

                alignas(64) std::array<std::array<double, N>, N> V = {0.0};
                for (size_t i = 0; i < N; ++i) {
                    V[i][i] = 1.0;
                }
                double f_val = 0.0;
                std::array<double, N> g = {0.0};
                AutoDiff::value_and_gradient<N>(f, x, f_val, g);

                size_t iter = 0;
                for (iter = 1; iter <= max_iter; ++iter) {
                    double g_norm_sq = dot<N>(g, g);
                    if (g_norm_sq < tol_sq) {
                        break;
                    }
                    // 탐색 방향 p = -V * g
                    std::array<double, N> p = mat_vec_mul<N>(V, g);
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        p[i] = -p[i];
                    }

                    double alpha = BacktrackingLineSearch::search<N>(f, x, p, f_val, g, 1.0, 0.5, 1e-4, false);

                    std::array<double, N> x_new = {0.0};
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        x_new[i] = std::fma(alpha, p[i], x[i]);
                    }

                    double f_new = 0.0;
                    std::array<double, N> g_new = {0.0};
                    AutoDiff::value_and_gradient<N>(f, x_new, f_new, g_new);

                    std::array<double, N> s = {0.0};
                    std::array<double, N> y = {0.0};
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        s[i] = x_new[i] - x[i];
                        y[i] = g_new[i] - g[i];
                    }

                    double y_dot_s = dot<N>(y, s);
                    if (y_dot_s > 1e-10) {
                        const double rho = 1.0 / y_dot_s;
                        auto Vy = mat_vec_mul<N>(V, y);
                        const double yVy = dot<N>(y, Vy);
                        const double scalar_term = rho * (1.0 + rho * yVy);

                        for (size_t i = 0; i < N; ++i) {
                            for (size_t j = i; j < N; ++j) {
                                double updated_val = V[i][j] + scalar_term * s[i] * s[j] - rho * (Vy[i] * s[j] + s[i] * Vy[j]);
                                V[i][j] = updated_val;
                                if (i != j) {
                                    V[j][i] = updated_val;
                                }
                            }
                        }
                    } else if (iter > 1) {
                        for (size_t i = 0; i < N; ++i) {
                            for (size_t j = 0; j < N; ++j) {
                                V[i][j] = (i == j) ? 1.0 : 0.0;
                            }
                        }
                    }
                    x = x_new;
                    f_val = f_new;
                    g = g_new;
                }

                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                return {x, f_val, iter, duration.count()};
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_BFGS_HPP_