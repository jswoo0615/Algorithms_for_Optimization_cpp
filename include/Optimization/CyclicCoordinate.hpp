#ifndef OPTIMIZATION_CYCLIC_COORDINATE_HPP_
#define OPTIMIZATION_CYCLIC_COORDINATE_HPP_

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "Optimization/LineSearch.hpp"

namespace Optimization {

    template <size_t N>
    struct OptimizationResultND {
        std::array<double, N> x_opt;
        double f_opt;
        size_t iterations;
        long long elapsed_ns;
    };

    /**
     * @brief LineSearch 모듈을 순환 좌표 탐색에 연결
     * @details 구간 추정 (Bracket Minimum) 후 황금 분할 탐색 (Golden Section Search)을 연속 수행
     */
    struct GoldenSectionStrategy {
        template <size_t N, typename Func>
        [[nodiscard]] std::array<double, N> operator()(Func f, const std::array<double, N>& x,
                                                       const std::array<double, N>& d) const noexcept {
            // 1. 최솟값이 존재하는 구간 [a, b] 탐색
            auto [a, b] = LineSearch::bracket_minimum<N>(f, x, d, 1e-2, 2.0, false);

            // 2. 황금 분할 탐색으로 최적의 이동 보폭 (alpha) 계산
            double alpha = LineSearch::golden_section_search<N>(f, x, d, a, b, 1e-5, false);

            // 3. 최적 위치로 이동 (FMA 적용)
            alignas(64) std::array<double, N> x_new = {0.0};
            #pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                x_new[i] = std::fma(alpha, d[i], x[i]);
            }
            return x_new;
        }
    };

    /**
     * @brief Cyclic Coordinate Search (순환 좌표 탐색) 알고리즘
     */
    class CyclicCoordinateSearch {
        public:
            CyclicCoordinateSearch() = delete;  // 인스턴스화 방지

            // ================================================
            // Algorithm 7.2 : 기본 순환 좌표 탐색
            // ================================================
            template <size_t N, typename Func, typename LineSearchFunc = GoldenSectionStrategy>
            [[nodiscard]] static OptimizationResultND<N> optimize(
                Func f, std::array<double, N> x_init, LineSearchFunc line_search = GoldenSectionStrategy{},
                double tol = 1e-5, size_t max_iter = 500, bool verbose = false) noexcept {
                static_assert(N > 0, "Dimension N must be greater than 0");
                auto start_clock = std::chrono::high_resolution_clock::now();

                alignas(64) std::array<double, N> x = x_init;
                const double tol_sq = tol * tol;
                size_t iter = 0;

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🧭 Cyclic Coordinate Search Started \n";
                    std::cout << "========================================================\n";
                }

                for (iter = 1; iter <= max_iter; ++iter) {
                    alignas(64) std::array<double, N> x_prime = x;
                    for (size_t i = 0; i < N; ++i) {
                        alignas(64) std::array<double, N> d = {0.0};
                        d[i] = 1.0;
                        x = line_search(f, x, d);
                    }

                    double delta_sq = 0.0;
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        const double diff = x[i] - x_prime[i];
                        delta_sq = std::fma(diff, diff, delta_sq);
                    }

                    if (verbose && (iter % 10 == 0 || iter == 1)) {
                        std::cout << "[Iter " << std::setw(3) << iter 
                                << "] f(x): " << std::fixed << std::setprecision(6) << f(x) 
                                << " | ||Δ||: " << std::sqrt(delta_sq) << "\n";
                    }

                    if (delta_sq < tol_sq) {
                        break;
                    }
                }

                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                return {x, f(x), iter, duration.count()};
            }

            // ================================================
            // Algorithm 7.3 : 가속 스텝을 포함한 순환 좌표 탐색
            // ================================================
            template <size_t N, typename Func, typename LineSearchFunc = GoldenSectionStrategy>
            [[nodiscard]] static OptimizationResultND<N> optimize_accelerated(
                Func f, std::array<double, N> x_init, LineSearchFunc line_search = GoldenSectionStrategy{},
                double tol = 1e-5, size_t max_iter = 500, bool verbose = false) noexcept {
                static_assert(N > 0, "Dimension N must be greater than 0");
                auto start_clock = std::chrono::high_resolution_clock::now();

                alignas(64) std::array<double, N> x = x_init;
                const double tol_sq = tol * tol;
                size_t iter = 0;

                for (iter = 1; iter <= max_iter; ++iter) {
                    alignas(64) std::array<double, N> x_prime = x;
                    for (size_t i = 0; i < N; ++i) {
                        alignas(64) std::array<double, N> d = {0.0};
                        d[i] = 1.0;
                        x = line_search(f, x, d);
                    }

                    alignas(64) std::array<double, N> d_acc = {0.0};
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        d_acc[i] = x[i] - x_prime[i];
                    }
                    x = line_search(f, x, d_acc);

                    double delta_sq = 0.0;
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) {
                        const double diff = x[i] - x_prime[i];
                        delta_sq = std::fma(diff, diff, delta_sq);
                    }

                    if (delta_sq < tol_sq) {
                        break;
                    }
                }

                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                return {x, f(x), iter, duration.count()};
            }
    };
} // namespace Optimization
#endif // OPTIMIZATION_CYCLIC_COORDINATE_HPP_