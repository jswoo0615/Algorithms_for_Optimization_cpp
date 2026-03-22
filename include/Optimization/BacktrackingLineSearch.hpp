#ifndef OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_
#define OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_

#include <array>
#include <iomanip>
#include <iostream>
#include <cmath>
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
    /**
     * @brief Algorithm 4.2 : Backtracking Line Search
     * 아르미호 (Armijo) 충분 감소 조건을 만족할 때까지 보폭을 줄여나가는 근사 선 탐색 알고리즘
     */
    class BacktrackingLineSearch {
        private:
            template <size_t N>
            [[nodiscard]] static constexpr std::array<double, N> ray_point(
                const std::array<double, N>& x, const std::array<double, N>& d, double alpha) noexcept {
                std::array<double, N> pt = {0.0};
                #pragma omd simd
                for (size_t i = 0; i < N; ++i) {
                    pt[i] = std::fma(alpha, d[i], x[i]);    // x[i] + (alpha * d[i])
                }
                return pt;
            }

        public:
            template <size_t N, typename Func>
            [[nodiscard]] static double search(Func f, const std::array<double, N>& x, 
                                               const std::array<double, N>& d, double f_x, 
                                               const std::array<double, N>& grad_x, double alpha = 1.0, 
                                               double p = 0.5, double c = 1e-4, bool verbose = false) noexcept {
                // 방향 도함수 (Directional Derivative) 계산 : ∇f(x)^T * d
                double dir_deriv = 0.0;
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    dir_deriv = std::fma(grad_x[i], d[i], dir_deriv);
                }

                if (dir_deriv >= 0.0 && verbose) {
                    std::cout << "  [Warning] Not a descent direction! dir_deriv: " << dir_deriv << "\n";
                }

                size_t iter = 0;
                while (true) {
                    iter++;
                    auto x_new = ray_point<N>(x, d, alpha);
                    double f_new = AutoDiff::value<N>(f, x_new);

                    // 목표 감소치 계산 (출발점 함수값 + c * alpha * dir_deriv)
                    double target_val = f_x + c * alpha * dir_deriv;

                    if (verbose) {
                        std::cout << "  ↳ [Backtrack Iter " << std::setw(2) << iter
                                << "] alpha: " << std::fixed << std::setprecision(6) << alpha
                                << " | f_new: " << f_new << " | Target: " << target_val << "\n";
                    }

                    // Armijo 조건 검사
                    if (f_new <= target_val) {
                        if (verbose) std::cout << "  ↳ [Accepted] Armijo condition satisfied!\n";
                        return alpha;
                    }

                    alpha *= p;

                    if (alpha < 1e-10) {
                        if (verbose) std::cout << "  ↳ [Failsafe] Alpha reached minimum limit.\n";
                        return alpha;
                    }
                }
            }

            template <size_t N, typename Func>
            [[nodiscard]] static double search(Func f, const std::array<double, N>& x, const std::array<double, N>& d,
                                               double alpha = 1.0, double p = 0.5, double c = 1e-4,
                                               bool verbose = false) noexcept {
                double f_x = 0.0;
                std::array<double, N> grad_x = {0.0};
                AutoDiff::value_and_gradient<N>(f, x, f_x, grad_x);

                return search<N>(f, x, d, f_x, grad_x, alpha, p, c, verbose);
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_