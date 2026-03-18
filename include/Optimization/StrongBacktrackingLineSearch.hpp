#ifndef OPTIMIZATION_STRONG_BACKTRACKING_LINE_SEARCH_HPP_
#define OPTIMIZATION_STRONG_BACKTRACKING_LINE_SEARCH_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
/**
 * @brief [Chapter 4] Algorithm 4.3 : Strong Backtracking Line Search
 * Armijo (충분 감소) 조건과 Strong Curvature (강한 곡률) 조건을 모두 만족하는
 * 강한 울프 조건 (Strong Wolfe Conditions) Line 탐색
 */
class StrongBacktrackingLineSearch {
   private:
    template <size_t N>
    static std::array<double, N> ray_point(const std::array<double, N>& x,
                                           const std::array<double, N>& d, double alpha) {
        std::array<double, N> pt;
        for (size_t i = 0; i < N; ++i) {
            pt[i] = x[i] + alpha * d[i];
        }
        return pt;
    }

   public:
    template <size_t N, typename Func>
    static double search(Func f, const std::array<double, N>& x, const std::array<double, N>& d,
                         double alpha_init = 1.0, double c1 = 1e-4, double c2 = 0.9,
                         bool verbose = false) {
        // 1. 출발점 정보 획득
        double f_x;
        std::array<double, N> grad_x;
        AutoDiff::value_and_gradient<N>(f, x, f_x, grad_x);

        double dir_deriv = 0.0;
        for (size_t i = 0; i < N; ++i) {
            dir_deriv += grad_x[i] * d[i];
        }
        if (dir_deriv >= 0.0 && verbose) {
            std::cout << "  [Warning] Not a descent direction!\n";
        }
        // 2. 탐색 구간 초기화
        double alpha_lo = 0.0;
        double alpha_hi = 1e9;  // 초기 상한선은 무한대로 설정
        double alpha = alpha_init;

        size_t iter = 0;
        while (true) {
            iter++;
            auto x_new = ray_point<N>(x, d, alpha);
            double f_new;
            std::array<double, N> grad_new;
            AutoDiff::value_and_gradient<N>(f, x_new, f_new, grad_new);

            double dir_deriv_new = 0.0;
            for (size_t i = 0; i < N; ++i) {
                dir_deriv_new += grad_new[i] * d[i];
            }
            if (verbose) {
                std::cout << "  ↳ [Wolfe Iter " << std::setw(2) << iter << "] alpha: " << std::fixed
                          << std::setprecision(6) << alpha << " | Range: [" << alpha_lo << ", "
                          << (alpha_hi > 1e8 ? "INF" : std::to_string(alpha_hi)) << "]\n";
            }

            // [조건 1] Armijo 검사 : 너무 멀리 갔는가? (함수값이 충분히 안 떨어짐)
            if (f_new > f_x + c1 * alpha * dir_deriv || dir_deriv_new > 0.0) {
                if (verbose)
                    std::cout << "      -> Armijo failed or went uphill. Shrinking upper bound.\n";
                alpha_hi = alpha;                     // 상한선을 현재 위치로 당김
                alpha = (alpha_lo + alpha_hi) / 2.0;  // 구간의 절반으로 후퇴
            }

            // [조건 2] Strong Curvature 검사 : 너무 조금 갔는가? (기울기가 아직도 너무 가파름)
            else if (std::abs(dir_deriv_new) > c2 * std::abs(dir_deriv)) {
                if (verbose)
                    std::cout << "      -> Curvature failed (Too steep). Expanding lower bound.\n";
                alpha_lo = alpha;  // 하한선을 현재 위치로 올림
                if (alpha_hi >= 1e8) {
                    alpha *= 2.0;  // 상한선을 아직 못 찾았다면 보폭을 2배로 늘림
                } else {
                    alpha = (alpha_lo + alpha_hi) / 2.0;  // 상한선이 있다면 그 사이의 절반으로 전진
                }
            }

            // [조건 3] 완벽한 타격 (두 조건 모두 통과)
            else {
                if (verbose) std::cout << "  ↳ [Accepted] Strong Wolfe conditions satisfied!\n";
                return alpha;
            }

            // 무한 루프 방지 (정밀도 한계)
            if (std::abs(alpha_hi - alpha_lo) < 1e-10) {
                if (verbose)
                    std::cout << "  ↳ [Failsafe] Bracket too small, returning best alpha.\n";
                return alpha;
            }
        }
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_STRONG_BACKTRACKING_LINE_SEARCH_HPP_