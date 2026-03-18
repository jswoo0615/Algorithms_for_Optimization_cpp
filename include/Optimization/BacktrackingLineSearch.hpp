#ifndef OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_
#define OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_

#include <array>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
/**
 * @brief [Chapter 4] Algorithm 4.2 : Backtracking Line Search
 * 아르미호 (Armijo) 충분 감소 조건을 만족할 때까지 보폭을 줄여나가는 실전용 근사 선 탐색 알고리즘
 */
class BacktrackingLineSearch {
   private:
    // --- 내부 헬퍼 : 현재 위치 x에서 방향 d로 alpha만큼 이동한 좌표 변환 ---
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
    /**
     * @param f 목적 함수
     * @param x 현재 위치
     * @param d 탐색 방향 (반드시 하강 방향이어야 함)
     * @param alpha 초기 보폭 (기본값 1.0, 뉴턴법에서는 무조건 1.0으로 시작해야 함)
     * @param p 보폭 축소 비율 (기본값 0.5, 절반씩 깎음. 책에서는 rho로 표기)
     * @param c 충분 감소 조건 (Armijo) 관대도 (기본값 1e-4, 보통 0.0001 같은 아주 작은 값 사용)
     * @param verbose 로깅 여부
     */
    template <size_t N, typename Func>
    static double search(Func f, const std::array<double, N>& x, const std::array<double, N>& d,
                         double alpha = 1.0, double p = 0.5, double c = 1e-4,
                         bool verbose = false) {
        // 1. 현재 위치의 함수값 (f_x)과 기울기 (grad_x) 획득
        double f_x;
        std::array<double, N> grad_x;
        AutoDiff::value_and_gradient<N>(f, x, f_x, grad_x);

        // 2. 방향 도함수 (Directional Derivative) 계산 : ∇f(x)^T * d
        // 이 방향으로 한 발자국 내디딜 때 예상되는 하강 기울기
        double dir_deriv = 0.0;
        for (size_t i = 0; i < N; ++i) {
            dir_deriv += grad_x[i] * d[i];
        }
        // 하강 방향 (Descent Direction)이 아니면 경고 (내적이 양수면 오르막길)
        if (dir_deriv >= 0.0 && verbose) {
            std::cout << "  [Warning] Not a descent direction! dir_deriv: " << dir_deriv << "\n";
        }

        size_t iter = 0;
        while (true) {
            iter++;
            // 3. 임시 이동 위치 계산 : x_new = x + alpha * d
            auto x_new = ray_point<N>(x, d, alpha);
            double f_new = AutoDiff::value<N>(f, x_new);

            // 4. 목표 감소치 계산 (출발점 함수값 + 관대함 (c) * 보폭 * 기울기)
            double target_val = f_x + c * alpha * dir_deriv;

            if (verbose) {
                std::cout << "  ↳ [Backtrack Iter " << std::setw(2) << iter
                          << "] alpha: " << std::fixed << std::setprecision(6) << alpha
                          << " | f_new: " << f_new << " | Target: " << target_val << "\n";
            }

            // 5. Armijo 조건 검사 (충분히 감소했는가?)
            if (f_new <= target_val) {
                if (verbose) std::cout << "  ↳ [Accepted] Armijo condition satisfied!\n";
                return alpha;
            }

            // 6. 실패 시 보폭 축소 (Backtracking)
            alpha *= p;

            // 무한 루프 방지 Failsafe (컴퓨터 정밀도 한계)
            if (alpha < 1e-10) {
                if (verbose) std::cout << "  ↳ [Failsafe] Alpha reached minimum limit.\n";
                return alpha;
            }
        }
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_