#ifndef OPTIMIZATION_CONJUGATE_GRADIENT_HPP_
#define OPTIMIZATION_CONJUGATE_GRADIENT_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/LineSearch.hpp"

namespace Optimization {
class ConjugateGradient {
   public:
    // =====================================================================
    // Algorithm 5.1: Conjugate Gradient (Fletcher-Reeves Method)
    // alpha, beta를 사람이 정하지 않고 스스로 수학적 최적값을 찾아 돌파합니다.
    // =====================================================================
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, size_t max_iter = 1000,
                                          double tol = 1e-4, bool verbose = false) {
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " ⚔️ Conjugate Gradient (Fletcher-Reeves) Started\n";
            std::cout << "========================================================\n";
        }

        double f_x;
        std::array<double, N> g;
        AutoDiff::value_and_gradient<N>(f, x, f_x, g);

        // 첫 번째 방향은 순수 경사 하강법과 동일하게 역방향으로 시작
        std::array<double, N> d;
        for (size_t i = 0; i < N; ++i) {
            d[i] = -g[i];
        }

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            // 1. 현재 기울기의 크기 (Norm) 계산 및 수렴 확인
            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g[i] * g[i];
            }

            if (std::sqrt(g_norm_sq) < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 2. Exact Line Search를 통해 완벽한 보폭 (alpha) 스스로 탐색
            // Bracket을 머넞 잡고, Bisection이 0.0이 되는 지점을 찾습니다
            auto bracket = LineSearch::bracket_sign_change<N>(f, x, d, 0.0, 0.001, 2.0, false);
            double alpha =
                LineSearch::bisection<N>(f, x, d, bracket.first, bracket.second, 1e-4, false);

            // 3. 위치 업데이트
            for (size_t i = 0; i < N; ++i) {
                x[i] += alpha * d[i];
            }

            // 4. 새로운 위치에서 기울기 계산
            std::array<double, N> g_new;
            AutoDiff::value_and_gradient<N>(f, x, f_x, g_new);

            // 5. Fletcher-Reeves 공식을 이용한 beta 비율 계산
            double g_new_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_new_norm_sq += g_new[i] * g_new[i];
            }

            double beta = g_new_norm_sq / g_norm_sq;

            // 6. 새로운 공액 방향 (Conjugate Direction) 설정
            for (size_t i = 0; i < N; ++i) {
                d[i] = -g_new[i] + beta * d[i];
            }

            // 상태 업데이트 (g_new를 다음 루프의 g로)
            g = g_new;

            if (verbose && iter % 10 == 0) {
                std::cout << "[Iter " << std::setw(3) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << "\n";
            }
        }
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0] << ", " << x[1] << " ...]\n";
            std::cout << "========================================================\n";
        }
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_CONJUGATE_GRADIENT_HPP_