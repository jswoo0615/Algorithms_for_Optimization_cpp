#ifndef OPTIMIZATION_GRADIENT_DESCENT_HPP_
#define OPTIMIZATION_GRADIENT_DESCENT_HPP_

#include <array>
#include <functional>
#include <vector>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/StrongBacktrackingLineSearch.hpp"

namespace Optimization {
class GradientDescent {
   public:
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, double tol = 1e-6,
                                          int max_iter = 1000, bool verbose = false) {
        if (verbose) std::cout << "🚀 Gradient Descent Started...\n";

        for (int i = 0; i < max_iter; ++i) {
            // 1. 현재 위치의 함수값과 기울기 (Gradient) 계산
            double f_val;
            std::array<double, N> grad;
            AutoDiff::value_and_gradient<N>(f, x, f_val, grad);

            // 2. 수렴 확인 (기울기가 거의 0이면 바닥에 도착)
            double g_norm = 0.0;
            for (double g : grad) {
                g_norm += g * g;
            }
            g_norm = std::sqrt(g_norm);

            if (verbose) {
                std::cout << "[Iter " << i << "] f(x) = " << f_val << " | |grad| = " << g_norm
                          << "\n";
            }

            if (g_norm < tol) break;

            // 3. 하강 방향 설정 (기울기의 반대 방향)
            std::array<double, N> direction;
            for (size_t j = 0; j < N; ++j) direction[j] = -grad[j];

            // 4. 보폭 (alpha) 결정 (Strong Wolfe 사용)
            double alpha = StrongBacktrackingLineSearch::search<N>(f, x, direction);

            // 5. 위치 업데이트 : x = x + alpha * d
            for (size_t j = 0; j < N; ++j) {
                x[j] += alpha * direction[j];
            }
        }
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_GRADIENT_DESCENT_HPP_