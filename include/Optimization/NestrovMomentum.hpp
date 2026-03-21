#ifndef OPTIMIZATION_NESTROV_MOMENTUM_HPP_
#define OPTIMIZATION_NESTROV_MOMENTUM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
class NestrovMomentum {
   public:
    // ==============================================================
    // Algorithm 5.3 : Nestrov Momentum
    // 미래의 위치 (Look-ahead)를 먼저 예측하고 그곳의 기울기를 이용해 업데이트합니다
    // ==============================================================
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.001,
                                          double beta = 0.9, size_t max_iter = 15000,
                                          double tol = 1e-4, bool verbose = false) {
        std::array<double, N> v = {0.0};
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🔮 Nesterov Momentum Started (alpha=" << alpha << ", beta=" << beta
                      << ")\n";
            std::cout << "========================================================\n";
        }
        for (size_t iter = 1; iter <= max_iter; ++iter) {
            // 1. 미래의 위치 (Look-ahead point) 계산 : 먼저 관성대로 가본다
            std::array<double, N> x_lookahead;
            for (size_t i = 0; i < N; ++i) {
                x_lookahead[i] = x[i] + beta * v[i];
            }

            // 2. 미래 위치에서의 기울기 (g_lookahead) 획득
            double f_lookahead;
            std::array<double, N> g_lookahead;
            AutoDiff::value_and_gradient<N>(f, x_lookahead, f_lookahead, g_lookahead);

            // 종료 조건 검사 (미래 위치의 기울기 기준)
            double g_norm = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm += g_lookahead[i] * g_lookahead[i];
            }
            g_norm = std::sqrt(g_norm);

            if (g_norm < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. 네스테로프 업데이트 (미래의 기울기를 반영하여 속도와 위치 갱신)
            for (size_t i = 0; i < N; ++i) {
                v[i] = beta * v[i] - alpha * g_lookahead[i];
                x[i] = x[i] + v[i];  // 실제 위치 업데이트
            }
        }

        if (verbose) {
            std::cout << "========================================================\n";
            // 10차원이니까 다 출력하면 기니, 앞의 두 개만 확인합시다!
            std::cout << " 🏁 Final Optimal Point: [" << x[0] << ", " << x[1] << "]\n";
            std::cout << "========================================================\n";
        }
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_NESTROV_MOMENTUM_HPP_