#ifndef OPTIMIZATION_MOMENTUM_HPP_
#define OPTIMIZATION_MOMENTUM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
class Momentum {
   public:
    // =======================================================================
    // Algorithm 5.2 : Momentum (Heavy Ball Method)
    // - alpha : 학습률 (보폭, Learning Rate). 보통 고정된 값 사용
    // - beta : 관성 계수 (Momentum). 과거 속도를 얼마나 유지할지 (보통 0.9)
    // =======================================================================
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.001,
                                          double beta = 0.9, size_t max_iter = 15000,
                                          double tol = 1e-4, bool verbose = false) {
        std::array<double, N> v = {0.0};  // 초기 속도 (Velocity)는 0으로 시작
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🌪️ Momentum Optimizer Started (alpha=" << alpha << ", beta=" << beta
                      << ")\n";
            std::cout << "========================================================\n";
        }
        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x;
            std::array<double, N> g;

            // 1. 현재 위치의 함수값 및 기울기 평가
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 2. 기울기 노름 (Norm) 계산 및 종료 조건 확인
            double g_norm = 0.0;
            for (size_t i = 0; i < N; ++i) g_norm += g[i] * g[i];
            g_norm = std::sqrt(g_norm);

            if (g_norm < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. 모멘텀 업데이트
            for (size_t i = 0; i < N; ++i) {
                // 과거의 속도 (v)를 beta만큼 살리고, 현재 기울기 (g)를 alpha만큼 빼줌
                v[i] = beta * v[i] - alpha * g[i];

                // 새로운 위치로 이동
                x[i] = x[i] + v[i];
            }
            if (verbose && iter % 1000 == 0) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << g_norm << "\n";
            }
        }
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0] << ", " << x[1] << "]\n";
            std::cout << "========================================================\n";
        }
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_MOMENTUM_HPP_