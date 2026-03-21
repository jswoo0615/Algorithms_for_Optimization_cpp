#ifndef OPTIMIZATION_RMPROPS_HPP_
#define OPTIMIZATION_RMPROPS_HPP_

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
  /**
   * @brief RMSProp (Root Mean Square Propagation) 최적화 알고리즘을 구현한 클래스
   * @note 지수 이동 평균 (EMA)을 사용하여 AdaGrad의 학습률 소멸 문제를 해결
   * 동적 할당 원천 차단 및 컴파일 타임 최적화 적용
   */

  class RMSProp {
    public:
      template <size_t N, typename Func>
      [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.01, double decay = 0.9,
                                                          double epsilon = 1e-8, size_t max_iter = 15000, double tol = 1e-4, bool verbose = false) {
        // 컴파일 타임 차원 검증
        static_assert(N > 0, "Dimension N must be greater than 0");

        // 과거 기울기 제곤의 지수 이동 평균 (G)
        std::array<double, N> G = {0.0};

        // [성능 최적화] 매 이터레이션마다 sqrt를 호출하는 비용 제거하기 위해 tol을 미리 제곱
        const double tol_sq = tol * tol;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " ⚡ RMSProp Optimizer Started (alpha=" << alpha << ", decay=" << decay << ")\n";
            std::cout << "========================================================\n";
        }

        for (size_t iter = 1; iter <= max_iter; ++iter) {
          double f_x = 0.0;
          std::array<double, N> g = {0.0};

          // 1. O(1) 할당 Auto Diff 호출
          AutoDiff::value_and_gradient<N> (f, x, f_x, g);

          // 2. 기울기 L2-Norm 제곱 계산
          double g_norm_sq = 0.0;
          for (size_t i = 0; i < N; ++i) {
            g_norm_sq += g[i] * g[i];
          }

          // [최적화 포인트] g_norm_sq < tol^2 사용
          if (g_norm_sq < tol_sq) {
            if (verbose) {
                std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
            }
            break;
          }

          // 3. RMSProp 파라미터 업데이트
          for (size_t i = 0; i < N; ++i) {
            // 과거 기억 (G)은 decay만큼 남기고, 새로운 기울기 제곱을 더함 (EMA)
            G[i] = decay * G[i] + (1.0 - decay) * g[i] * g[i];

            x[i] -= (alpha / (std::sqrt(G[i]) + epsilon)) * g[i];
          }

          // 런타임 분기 예측 최적화를 위한 조건 배열
          if (verbose && (iter % 1000 == 0)) {
              std::cout << "[Iter " << std::setw(5) << iter << "] f(x): "
                        << std::fixed << std::setprecision(6) << f_x
                        << " | ||g||: " << std::sqrt(g_norm_sq) << "\n"; // 출력 시에만 sqrt 연산 수행
          }
        }
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        return x;
      }
  };
} // namespace Optimization

#endif // OPTIMIZATION_RMPROPS_HPP_