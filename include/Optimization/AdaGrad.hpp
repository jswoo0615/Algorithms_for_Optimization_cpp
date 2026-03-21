#ifndef OPTIMIZATION_ADAGRAD_HPP_
#define OPTIMIZATION_ADAGRAD_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
/**
 * @brief AdaGrad (Adaptive Gradient) 최적화 알고리즘을 구현한 클래스
 * @note MISRA C++ 및 실시간 제어 (RT) 표준 준수 : 동적 할당 배제, In-place 연산
 */
class AdaGrad {
   public:
    AdaGrad() = delete;  // 인스턴스화 방지

    // ==============================================================================
    // Algorithm 5.4 : AdaGrad
    // 차원별로 과거 기울기의 제곱을 누적하여 맞춤형 학습률 제공
    // ==============================================================================
    template <size_t N, typename Func>
    [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x,
                                                        double alpha = 0.1, double epsilon = 1e-8,
                                                        size_t max_iter = 15000, double tol = 1e-4,
                                                        bool verbose = false) {
        // 0차원 배열 주입을 컴파일 타임에 원천 차단
        static_assert(N > 0, "Dimension N must be greater than 0.");
        // 과거 기울기 제곱 누적합 (동적 할당 없는 스택 메모리)
        // 제어 이론 관례에 따라 Velocity(v)와 구분하기 위해 G 사용
        std::array<double, N> G = {0.0};

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🎯 AdaGrad Optimizer Started (alpha=" << alpha << ", eps=" << epsilon
                      << ")\n";
            std::cout << "========================================================\n";
        }

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x = 0.0;
            std::array<double, N> g = {0.0};

            // 1. O(1) 할당 초고속 Auto Diff 계산
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 2. 기울기 l2-Norm 계산 및 조기 종료 검증
            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g[i] * g[i];
            }

            const double g_norm = std::sqrt(g_norm_sq);
            if (g_norm < epsilon) {
                if (verbose) {
                    std::cout << "✅ Convergence achieved at iteration " << iter
                              << " with gradient norm " << g_norm << ".\n";
                }
                break;
            }

            // 3. AdaGrad 파라미터 업데이트
            for (size_t i = 0; i < N; ++i) {
                G[i] += g[i] * g[i];  // 과거 기울기 제곱 누적
                // [수치적 안정성] epsilon을 루트 밖에서 다하여 발산 (Divergence) 방지
                // [실시간성] -= In-place 연산자로 캐시 히트율 극대화
                x[i] -= (alpha / (std::sqrt(G[i]) + epsilon)) * g[i];
            }

            // 런타임 분기 예측 (Branch Prediction) 최적화를 위해 % 연산을 최소화
            if (verbose && (iter % 1000 == 0)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << g_norm << "\n";
            }
        }
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            // C++17 constexpr if를 활용하여 N차원에 따른 안전한 출력 보장
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_ADAGRAD_HPP_
