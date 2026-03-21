#ifndef OPTIMIZATION_ADAM_HPP_
#define OPTIMIZATION_ADAM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
/**
 * @brief Adam (Adaptive Moment Estimation) 최적화 알고리즘을 구현한 클래스
 * @note 1차 모멘트와 2차 모멘트를 사용하여 학습률을 동적으로 조정
 * 실시간 (RT) 환경을 위해 std::pow 호출을 제거하고 누적 곱 (Running product) 적용
 */
class Adam {
   public:
    Adam() = delete;  // 인스턴스화 방지
    template <size_t N, typename Func>
    [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x,
                                                        double alpha = 0.1, double beta1 = 0.9,
                                                        double beta2 = 0.999, double epsilon = 1e-8,
                                                        size_t max_iter = 15000, double tol = 1e-4,
                                                        bool verbose = false) {
        static_assert(N > 0, "Dimension N must be greater than 0");
        std::array<double, N> m = {0.0};  // 1차 모멘트 초기화
        std::array<double, N> v = {0.0};  // 2차 모멘트 초기화

        // [성능 최적화] 매 이터레이션마다 sqrt를 호출하기 않기 위함
        const double tol_sq = tol * tol;

        double beta1_t = beta1;
        double beta2_t = beta2;
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 👑 Adam Optimizer Started (alpha=" << alpha << ")\n";
            std::cout << "========================================================\n";
        }
        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x = 0.0;
            std::array<double, N> g = {0.0};
            AutoDiff::value_and_gradient(f, x, f_x, g);

            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g[i] * g[i];
            }

            if (g_norm_sq < tol_sq) {
                if (verbose) {
                    std::cout << "✅ Convergence achieved at iteration " << iter
                              << " (f(x) = " << f_x << ", ||g||^2 = " << g_norm_sq << ")\n";
                }
                break;
            }

            // [성능 최적화] 차원 (N) 루프 진입 전, 공통 분모 (Bias correction)를 미리 계산
            const double one_minus_beta1_t = 1.0 - beta1_t;
            const double one_minus_beta2_t = 1.0 - beta2_t;

            for (size_t i = 0; i < N; ++i) {
                // 1. 모멘텀 업데이트
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];

                // 2. 편향 보정 (수만 번의 std::pow를 단 2번의 나눗셈을 압축)
                const double m_hat = m[i] / one_minus_beta1_t;
                const double v_hat = v[i] / one_minus_beta2_t;

                // 3. 파라미터 업데이트 (In-place)
                x[i] -= (alpha / (std::sqrt(v_hat) + epsilon)) * m_hat;
            }

            // 다음 이터레이션을 위해 누적 곱 업데이트 (O(1) 연산)
            beta1_t *= beta1;
            beta2_t *= beta2;

            if (verbose && (iter % 1000 == 0)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << std::sqrt(g_norm_sq)
                          << "\n";
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
}  // namespace Optimization

#endif  // OPTIMIZATION_ADAM_HPP_
