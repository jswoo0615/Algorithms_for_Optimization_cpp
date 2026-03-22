#ifndef OPTIMIZATION_HOOKE_JEEVES_HPP_
#define OPTIMIZATION_HOOKE_JEEVES_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt;
    double f_opt;
    size_t iterations;
    long long elapsed_ns;
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {
/**
 * @brief Hooke-Jeeves Pattern Search
 * @note 미분 (Gradient) 없이 탐색 (Exploratory)과 패턴 이동 (Pattern)을 결합하여 가속
 * 동적 할당 차단 및 캐시 정렬 (alignas(64)) 적용
 */
class HookeJeeves {
   private:
    /**
     * @brief 더듬기 (Exploratory Move)
     * @details 각 차원 축을 순차적으로 탐색하여 개선될 경우 즉시 위치를 확정합니다.
     */
    template <size_t N, typename Func>
    static constexpr std::array<double, N> explore(Func f, std::array<double, N> x, double alpha,
                                                   double& f_best) noexcept {
        for (size_t i = 0; i < N; ++i) {
            // 양 (+)의 방향 탐색
            x[i] += alpha;
            double f_plus = f(x);
            if (f_plus < f_best) {
                f_best = f_plus;
                continue;
            }

            // 음 (-)의 방향 탐색 (-2.0 * alpha로 반대편 이동)
            x[i] -= 2.0 * alpha;
            double f_minus = f(x);
            if (f_minus < f_best) {
                f_best = f_minus;
                continue;
            }

            // 양쪽 모두 실패 시 원래 위치 복구
            x[i] += alpha;
        }
        return x;
    }

   public:
    HookeJeeves() = delete;
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_start,
                                                          double alpha = 0.5, double shrink = 0.5,
                                                          double tol = 1e-6, size_t max_iter = 2000,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🐸 Hooke-Jeeves Pattern Search Started\n";
            std::cout << "========================================================\n";
        }

        alignas(64) std::array<double, N> x = x_start;
        double f_best = f(x);
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            double f_prev = f_best;

            // 1. 현재 위치에서 탐색 이동 (더듬기)
            alignas(64) std::array<double, N> x_new = explore<N>(f, x, alpha, f_best);

            // 부동 소숫점 배열 비교 대신 함수값 개선 여부로 성공 판별 (강건성 확보)
            if (f_best < f_prev) {
                // 2. 패턴 이동 (Pattern Move)  - FMA 가속 모멘텀
                alignas(64) std::array<double, N> x_pattern = {0.0};
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    // x_new + (x_new - x) = 2.0 * x_new - x
                    x_pattern[i] = std::fma(2.0, x_new[i], -x[i]);
                }

                double f_pattern = f(x_pattern);
                alignas(64) std::array<double, N> x_pattern_new =
                    explore<N>(f, x_pattern, alpha, f_pattern);

                // 점프 후 탐색 결과가 기존 최적값보다 좋으면 채택, 아니면 탐색만 한 결과로 롤백
                if (f_pattern < f_best) {
                    x = x_pattern_new;
                    f_best = f_pattern;
                } else {
                    x = x_new;
                }
            } else {
                // 3. 수축 (Shrink) - 전방위 탐색 실패 시 보폭 축소
                alpha *= shrink;
                if (alpha < tol) {
                    if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                    break;
                }
            }

            if (verbose && (iter % 100 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(4) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_best << " | Step(alpha): " << alpha << "\n";
            }
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        return {x, f_best, iter, duration.count()};
    }
};
}  // namespace Optimization
#endif  // OPTIMIZATION_HOOKE_JEEVES_HPP_