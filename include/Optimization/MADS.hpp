#ifndef OPTIMIZATION_MADS_HPP_
#define OPTIMIZATION_MADS_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

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
 * @brief Mesh Adaptive Direct Search (MADS)
 * @note 무작위로 생성되는 양의 생성 집합 (Positive Spanning Set)을 사용하여 미분 없이 탐색
 * [주의] 확률론적 동작을 포함하므로 실시간 (RT) 제어 루프 밖의 오프라인 환경에서 사용해야 합니다.
 */
class MeshAdaptiveDirectSearch {
   private:
    // O(1) 정적 배열 및 난수 참조 전달을 통한 지연 없는 랜덤 기저 (Basis) 생성
    template <size_t N, typename Generator>
    [[nodiscard]] static std::array<std::array<double, N>, N + 1> generate_positive_spanning_set(
        double alpha, Generator& gen) {
        alignas(64) std::array<std::array<double, N>, N + 1> D = {0.0};
        alignas(64) std::array<std::array<double, N>, N> L = {0.0};

        // 1. 델타 (Delta) 크기 결정
        int delta = static_cast<int>(std::round(1.0 / std::sqrt(alpha)));
        if (delta < 1) {
            delta = 1;
        }
        std::uniform_int_distribution<int> sign_dist(0, 1);
        std::uniform_int_distribution<int> lower_dist(-delta + 1, delta - 1);

        // 2. 하삼각 행렬 (Lower Triangular Matrix) L 생성
        for (size_t i = 0; i < N; ++i) {
            L[i][i] = (sign_dist(gen) == 1) ? delta : -delta;  // 대각 성분
            for (size_t j = 0; j < i; ++j) {
                L[i][j] = lower_dist(gen);  // 하삼각 성분
            }
        }

        // 3. 행과 열을 랜덤하게 섞기 (Permutation)
        std::array<size_t, N> row_p, col_p;
        std::iota(row_p.begin(), row_p.end(), 0);  // 0, 1, 2, ... 할당
        std::iota(col_p.begin(), col_p.end(), 0);
        std::shuffle(row_p.begin(), row_p.end(), gen);
        std::shuffle(col_p.begin(), col_p.end(), gen);

        // 4. 방향 행렬 D 계산 및 N + 1 번째 방향 (Negative Sum) 추가
        for (size_t d_idx = 0; d_idx < N; ++d_idx) {
            for (size_t dim = 0; dim < N; ++dim) {
                double val = L[row_p[dim]][col_p[d_idx]];
                D[d_idx][dim] = val;
                D[N][dim] -= val;  // 나머지 벡터들의 역벡터 합 (Positive Spanning Set 조건 완수)
            }
        }
        return D;
    }

   public:
    MeshAdaptiveDirectSearch() = delete;  // 인스턴스화 방지

    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          double tol = 1e-5,
                                                          size_t max_iter = 10000,
                                                          bool verbose = false) {
        static_assert(N > 0, "Dimension N must be greater than 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        double alpha = 1.0;
        alignas(64) std::array<double, N> x = x_init;
        double y = f(x);

        // [핵심] 정적 thread_local 난수 생성기로 지연 (Latency) 원천 차단
        static thread_local std::mt19937 gen(std::random_device{}());

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🕸️ MADS (Mesh Adaptive Direct Search) Started \n";
            std::cout << "========================================================\n";
        }

        size_t iter = 0;
        while (alpha > tol && iter < max_iter) {
            iter++;
            bool improved = false;

            // N + 1개의 무작위 탐색 방향 세트 획득
            auto D = generate_positive_spanning_set<N>(alpha, gen);

            // [기회주의적 탐색]
            for (size_t i = 0; i <= N; ++i) {
                alignas(64) std::array<double, N> x_prime = {0.0};
#pragma omp simd
                for (size_t j = 0; j < N; ++j) {
                    x_prime[j] = std::fma(alpha, D[i][j], x[j]);
                }

                double y_prime = f(x_prime);

                if (y_prime < y) {
                    x = x_prime;
                    y = y_prime;
                    improved = true;

                    // [가속 스텝] 해당 방향으로 성과가 좋으면 3배 (3 * alpha) 반영
                    alignas(64) std::array<double, N> x_exp = {0.0};
#pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        x_exp[j] = std::fma(3.0 * alpha, D[i][j], x[j]);
                    }

                    double y_exp = f(x_exp);

                    if (y_exp < y) {
                        x = x_exp;
                        y = y_exp;
                    }
                    break;  // 하나라도 개선되면 즉시 내부 루프 탈출
                }
            }

            // 성과에 따른 Mesh 크기 및 보폭 조절
            if (improved) {
                alpha = std::min(4.0 * alpha, 1.0);
            } else {
                alpha /= 4.0;
            }

            if (verbose && (!improved || iter % 10 == 0)) {
                std::cout << "[Iter " << std::setw(4) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << y << " | alpha: " << alpha << "\n";
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

        return {x, y, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_MADS_HPP_