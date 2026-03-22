#ifndef OPTIMIZATION_GENERALIZED_PATTERN_SEARCH_HPP_
#define OPTIMIZATION_GENERALIZED_PATTERN_SEARCH_HPP_

#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

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
} // namespace Optimization
#endif // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {
    /**
     * @brief Generalized Pattern Search (일반화된 패턴 탐색) 알고리즘
     * @note 기회주의적 탐색 및 동적 정렬 (Dynamic Ordering) 적용
     * MISRA C++ 준수 O(1) 정적 배열 시프트 및 FMA 가속
     */
    class GeneralizedPatternSearch {
        public:
            GeneralizedPatternSearch() = delete;

            // N : 상태 공간 차원 수, M : 방향 집합 (Positive Spanning Set)의 갯수
            template <size_t N, size_t M, typename Func>
            [[nodiscard]] static OptimizationResultND<N> optimize(
                Func f, std::array<double, N> x_init, std::array<std::array<double, N>, M> D, 
                double alpha = 1.0, double epsilon = 1e-5, double gamma = 0.5,
                size_t max_iter = 10000, bool verbose = false) noexcept {
                static_assert(N > 0, "Dimension N must be greater than 0");
                static_assert(M >= N + 1, "A positive spanning set must have at least N+1 directions");

                auto start_clock = std::chrono::high_resolution_clock::now();

                alignas(64) std::array<double, N> x = x_init;
                double y = f(x);
                size_t iter = 0;

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🕸️ Generalized Pattern Search Started (alpha0=" << alpha << ")\n";
                    std::cout << "========================================================\n";
                }

                while (alpha > epsilon && iter < max_iter) {
                    iter++;
                    bool improved = false;

                    for (size_t i = 0; i < M; ++i) {
                        alignas(64) std::array<double, N> x_prime = {0.0};

                        // 탐색 위치 계산 (SIMD + FMA)
                        #pragma omp simd
                        for (size_t j = 0; j < N; ++j) {
                            x_prime[j] = std::fma(alpha, D[i][j], x[j]);
                        }

                        double y_prime = f(x_prime);

                        // 기회주의적 (Opportunistic) 업데이트
                        if (y_prime < y) {
                            x = x_prime;
                            y = y_prime;
                            improved = true;

                            // 동적 정렬 (Dynamic Ordering) : 성공한 방향을 맨 앞으로 시프트
                            if (i > 0) {
                                alignas(64) std::array<double, N> best_d = D[i];
                                for (size_t k = i; k > 0; --k) {
                                    D[k] = D[k - 1];
                                }
                                D[0] = best_d;
                            }
                            break;
                        }
                    }

                    // M개의 방향 모두 실패 시 보폭 축소
                    if (!improved) {
                        alpha *= gamma;
                    }
                    if (verbose && (!improved || iter % 100 == 0)) {
                        std::cout << "[Iter " << std::setw(4) << iter 
                                << "] f(x): " << std::fixed << std::setprecision(6) << y 
                                << " | alpha: " << alpha << "\n";
                    }
                }

                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🏁 Final Optimal Point: [" << x[0];
                    if constexpr (N > 1) 
                        std::cout << ", " << x[1];
                    if constexpr (N > 2) 
                        std::cout << ", ...";
                    std::cout << "]\n========================================================\n";
                }
                return {x, y, iter, duration.count()};
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_GENERALIZED_PATTERN_SEARCH_HPP_