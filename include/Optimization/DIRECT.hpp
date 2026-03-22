#ifndef OPTIMIZATION_DIRECT_HPP_
#define OPTIMIZATION_DIRECT_HPP_

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
     * @brief Divided Rectangle (DIRECT) 전역 최적화 알고리즘
     * @note 립시츠 상수 없이 단위 하이퍼큐브 공간을 분할하여 전역 최솟값을 탐색합니다.
     * MISRA C++ 준수. 동적 할당 (std::vector) 제로 및 스택 안전 (Static Pool) 설계
     */
    template <size_t N, size_t MAX_INTERVALS = 10000>
    class DIRECT {
        private:
            struct Interval {
                alignas(64) std::array<double, N> c;    // 중앙점 좌표 (정규화된 공간 [0, 1]^N)
                double y;                               // 함수값 f(c)
                std::array<size_t, N> depths;           // 차원별 분할 횟수
                bool active;                            // 메모리 풀 사용 여부

                [[nodiscard]] size_t min_depth() const noexcept {
                    size_t md = depths[0];
                    for (size_t i = 1; i < N; ++i) {
                        if (depths[i] < md) {
                            md = depths[i];
                        }
                    }
                    return md;
                }
            };

            [[nodiscard]] static constexpr std::array<double, N> denormalize(
                const std::array<double, N>& x,
                const std::array<double, N>& a,
                const std::array<double, N>& b) noexcept {
                std::array<double, N> real_x = {0.0};
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    // x[i] * (b[i] - a[i]) + a[i]
                    real_x[i] = std::fma(x[i], b[i] - a[i], a[i]);
                }
                return real_x;
            }
        public:
            DIRECT() = delete;

            template <typename Func>
            [[nodiscard]] static OptimizationResultND<N> optimize(
                Func f, std::array<double, N> a, std::array<double, N> b,
                double epsilon = 1e-4, size_t max_iter = 100, bool verbose = false) noexcept {
                static_assert(N > 0, "Dimension N must be greater than 0");
                auto start_clock = std::chrono::high_resolution_clock::now();

                // 정적 메모리 풀 (RT 환경 스택 오버플로우 방지 및 스레드 안정성 확보)
                static thread_local std::array<Interval, MAX_INTERVALS> pool;
                for (size_t i = 0; i < MAX_INTERVALS; ++i) {
                    pool[i].active = false;
                }
                size_t pool_size = 0;

                std::array<double, N> c_init = {0.0};
                for (size_t i = 0; i < N; ++i) {
                    c_init[i] = 0.5;
                }
                pool[0].c = c_init;
                pool[0].y = f(denormalize(c_init, a, b));

                for (size_t i = 0; i < N; ++i) {
                    pool[0].depths[i] = 0;
                }

                pool[0].active = true;
                pool_size = 1;

                double y_best = pool[0].y;
                alignas(64) std::array<double, N> c_best = pool[0].c;

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🌐 DIRECT (Divided Rectangles) Search Started \n";
                    std::cout << "========================================================\n";
                }

                // 동적 할당 std::vector를 대체하는 정적 인덱스 버퍼
                static thread_local std::array<size_t, MAX_INTERVALS> S;
                size_t iter = 0;
                for (iter = 1; iter <= max_iter; ++iter) {
                    size_t S_size = 0;
                    double min_y_val = y_best;

                    // 잠재적 최적 구간 (Potentially Optimal Intervals) 탐색
                    for (size_t i = 0; i < pool_size; ++i) {
                        if (pool[i].active && pool[i].y <= min_y_val + epsilon) {
                            S[S_size++] = i;
                        }
                    }

                    for (size_t s_idx = 0; s_idx < S_size; ++s_idx) {
                        size_t idx = S[s_idx];

                        if (pool_size + 2 * N > MAX_INTERVALS) {
                            break;
                        }

                        Interval& parent = pool[idx];
                        size_t d = parent.min_depth();

                        for (size_t dim = 0; dim < N; ++dim) {
                            if (parent.depths[dim] == d) {
                                double delta = std::pow(3.0, -static_cast<double>(d + 1));

                                Interval left = parent;
                                left.c[dim] -= delta;
                                left.depths[dim] += 1;
                                left.y = f(denormalize(left.c, a, b));

                                Interval right = parent;
                                right.c[dim] += delta;
                                right.depths[dim] += 1;
                                right.y = f(denormalize(right.c, a, b));

                                pool[pool_size++] = left;
                                pool[pool_size++] = right;
                                parent.depths[dim] += 1;

                                if (left.y < y_best) {
                                    y_best = left.y;
                                    c_best = left.c;
                                }
                                if (right.y < y_best) {
                                    y_best = right.y;
                                    c_best = right.c;
                                }
                            }
                        }
                    }
                    if (verbose && (iter % 10 == 0 || iter == 1)) {
                        std::cout << "[Iter " << std::setw(3) << iter 
                                << "] Best f(x): " << std::fixed << std::setprecision(6) << y_best 
                                << " | Active Intervals: " << pool_size << "\n";
                    }
                    
                    if (pool_size >= MAX_INTERVALS - 2 * N) {
                        if (verbose) 
                            std::cout << "  ↳ ⚠️ MAX_INTERVALS capacity reached!\n";
                        break;
                    }
                }

                std::array<double, N> final_x = denormalize(c_best, a, b);

                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🏁 Final Global Optimal Point: [" << final_x[0];
                    if constexpr (N > 1) std::cout << ", " << final_x[1];
                    if constexpr (N > 2) std::cout << ", ...";
                    std::cout << "]\n========================================================\n";
                }
                
                return {final_x, y_best, iter, duration.count()};
            }
    };
} // namespace Optimization
#endif // OPTIMIZATION_DIRECT_HPP_