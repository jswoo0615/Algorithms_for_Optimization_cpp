#ifndef OPTIMIZATION_NELDER_MEAD_HPP_
#define OPTIMIZATION_NELDER_MEAD_HPP_

#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
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
     * @brief Nelder-Mead Simplex Method (넬더-미드 심플렉스 방법)
     * @note 미분 정보 없이 N + 1개의 정점으로 이루어진 아메바 (Amoeba) 형태를 굴리며 최적화합니다.
     * MISRA C++ 준수. O(1) 동적 할당 제로 및 SIMD/FMA 가속 적용
     */
    class NelderMead {
        private:
            template <size_t N>
            struct Vertex {
                alignas(64) std::array<double, N> x;
                double f_val;
                bool operator<(const Vertex& other) const noexcept {
                    return f_val < other.f_val;
                }
            };
        public:
            NelderMead() = delete;

            template <size_t N, typename Func>
            [[nodiscard]] static OptimizationResultND<N> optimize(
                Func f, std::array<double, N> x_start,
                double step = 1.0, double tol = 1e-6,
                size_t max_iter = 2000, bool verbose = false) noexcept {
                static_assert(N > 0, "Dimension N must be greater than 0");
                auto start_clock = std::chrono::high_resolution_clock::now();

                // Nelder-Mead Hyperparamter
                const double alpha = 1.0;   // 반사 (Reflection)
                const double gamma = 2.0;   // 확장 (Expansion)
                const double rho = 0.5;     // 수축 (Contraction)
                const double sigma = 0.5;   // 축소 (Shrinkage)
                const double tol_sq = tol * tol;    // [성능 최적화] 분산 비교용

                // 1. 초기 심플렉스 (Simplex) 생성 (동적 할당 없음)
                std::array<Vertex<N>, N + 1> simplex = {};
                simplex[0].x = x_start;
                simplex[0].f_val = f(simplex[0].x);

                for (size_t i = 0; i < N; ++i) {
                    simplex[i + 1].x = x_start;
                    simplex[i + 1].x[i] += step;            // 각 축 방향으로 step만큼 직교 이동하여 정점 생성
                    simplex[i + 1].f_val = f(simplex[i + 1].x);
                }

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🦠 Nelder-Mead Simplex (The Amoeba) Started\n";
                    std::cout << "========================================================\n";
                }

                size_t iter = 0;
                for (iter = 1; iter <= max_iter; ++iter) {
                    // 2. 정점 정렬 (0: Best, N: Worst)
                    std::sort(simplex.begin(), simplex.end());

                    // 3. 수렴 조건 검사 (분산이 tol_sq 미만인지 확인하여 sqrt 연산 생략)
                    double mean_y = 0.0;
                    #pragma omp simd
                    for (size_t i = 0; i <= N; ++i) {
                        mean_y += simplex[i].f_val;
                    }
                    mean_y /= (N + 1);

                    double var_y = 0.0;
                    #pragma omp simd
                    for (size_t i = 0; i <= N; ++i) {
                        double diff = simplex[i].f_val - mean_y;
                        var_y = std::fma(diff, diff, var_y);
                    }
                    var_y /= (N + 1);

                    if (var_y < tol_sq) {
                        if (verbose) 
                            std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                        break;
                    }

                    // 4. Centroid (최악의 점 simplex[N]을 제외한 무게중심)
                    alignas(64) std::array<double, N> x_o = {0.0};
                    for (size_t i = 0; i < N; ++i) { // i < N (최악점 제외)
                        #pragma omp simd
                        for (size_t j = 0; j < N; ++j) {
                            x_o[j] += simplex[i].x[j];
                        }
                    }
                    #pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        x_o[j] /= N;
                    }

                    // 5. 반사 (Reflection)
                    alignas(64) std::array<double, N> x_r;
                    #pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        x_r[j] = std::fma(alpha, x_o[j] - simplex[N].x[j], x_o[j]);
                    }
                    double f_r = f(x_r);

                    if (f_r >= simplex[0].f_val && f_r < simplex[N - 1].f_val) {
                        simplex[N].x = x_r;
                        simplex[N].f_val = f_r;
                    } else if (f_r < simplex[0].f_val) {
                        // 6. 확장 (Expansion)
                        alignas(64) std::array<double, N> x_e;
                        #pragma omp simd
                        for (size_t j = 0; j < N; ++j) {
                            x_e[j] = std::fma(gamma, x_r[j] - x_o[j], x_o[j]);
                        }
                        double f_e = f(x_e);

                        if (f_e < f_r) {
                            simplex[N].x = x_e;
                            simplex[N].f_val = f_e;
                        } else {
                            simplex[N].x = x_r;
                            simplex[N].f_val = f_r;
                        }
                    } else {
                        // 7. 수축 (Contraction)
                        bool do_shrink = false;

                        if (f_r < simplex[N].f_val) {
                            // 외부 수축 (Outside Contraction)
                            alignas(64) std::array<double, N> x_c;
                            #pragma omp simd
                            for (size_t j = 0; j < N; ++j) {
                                x_c[j] = std::fma(rho, x_r[j] - x_o[j], x_o[j]);
                            }
                            double f_c = f(x_c);

                            if (f_c <= f_r) {
                                simplex[N].x = x_c;
                                simplex[N].f_val = f_c;
                            } else {
                                do_shrink = true;
                            }
                        } else {
                            // 내부 수축 (Inside Contraction)
                            alignas(64) std::array<double, N> x_c;
                            #pragma omp simd
                            for (size_t j = 0; j < N; ++j) {
                                x_c[j] = std::fma(rho, simplex[N].x[j] - x_o[j], x_o[j]);
                            }
                            double f_c = f(x_c);

                            if (f_c < simplex[N].f_val) {
                                simplex[N].x = x_c;
                                simplex[N].f_val = f_c;
                            } else {
                                do_shrink = true;
                            }
                        }

                        // 8. 축소 (Shrinkage) - 모든 점을 최우수 점 (simplex[0])을 향해 당김
                        if (do_shrink) {
                            for (size_t i = 1; i <= N; ++i) {
                                #pragma omp simd
                                for (size_t j = 0; j < N; ++j) {
                                    simplex[i].x[j] = std::fma(sigma, simplex[i].x[j] - simplex[0].x[j], simplex[0].x[j]);
                                }
                                simplex[i].f_val = f(simplex[i].x);
                            }
                        }
                    }
                    if (verbose && (iter % 50 == 0 || iter == 1)) {
                        std::cout << "[Iter " << std::setw(3) << iter 
                                << "] Best f(x): " << std::fixed << std::setprecision(8) << simplex[0].f_val 
                                << " | Variance: " << var_y << "\n";
                    }
                }
                
                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);
                
                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🏁 Final Optimal Point: [" << simplex[0].x[0];
                    if constexpr (N > 1) 
                        std::cout << ", " << simplex[0].x[1];
                    if constexpr (N > 2) 
                        std::cout << ", ...";
                    std::cout << "]\n========================================================\n";
                }

                return {simplex[0].x, simplex[0].f_val, iter, duration.count()};
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_NELDER_MEAD_HPP_
