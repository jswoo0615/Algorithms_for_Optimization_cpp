#ifndef OPTIMIZATION_CROSS_ENTROPY_METHOD_HPP_
#define OPTIMIZATION_CROSS_ENTROPY_METHOD_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
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
 * @brief Cross-Entropy Method (교차 엔트로피 방법)
 * @note 다수의 샘플 대신 상위 샘플들로 확률 분포를 업데이트하며 전역 최적해 찾는 방법
 * ASPICE 검증을 위한 시드 고정 (Deterministic) 및 Reparameterization 최적화 적용
 */
class CrossEntropy {
   private:
    template <size_t N>
    struct Sample {
        alignas(64) std::array<double, N> x;
        double y;
        bool operator<(const Sample& other) const noexcept { return y < other.y; }
    };

   public:
    CrossEntropy() = delete;
    template <size_t N, size_t M = 100, size_t M_ELITE = 10, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> mu_init,
                                                          std::array<double, N> sigma_sq_init,
                                                          size_t max_iter = 100, double tol = 1e-5,
                                                          uint32_t seed = 12345,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");
        static_assert(M >= M_ELITE, "Total samples M must be greater than M_ELITE");

        auto start_clock = std::chrono::high_resolution_clock::now();

        alignas(64) std::array<double, N> mu = mu_init;
        alignas(64) std::array<double, N> sigma_sq = sigma_sq_init;

        // 정적 할당을 통한 O(M) 스택 메모리 사용 (힙 할당 원천 차단)
        static thread_local std::array<Sample<N>, M> population;

        // 오프라인 튜닝 시 재현성 (REproductibility)을 보장하기 위한 시드 주입
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        // 단일 표준 정규 분포 객체 재사용 (성능 최적화)
        std::normal_distribution<double> standard_normal(0.0, 1.0);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🎯 Cross-Entropy Method Started (Seed: " << seed << ")\n";
            std::cout << "========================================================\n";
        }

        size_t iter = 0;
        for (iter = 1; iter <= max_iter; ++iter) {
            // 1. Reparameterization Trick을 이용한 O(M*N) 샘플링
            for (size_t p = 0; p < M; ++p) {
                double z = standard_normal(gen);
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    // x_i = mu_i + std_dex_i * z (FMA 가속)
                    population[p].x[i] = std::fma(std::sqrt(sigma_sq[i]), z, mu[i]);
                }
                population[p].y = f(population[p].x);
            }

            // 2. 전체 정렬 대신 Partial Sort 사용으로 O(M log M_ELITE) 최적화
            std::partial_sort(population.begin(), population.begin() + M_ELITE, population.end());

            alignas(64) std::array<double, N> new_mu = {0.0};
            alignas(64) std::array<double, N> new_sigma_sq = {0.0};

            // 3.1. 엘리트 샘플 기반 새로운 평균 (mu) 계산
            for (size_t e = 0; e < M_ELITE; ++e) {
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    new_mu[i] += population[e].x[i];
                }
            }

#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                new_mu[i] /= static_cast<double>(M_ELITE);
            }

            // 3.2 대각 공분산 (Diagonal Covariance) 행렬 업데이트
            double max_variance = 0.0;
            for (size_t e = 0; e < M_ELITE; ++e) {
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    double diff = population[e].x[i] - new_mu[i];
                    new_sigma_sq[i] = std::fma(diff, diff, new_sigma_sq[i]);
                }
            }

            for (size_t i = 0; i < N; ++i) {
                new_sigma_sq[i] /= static_cast<double>(M_ELITE);
                if (new_sigma_sq[i] > max_variance) {
                    max_variance = new_sigma_sq[i];
                }
            }

            mu = new_mu;
            sigma_sq = new_sigma_sq;

            // population[0].y로 최 상위 1위의 값 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(6) << population[0].y
                          << " | Max Var: " << max_variance << "\n";
            }

            // 4. 종료 조건: 분산이 붕괴(Collapse)하여 한 점으로 수렴했는가?
            if (max_variance < tol) {
                if (verbose)
                    std::cout << "  ↳ ✅ Converged (Variance collapsed) at Iteration: " << iter
                              << "\n";
                break;
            }
        }
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << mu[0];
            if constexpr (N > 1) std::cout << ", " << mu[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        return {mu, population[0].y, iter, duration.count()};
    }
};
}  // namespace Optimization

namespace Optimization {}  // namespace Optimization

#endif  // OPTIMIZATION_CROSS_ENTROPY_METHOD_HPP_