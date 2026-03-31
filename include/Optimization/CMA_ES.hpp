#ifndef OPTIMIZATION_CMA_ES_HPP_
#define OPTIMIZATION_CMA_ES_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// 결과 반환 구조체 유지
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
#endif

namespace Optimization {

/**
 * @brief CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
 * @note 목적 함수의 역-헤시안(Inverse Hessian)을 학습하듯 공분산 행렬을 적응시키는 전역 최적화
 * 기법. 정적 메모리(O(1)) 기반의 촐레스키 분해(Cholesky Decomposition) 엔진 내장.
 */
class CMA_ES {
   private:
    template <size_t N>
    struct Sample {
        alignas(64) std::array<double, N> x;
        alignas(64) std::array<double, N> z;  // 표준 정규 난수 벡터
        double y;
        bool operator<(const Sample& other) const noexcept { return y < other.y; }
    };

    // [내장 엔진] Cholesky 분해: A = L * L^T (A는 Positive Definite Symmetric 이어야 함)
    template <size_t N>
    [[nodiscard]] static bool cholesky_decomposition(
        const std::array<std::array<double, N>, N>& A,
        std::array<std::array<double, N>, N>& L) noexcept {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) sum += L[i][k] * L[j][k];
                if (i == j) {
                    double val = A[i][i] - sum;
                    if (val <= 0.0) return false;  // Not positive definite
                    L[i][j] = std::sqrt(val);
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        return true;
    }

   public:
    CMA_ES() = delete;

    template <size_t N, size_t M = 10, size_t M_ELITE = 5, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> mu_init,
                                                          double sigma = 1.0,
                                                          size_t max_iter = 1000, double tol = 1e-6,
                                                          uint32_t seed = 12345,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension must be > 0");
        static_assert(M >= M_ELITE && M_ELITE > 0, "Invalid sample size");

        auto start_clock = std::chrono::high_resolution_clock::now();

        // 알고리즘 파라미터 사전 연산 (Zero-overhead in loop)
        std::array<double, M_ELITE> w = {0.0};
        double sum_w = 0.0, sum_wq = 0.0;
        for (size_t i = 0; i < M_ELITE; ++i) {
            w[i] = std::log((M + 1.0) / 2.0) - std::log(i + 1.0);
            sum_w += w[i];
        }
        for (size_t i = 0; i < M_ELITE; ++i) {
            w[i] /= sum_w;
            sum_wq += w[i] * w[i];
        }
        const double mu_eff = 1.0 / sum_wq;  // Variance effective selection mass

        // Adaptation 파라미터들
        const double c_sigma = (mu_eff + 2.0) / (N + mu_eff + 5.0);
        const double d_sigma =
            1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (N + 1.0)) - 1.0) + c_sigma;
        const double c_cov = (4.0 + mu_eff / N) / (N + 4.0 + 2.0 * mu_eff / N);
        const double c_1 = 2.0 / ((N + 1.3) * (N + 1.3) + mu_eff);
        const double c_mu = std::min(
            1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((N + 2.0) * (N + 2.0) + mu_eff));
        const double E_norm_N = std::sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N));

        alignas(64) std::array<double, N> mu = mu_init;
        alignas(64) std::array<double, N> p_sigma = {0.0};
        alignas(64) std::array<double, N> p_cov = {0.0};

        // 공분산 행렬 C 초기화 (단위 행렬)
        alignas(64) std::array<std::array<double, N>, N> C = {0.0};
        for (size_t i = 0; i < N; ++i) C[i][i] = 1.0;

        static thread_local std::mt19937 gen(seed);
        std::normal_distribution<double> std_norm(0.0, 1.0);
        static thread_local std::array<Sample<N>, M> pop;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 👑 CMA-ES Optimizer Started (Seed: " << seed << ")\n";
            std::cout << "========================================================\n";
        }

        size_t iter = 0;
        for (iter = 1; iter <= max_iter; ++iter) {
            alignas(64) std::array<std::array<double, N>, N> L = {0.0};

            // 공분산 행렬 C 분해 (Cholesky)
            if (!cholesky_decomposition<N>(C, L)) {
                if (verbose) std::cout << "  ↳ ⚠️ Matrix not positive definite. Stopping.\n";
                break;
            }

            // 1. M개의 샘플링 및 평가
            for (size_t k = 0; k < M; ++k) {
#pragma omp simd
                for (size_t i = 0; i < N; ++i) pop[k].z[i] = std_norm(gen);

                for (size_t i = 0; i < N; ++i) {
                    double Lz = 0.0;
#pragma omp simd
                    for (size_t j = 0; j <= i; ++j) Lz = std::fma(L[i][j], pop[k].z[j], Lz);
                    pop[k].x[i] = std::fma(sigma, Lz, mu[i]);
                }
                pop[k].y = f(pop[k].x);
            }

            // 2. 엘리트 선별 (Partial Sort)
            std::partial_sort(pop.begin(), pop.begin() + M_ELITE, pop.end());

            // [종료 조건]
            if (pop[0].y < tol || sigma < 1e-12) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. 평균 및 가중 평균 이동량(step_z) 계산
            alignas(64) std::array<double, N> step_z = {0.0};
            alignas(64) std::array<double, N> step_x = {0.0};

            for (size_t e = 0; e < M_ELITE; ++e) {
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    step_z[i] = std::fma(w[e], pop[e].z[i], step_z[i]);
                }
            }

            for (size_t i = 0; i < N; ++i) {
                double Lz = 0.0;
#pragma omp simd
                for (size_t j = 0; j <= i; ++j) Lz = std::fma(L[i][j], step_z[j], Lz);
                step_x[i] = Lz;
                mu[i] = std::fma(sigma, step_x[i], mu[i]);
            }

            // 4. 진화 경로(Evolution Paths) 업데이트
            double norm_p_sigma = 0.0;
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                p_sigma[i] = std::fma(1.0 - c_sigma, p_sigma[i],
                                      std::sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * step_z[i]);
                norm_p_sigma = std::fma(p_sigma[i], p_sigma[i], norm_p_sigma);
            }
            norm_p_sigma = std::sqrt(norm_p_sigma);

            double h_sigma = (norm_p_sigma / std::sqrt(1.0 - std::pow(1.0 - c_sigma, 2.0 * iter)) <
                              (1.4 + 2.0 / (N + 1.0)) * E_norm_N)
                                 ? 1.0
                                 : 0.0;

#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                p_cov[i] =
                    std::fma(1.0 - c_cov, p_cov[i],
                             h_sigma * std::sqrt(c_cov * (2.0 - c_cov) * mu_eff) * step_x[i]);
            }

            // 5. 공분산 행렬 C 업데이트
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i; j < N; ++j) {  // 대칭 행렬이므로 절반만 연산
                    double rank_mu_update = 0.0;
                    for (size_t e = 0; e < M_ELITE; ++e) {
                        double Lz_i = 0.0, Lz_j = 0.0;
                        for (size_t k = 0; k <= i; ++k) Lz_i += L[i][k] * pop[e].z[k];
                        for (size_t k = 0; k <= j; ++k) Lz_j += L[j][k] * pop[e].z[k];
                        rank_mu_update += w[e] * Lz_i * Lz_j;
                    }

                    double rank_1_update =
                        p_cov[i] * p_cov[j] + (1.0 - h_sigma) * c_cov * (2.0 - c_cov) * C[i][j];
                    C[i][j] =
                        (1.0 - c_1 - c_mu) * C[i][j] + c_1 * rank_1_update + c_mu * rank_mu_update;
                    C[j][i] = C[i][j];  // 대칭성 강제
                }
            }

            // 6. 보폭(Sigma) 업데이트
            sigma *= std::exp((c_sigma / d_sigma) * (norm_p_sigma / E_norm_N - 1.0));

            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(5) << pop[0].y << " | sigma: " << sigma << "\n";
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
        return {mu, pop[0].y, iter, duration.count()};
    }
};

}  // namespace Optimization
#endif  // OPTIMIZATION_CMA_ES_HPP_