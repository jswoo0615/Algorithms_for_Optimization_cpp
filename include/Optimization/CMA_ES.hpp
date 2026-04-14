#ifndef OPTIMIZATION_CMA_ES_HPP_
#define OPTIMIZATION_CMA_ES_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// 결과 반환용 구조체 중복 정의 방지
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 담는 구조체
 * @tparam N 최적화 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt;  ///< 최적해 (Optimal Point)
    double f_opt;                 ///< 최적해에서의 목적 함수 값
    size_t iterations;            ///< 최적해를 찾는 데 소요된 반복 횟수
    long long elapsed_ns;         ///< 최적화에 소요된 시간 (나노초 단위)
};
}  // namespace Optimization
#endif

namespace Optimization {

/**
 * @brief CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 최적화 알고리즘
 * 
 * 진화 연산(Evolutionary Algorithm) 중 하나로, 확률 분포(다변량 정규 분포)를 사용하여 
 * 해 집단(Population)을 샘플링하고 가장 우수한 해(Elite)들을 바탕으로 분포를 업데이트하는 
 * 0차(Zero-Order) 전역 최적화(Global Optimization) 기법입니다.
 * 
 * 특히, 공분산 행렬(Covariance Matrix)을 지속적으로 적응(Adaptation)시켜 나가는 특징이 있으며, 
 * 이는 목적 함수의 역-헤시안(Inverse Hessian) 공간을 근사하는 것과 유사한 효과를 내어 
 * 일리컨디셔닝(Ill-conditioned)된 문제에서도 매우 강력한 성능을 발휘합니다.
 * 
 * @note 동적 할당(Heap Allocation)을 철저히 배제하고 정적 배열(`std::array`)만을 사용하여 구현되었으며, 
 * 내부적으로 O(1) 메모리를 사용하는 촐레스키 분해(Cholesky Decomposition) 엔진을 내장하고 있습니다.
 */
class CMA_ES {
   private:
    /**
     * @brief 각 세대(Generation)에서 샘플링된 하나의 해(개체)를 나타내는 구조체
     * @tparam N 변수의 차원
     */
    template <size_t N>
    struct Sample {
        alignas(64) std::array<double, N> x;  ///< 실제 샘플링된 파라미터 벡터
        alignas(64) std::array<double, N> z;  ///< N(0, I)에서 추출된 표준 정규 난수 벡터
        double y;                             ///< 해당 파라미터(x)에서의 목적 함수 평가값
        
        /** @brief 엘리트 선별을 위한 정렬 기준 (함수값이 작을수록 우수함) */
        bool operator<(const Sample& other) const noexcept { return y < other.y; }
    };

    /**
     * @brief 촐레스키 분해 (Cholesky Decomposition) 내장 엔진
     * 
     * 대칭 양의 정부호 행렬(Symmetric Positive Definite Matrix) A를 
     * 하삼각 행렬(Lower Triangular Matrix) L과 그 전치 행렬의 곱 (A = L * L^T)으로 분해합니다.
     * CMA-ES에서는 공분산 행렬 C로부터 다변량 정규 분포 샘플을 생성할 때 사용됩니다.
     * 
     * @param A 분해할 대상 행렬 (공분산 행렬)
     * @param L 분해 결과로 얻어지는 하삼각 행렬
     * @return bool 행렬 A가 양의 정부호가 아니어 분해에 실패하면 false 반환
     */
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
                    if (val <= 0.0) return false;  // 양의 정부호(Positive Definite) 조건 위배
                    L[i][j] = std::sqrt(val);
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        return true;
    }

   public:
    // 유틸리티 정적 클래스로만 사용되도록 인스턴스화 방지
    CMA_ES() = delete;

    /**
     * @brief CMA-ES 메인 최적화 함수
     * 
     * @tparam N 최적화 변수의 차원 수
     * @tparam M 한 세대(Generation)에서 샘플링할 개체(Population)의 수 (Lambda, 기본값 10)
     * @tparam M_ELITE 분포 업데이트에 반영할 상위 우수 개체의 수 (Mu, 기본값 5)
     * @tparam Func 목적 함수 타입
     * @param f 최소화하고자 하는 목적 함수
     * @param mu_init 탐색을 시작할 초기 평균 위치 벡터
     * @param sigma 초기 탐색 보폭(Step Size / Standard Deviation). 문제의 스케일에 맞게 설정해야 함.
     * @param max_iter 최대 진화(반복) 세대 수
     * @param tol 수렴 판정 허용 오차. 목적 함수값이 이 값보다 작아지면 최적화 종료.
     * @param seed 난수 생성을 위한 시드 값
     * @param verbose 콘솔에 최적화 진행 상태 출력 여부
     * @return OptimizationResultND<N> 최적 해, 최소 함수값, 소요 반복 횟수 및 시간
     */
    template <size_t N, size_t M = 10, size_t M_ELITE = 5, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> mu_init,
                                                          double sigma = 1.0,
                                                          size_t max_iter = 1000, double tol = 1e-6,
                                                          uint32_t seed = 12345,
                                                          bool verbose = false) noexcept {
        // 컴파일 타임 검증: 차원 수와 샘플 개수 등
        static_assert(N > 0, "Dimension must be > 0");
        static_assert(M >= M_ELITE && M_ELITE > 0, "Invalid sample size: M must be >= M_ELITE > 0");

        auto start_clock = std::chrono::high_resolution_clock::now();

        // =====================================================================
        // [초기화 1] 가중치 및 유효 선택 질량(Variance effective selection mass) 연산
        // 엘리트 해(상위 M_ELITE개)들이 평균 및 공분산 업데이트에 기여하는 가중치(w)를 계산합니다.
        // 루프 밖에서 1회만 계산(Zero-overhead in loop)합니다.
        // =====================================================================
        std::array<double, M_ELITE> w = {0.0};
        double sum_w = 0.0, sum_wq = 0.0;
        for (size_t i = 0; i < M_ELITE; ++i) {
            // 로그 가중치 (성적이 좋은 순서대로 큰 가중치를 부여)
            w[i] = std::log((M + 1.0) / 2.0) - std::log(i + 1.0);
            sum_w += w[i];
        }
        for (size_t i = 0; i < M_ELITE; ++i) {
            w[i] /= sum_w;          // 가중치 합이 1이 되도록 정규화
            sum_wq += w[i] * w[i];  // 유효 질량 계산을 위한 가중치 제곱합
        }
        // mu_eff: 실제 선택에 기여하는 유효한 엘리트 샘플 수 (보통 1 <= mu_eff <= M_ELITE)
        const double mu_eff = 1.0 / sum_wq;  

        // =====================================================================
        // [초기화 2] Adaptation(적응)에 필요한 하이퍼파라미터 상수 연산
        // 표준 CMA-ES 휴리스틱 수식을 따릅니다.
        // =====================================================================
        // 보폭(Sigma) 진화 경로(Evolution Path) 업데이트를 위한 학습률(Decay factor)
        const double c_sigma = (mu_eff + 2.0) / (N + mu_eff + 5.0);
        
        // 보폭(Sigma) 갱신 시 적용되는 댐핑(Damping) 파라미터
        const double d_sigma =
            1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (N + 1.0)) - 1.0) + c_sigma;
            
        // 공분산(Covariance) 진화 경로 업데이트를 위한 학습률
        const double c_cov = (4.0 + mu_eff / N) / (N + 4.0 + 2.0 * mu_eff / N);
        
        // Rank-1 업데이트 (단일 우수 탐색 방향 반영) 공분산 학습률
        const double c_1 = 2.0 / ((N + 1.3) * (N + 1.3) + mu_eff);
        
        // Rank-mu 업데이트 (다수 우수 탐색 방향의 분산 반영) 공분산 학습률
        const double c_mu = std::min(
            1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((N + 2.0) * (N + 2.0) + mu_eff));
            
        // 표준 정규 분포 벡터 크기의 기댓값 (보폭 조절 수렴 판단의 기준치)
        const double E_norm_N = std::sqrt(N) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N));


        // =====================================================================
        // [초기화 3] 진화 상태 변수 메모리 할당 (SIMD 성능을 위해 정렬됨)
        // =====================================================================
        alignas(64) std::array<double, N> mu = mu_init;       // 현재 분포의 평균 (탐색의 중심점)
        alignas(64) std::array<double, N> p_sigma = {0.0};    // 보폭 조절을 위한 진화 경로 (켤레 방향 누적)
        alignas(64) std::array<double, N> p_cov = {0.0};      // 공분산 행렬 업데이트를 위한 진화 경로

        // 공분산 행렬 C 초기화: 탐색 초기에는 각 차원이 독립적인 단위 행렬(Identity Matrix)로 시작
        alignas(64) std::array<std::array<double, N>, N> C = {0.0};
        for (size_t i = 0; i < N; ++i) C[i][i] = 1.0;

        // 난수 생성기 초기화 (스레드 로컬로 설정하여 다중 스레드 환경에서도 안전하도록 구성 가능)
        static thread_local std::mt19937 gen(seed);
        std::normal_distribution<double> std_norm(0.0, 1.0);
        
        // 해 집단(Population) 배열
        static thread_local std::array<Sample<N>, M> pop;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 👑 CMA-ES Optimizer Started (Seed: " << seed << ")\n";
            std::cout << "========================================================\n";
        }

        // =====================================================================
        // [메인 최적화 루프] 진화 세대 반복
        // =====================================================================
        size_t iter = 0;
        for (iter = 1; iter <= max_iter; ++iter) {
            // 하삼각 행렬 L 선언
            alignas(64) std::array<std::array<double, N>, N> L = {0.0};

            // 0. 공분산 행렬 C를 촐레스키 분해 (C = L * L^T)
            // L 행렬은 다변량 정규 분포에서 방향성(상관관계)을 부여하는 필터 역할을 합니다.
            if (!cholesky_decomposition<N>(C, L)) {
                if (verbose) std::cout << "  ↳ ⚠️ Matrix not positive definite. Stopping.\n";
                break;
            }

            // 1. M개의 개체 샘플링 및 목적 함수 평가
            for (size_t k = 0; k < M; ++k) {
                // (1) 표준 정규 난수 벡터 z 추출: z ~ N(0, I)
#pragma omp simd
                for (size_t i = 0; i < N; ++i) pop[k].z[i] = std_norm(gen);

                // (2) 실제 탐색 위치 x 산출: x = mu + sigma * (L * z)
                for (size_t i = 0; i < N; ++i) {
                    double Lz = 0.0;
#pragma omp simd
                    for (size_t j = 0; j <= i; ++j) Lz = std::fma(L[i][j], pop[k].z[j], Lz);
                    pop[k].x[i] = std::fma(sigma, Lz, mu[i]);
                }
                
                // (3) 목적 함수 평가
                pop[k].y = f(pop[k].x);
            }

            // 2. 엘리트 선별 (Partial Sort)
            // 목적 함수값이 작은 순서대로 상위 M_ELITE 개체까지만 정렬합니다. (나머지는 버림)
            std::partial_sort(pop.begin(), pop.begin() + M_ELITE, pop.end());

            // [종료 조건] 최고 엘리트 해의 목적 함수값이 허용 오차(tol) 이하이거나, 
            // 탐색 보폭(sigma)이 너무 작아져 더 이상 유의미한 탐색이 불가능한 경우 종료
            if (pop[0].y < tol || sigma < 1e-12) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. 새로운 평균(mu) 계산 및 평균의 이동량 산출
            alignas(64) std::array<double, N> step_z = {0.0}; // 정규화된 샘플 공간에서의 이동량
            alignas(64) std::array<double, N> step_x = {0.0}; // 실제 스케일 공간에서의 이동량 (sigma 제외)

            // 엘리트 샘플(z)들의 가중 평균 산출
            for (size_t e = 0; e < M_ELITE; ++e) {
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    step_z[i] = std::fma(w[e], pop[e].z[i], step_z[i]);
                }
            }

            // L 행렬을 곱하여 실제 공간의 이동량(step_x) 산출 및 새로운 탐색 중심점(mu) 업데이트
            // mu_new = mu_old + sigma * (L * step_z)
            for (size_t i = 0; i < N; ++i) {
                double Lz = 0.0;
#pragma omp simd
                for (size_t j = 0; j <= i; ++j) Lz = std::fma(L[i][j], step_z[j], Lz);
                step_x[i] = Lz;
                mu[i] = std::fma(sigma, step_x[i], mu[i]); // mu 업데이트
            }

            // 4. 진화 경로(Evolution Paths) 업데이트
            // 매 세대의 탐색 방향(Step)들이 지수 이동 평균(EMA)으로 누적되어, 
            // 연속적인 탐색 방향을 갖는지(경로가 길어지는지) 혹은 방향이 번갈아 바뀌는지(경로가 상쇄되는지) 기억합니다.

            double norm_p_sigma = 0.0;
            // (1) 보폭(sigma) 조절을 위한 진화 경로 (p_sigma)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                p_sigma[i] = std::fma(1.0 - c_sigma, p_sigma[i],
                                      std::sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * step_z[i]);
                norm_p_sigma = std::fma(p_sigma[i], p_sigma[i], norm_p_sigma);
            }
            norm_p_sigma = std::sqrt(norm_p_sigma); // ||p_sigma||

            // Heaviside step function: 진화 경로가 너무 길어질 때 공분산 행렬 업데이트가 발산하는 것을 막는 스위치
            double h_sigma = (norm_p_sigma / std::sqrt(1.0 - std::pow(1.0 - c_sigma, 2.0 * iter)) <
                              (1.4 + 2.0 / (N + 1.0)) * E_norm_N)
                                 ? 1.0
                                 : 0.0;

            // (2) 공분산 행렬(Covariance) 조절을 위한 진화 경로 (p_cov)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                p_cov[i] =
                    std::fma(1.0 - c_cov, p_cov[i],
                             h_sigma * std::sqrt(c_cov * (2.0 - c_cov) * mu_eff) * step_x[i]);
            }

            // 5. 공분산 행렬(C) 업데이트
            // 과거의 공분산 행렬에 Rank-1 업데이트(하나의 우세한 진화 경로)와 
            // Rank-mu 업데이트(현재 엘리트 샘플들의 분포 분산)를 가중합하여 갱신합니다.
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i; j < N; ++j) {  // 대칭 행렬(Symmetric)이므로 위쪽 삼각 부분(i <= j)만 계산
                    // [Rank-mu 업데이트 항 계산]
                    double rank_mu_update = 0.0;
                    for (size_t e = 0; e < M_ELITE; ++e) {
                        double Lz_i = 0.0, Lz_j = 0.0;
                        for (size_t k = 0; k <= i; ++k) Lz_i += L[i][k] * pop[e].z[k];
                        for (size_t k = 0; k <= j; ++k) Lz_j += L[j][k] * pop[e].z[k];
                        rank_mu_update += w[e] * Lz_i * Lz_j;
                    }

                    // [Rank-1 업데이트 항 계산]
                    double rank_1_update =
                        p_cov[i] * p_cov[j] + (1.0 - h_sigma) * c_cov * (2.0 - c_cov) * C[i][j];
                    
                    // 기존 C에 감쇠율을 곱하고 두 업데이트 항을 더하여 최종 C를 갱신
                    C[i][j] =
                        (1.0 - c_1 - c_mu) * C[i][j] + c_1 * rank_1_update + c_mu * rank_mu_update;
                    
                    // 대칭성(Symmetry) 유지 보장
                    C[j][i] = C[i][j];  
                }
            }

            // 6. 보폭(Sigma) 글로벌 스케일 업데이트
            // 진화 경로 p_sigma의 길이가 기댓값(E_norm_N)보다 길면 연속적인 하강 방향이라는 뜻이므로 보폭을 늘리고(sigma 증가),
            // 기댓값보다 짧으면 진동하고 있다는 뜻이므로 보폭을 줄입니다(sigma 감소).
            sigma *= std::exp((c_sigma / d_sigma) * (norm_p_sigma / E_norm_N - 1.0));

            // 진행 상태 주기적 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(5) << pop[0].y << " | sigma: " << sigma << "\n";
            }
        }

        // 최적화 종료 및 소요 시간 계산
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        // 최종 결과 반환 출력
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << mu[0];
            if constexpr (N > 1) std::cout << ", " << mu[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        
        // 구조체로 결과 반환 (탐색 중심점 mu를 최종 해로 간주)
        return {mu, pop[0].y, iter, duration.count()};
    }
};

}  // namespace Optimization
#endif  // OPTIMIZATION_CMA_ES_HPP_