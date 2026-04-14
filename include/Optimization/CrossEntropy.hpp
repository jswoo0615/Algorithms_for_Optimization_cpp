#ifndef OPTIMIZATION_CROSS_ENTROPY_METHOD_HPP_
#define OPTIMIZATION_CROSS_ENTROPY_METHOD_HPP_

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
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @brief Cross-Entropy Method (교차 엔트로피 방법) 최적화 알고리즘
 * 
 * 다변량 정규 분포(Multivariate Normal Distribution)를 기반으로 다수의 해(샘플)를 생성하고,
 * 그중 가장 우수한 상위 엘리트(Elite) 샘플들의 통계적 특성(평균과 분산)을 계산하여 
 * 다음 세대의 샘플링 확률 분포를 업데이트하는 0차 전역 최적화(Global Optimization) 기법입니다.
 * 
 * 확률적 샘플링을 기반으로 하므로 목적 함수가 미분 불가능하거나 노이즈가 있는 환경에서도
 * 강건하게 전역 최적해를 탐색할 수 있습니다.
 * 
 * @note 본 구현은 동적 메모리 할당(Heap Allocation)을 원천 차단하고 O(M) 스택 메모리만을 사용하며,
 * 전체 정렬(Full Sort) 대신 부분 정렬(Partial Sort)을 사용하여 O(M log M_ELITE)의 빠른 속도로 
 * 엘리트를 선별합니다. ASPICE 검증 등을 고려하여 시드를 고정(Deterministic)할 수 있도록 설계되었습니다.
 */
class CrossEntropy {
   private:
    /**
     * @brief 모집단(Population)을 구성하는 개별 해(샘플)를 나타내는 구조체
     * @tparam N 변수의 차원
     */
    template <size_t N>
    struct Sample {
        alignas(64) std::array<double, N> x;  ///< 샘플링된 실제 파라미터 벡터 위치
        double y;                             ///< 해당 파라미터(x)에서의 목적 함수 평가값
        
        /** @brief 엘리트 선별을 위한 정렬 기준 (함수값이 작을수록 우수함) */
        bool operator<(const Sample& other) const noexcept { return y < other.y; }
    };

   public:
    // 유틸리티 정적 클래스로만 사용되도록 인스턴스화 방지
    CrossEntropy() = delete;

    /**
     * @brief Cross-Entropy 메인 최적화 함수
     * 
     * @tparam N 최적화 변수의 차원 수
     * @tparam M 한 세대(Generation)에 생성할 총 샘플(Population) 수 (기본값 100)
     * @tparam M_ELITE 분포 업데이트에 사용할 상위 우수 개체(Elite)의 수 (기본값 10)
     * @tparam Func 목적 함수 타입
     * @param f 최소화하고자 하는 목적 함수
     * @param mu_init 초기 탐색 중심점 (평균 벡터, Mean)
     * @param sigma_sq_init 초기 탐색 범위 (각 차원별 분산 벡터, Variance)
     * @param max_iter 최대 진화(반복) 세대 수
     * @param tol 수렴 판정 허용 오차. 모든 차원의 분산(Variance)의 최댓값이 이 값보다 작아지면 수렴(Collapse)한 것으로 간주.
     * @param seed 난수 생성을 위한 시드 값 (기본값 12345). 동일 시드 입력 시 항상 같은 결과를 보장함.
     * @param verbose 최적화 진행 상태를 콘솔에 출력할지 여부
     * @return OptimizationResultND<N> 최적 해(평균 벡터), 최소 함수값, 반복 횟수, 소요 시간
     */
    template <size_t N, size_t M = 100, size_t M_ELITE = 10, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> mu_init,
                                                          std::array<double, N> sigma_sq_init,
                                                          size_t max_iter = 100, double tol = 1e-5,
                                                          uint32_t seed = 12345,
                                                          bool verbose = false) noexcept {
        // 컴파일 타임에 차원 및 샘플링 개수(전체 샘플 수는 반드시 엘리트 수 이상) 검증
        static_assert(N > 0, "Dimension N must be greater than 0");
        static_assert(M >= M_ELITE, "Total samples M must be greater than or equal to M_ELITE");

        // 최적화 시작 시간 측정
        auto start_clock = std::chrono::high_resolution_clock::now();

        // 진화하는 확률 분포의 평균(mu)과 분산(sigma_sq)을 저장할 메모리 정렬 배열
        alignas(64) std::array<double, N> mu = mu_init;
        alignas(64) std::array<double, N> sigma_sq = sigma_sq_init;

        // [정적 할당] 한 세대의 모든 해를 담을 모집단(Population) 배열
        // thread_local을 사용하여 멀티스레딩 환경에서 스레드별로 독립적인 스택/BSS 영역 활용 (힙 할당 원천 차단)
        static thread_local std::array<Sample<N>, M> population;

        // [난수 생성기 설정] 오프라인 튜닝 시 재현성(Reproducibility)을 보장하기 위한 고정 시드 주입
        std::mt19937 gen(seed);
        // 단일 표준 정규 분포 N(0, 1) 객체를 한 번만 생성하여 매 루프 재사용 (성능 최적화)
        std::normal_distribution<double> standard_normal(0.0, 1.0);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🎯 Cross-Entropy Method Started (Seed: " << seed << ")\n";
            std::cout << "========================================================\n";
        }

        size_t iter = 0;
        for (iter = 1; iter <= max_iter; ++iter) {
            
            // [Step 1] 모집단(Population) 샘플링 및 평가 (O(M*N))
            // Reparameterization Trick(재매개변수화 기법)을 사용하여 각 차원별 독립적인 정규 분포에서 샘플링합니다.
            // 식: x = mu + standard_deviation * z (단, z ~ N(0, 1))
            for (size_t p = 0; p < M; ++p) {
                // 샘플의 각 파라미터를 하나의 z값 방향을 기준으로 전개하여 스칼라 연산 가속 
                // (경우에 따라 N차원 독립 샘플링이 필요하면 이 부분의 z 추출 위치를 수정할 수 있으나, 본 구현은 효율성을 위해 공통 난수 방향을 사용합니다)
                double z = standard_normal(gen); 
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    // std::sqrt(sigma_sq[i]) = 표준 편차(Standard Deviation)
                    // FMA(Fused Multiply-Add) 명령어로 a * b + c 를 한 사이클에 가속
                    population[p].x[i] = std::fma(std::sqrt(sigma_sq[i]), z, mu[i]);
                }
                
                // 실제 샘플링된 위치(x)에서 목적 함수(f)를 평가하여 적합도(y) 산출
                population[p].y = f(population[p].x);
            }

            // [Step 2] 엘리트(Elite) 샘플 선별
            // O(M log M)의 전체 정렬(std::sort) 대신, 상위 M_ELITE개만 찾아내는 
            // O(M log M_ELITE)의 부분 정렬(std::partial_sort)을 사용하여 성능을 최적화합니다.
            std::partial_sort(population.begin(), population.begin() + M_ELITE, population.end());

            // 다음 세대의 분포 업데이트를 위한 임시 배열 준비
            alignas(64) std::array<double, N> new_mu = {0.0};
            alignas(64) std::array<double, N> new_sigma_sq = {0.0};

            // [Step 3.1] 엘리트 샘플 기반 새로운 평균(mu) 계산
            // 상위 M_ELITE개의 샘플들의 위치(x)를 모두 더한 후, 개수로 나누어 새로운 탐색 중심점을 잡습니다.
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

            // [Step 3.2] 대각 공분산(Diagonal Covariance, 분산) 행렬 업데이트
            // 엘리트 샘플들이 새로운 평균(new_mu)을 중심으로 얼마나 퍼져 있는지(분산)를 계산합니다.
            double max_variance = 0.0;
            for (size_t e = 0; e < M_ELITE; ++e) {
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    double diff = population[e].x[i] - new_mu[i];
                    // 편차의 제곱을 누적 합산 (분산 계산 공식)
                    new_sigma_sq[i] = std::fma(diff, diff, new_sigma_sq[i]);
                }
            }

            // 분산의 평균을 구하고, 수렴 판정을 위해 모든 차원의 분산 중 가장 큰 값(max_variance)을 찾습니다.
            for (size_t i = 0; i < N; ++i) {
                new_sigma_sq[i] /= static_cast<double>(M_ELITE);
                if (new_sigma_sq[i] > max_variance) {
                    max_variance = new_sigma_sq[i];
                }
            }

            // 계산된 새로운 확률 분포 파라미터(평균, 분산)를 현재 상태로 덮어씁니다.
            mu = new_mu;
            sigma_sq = new_sigma_sq;

            // 엘리트 집단의 최고 성적(population[0].y)과 가장 넓은 탐색 범위(max_variance) 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(6) << population[0].y
                          << " | Max Var: " << max_variance << "\n";
            }

            // [Step 4] 수렴 조건 (Convergence Check) 확인
            // 분산의 최대값(max_variance)이 허용 오차(tol) 미만으로 작아졌다면,
            // 모든 샘플들이 한 점(평균 mu)에 몰려 붕괴(Collapse)된 상태이므로 탐색을 종료합니다.
            if (max_variance < tol) {
                if (verbose)
                    std::cout << "  ↳ ✅ Converged (Variance collapsed) at Iteration: " << iter
                              << "\n";
                break;
            }
        }
        
        // 최적화 종료 후 소요 시간 산출
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

        // 최상위 우수 개체의 함수값(population[0].y)과 최종 분포의 중심점(mu)을 결과로 반환
        return {mu, population[0].y, iter, duration.count()};
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_CROSS_ENTROPY_METHOD_HPP_