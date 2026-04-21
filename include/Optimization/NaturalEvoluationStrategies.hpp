#ifndef OPTIMIZATION_NATURAL_EVOLUTION_STRATEGIES_HPP_
#define OPTIMIZATION_NATURAL_EVOLUTION_STRATEGIES_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

// 결과 반환 구조체 일관성 유지 (중복 정의 방지)
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 저장하는 구조체
 * @tparam N 최적화 대상 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt; ///< 최적해 (Optimal solution)
    double f_opt;                ///< 최적 함수값 (Optimal function value)
    size_t iterations;           ///< 총 반복 횟수 (Total iterations)
    long long elapsed_ns;        ///< 소요 시간 (나노초 단위)
};
}  // namespace Optimization
#endif

namespace Optimization {

/**
 * @class NaturalEvolutionStrategies
 * @brief 자연 진화 전략 (Natural Evolution Strategies, NES) 알고리즘 정적 클래스
 * 
 * @details NES는 확률 분포(주로 정규 분포)의 파라미터(평균 mu, 분산 sigma^2)를 업데이트하면서
 * 목적 함수의 기댓값을 최적화하는 미분 불필요(Derivative-Free) 최적화 기법입니다.
 * 일반적인 경사 하강법과 달리, 파라미터 공간의 기하학적 구조를 반영하는 피셔 정보 행렬(Fisher Information Matrix)의
 * 역행렬을 곱한 '자연 기울기(Natural Gradient)'를 사용하여 업데이트 방향을 결정합니다.
 * 
 * @note 본 구현체는 다음과 같은 특징을 가집니다:
 * 1. **Natural Gradient**: 피셔 정보 행렬 역행렬을 적용하여 바닐라 기울기가 가지는 스케일 및 나눗셈 폭탄(Division by zero) 문제를 해소하고 안전한 업데이트를 보장합니다.
 * 2. **Fitness Normalization (Z-Score)**: 목적 함수의 절대적인 스케일 차이에 의해 기울기가 폭주(Overshooting)하는 현상을 막기 위해 평가된 함수값들을 정규화합니다.
 */
class NaturalEvolutionStrategies {
   public:
    NaturalEvolutionStrategies() = delete; // 정적 메서드만 제공하므로 인스턴스화 방지

    /**
     * @brief NES 알고리즘을 수행하여 최적해를 찾는 함수
     * 
     * @tparam N 최적화하려는 변수의 차원 수
     * @tparam M 매 반복마다 샘플링할 개체(Population) 수 (기본값: 100)
     * @tparam Func 목적 함수 타입
     * @param f 최적화할 목적 함수 (미분 가능할 필요 없음)
     * @param mu_init 탐색 분포의 초기 평균 벡터 (시작 위치)
     * @param sigma_sq_init 탐색 분포의 초기 분산 벡터 (탐색 반경의 제곱)
     * @param alpha 학습률 (보폭, Learning Rate). 자연 기울기 업데이트의 크기를 조절 (기본값: 0.05)
     * @param max_iter 최대 반복(세대) 횟수
     * @param seed 난수 생성기 시드값 (기본값: 12345, 재현성 보장용)
     * @param verbose 진행 과정 콘솔 출력 여부
     * @return 최적해(mu), 최적값(best_y_global), 반복 횟수, 연산 시간을 포함하는 구조체
     */
    template <size_t N, size_t M = 100, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> mu_init,
                                                          std::array<double, N> sigma_sq_init,
                                                          double alpha = 0.05,
                                                          size_t max_iter = 100,
                                                          uint32_t seed = 12345,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be > 0"); // 컴파일 타임 차원 검사
        auto start_clock = std::chrono::high_resolution_clock::now(); // 소요 시간 측정 시작

        // 메모리 접근 속도 향상 및 SIMD 연산 최적화를 위한 64바이트 캐시 라인 정렬
        alignas(64) std::array<double, N> mu = mu_init;
        alignas(64) std::array<double, N> sigma_sq = sigma_sq_init;

        // thread_local 난수 생성기를 사용하여 병렬 환경에서도 안전하고 생성 오버헤드를 줄임
        static thread_local std::mt19937 gen(seed);
        std::normal_distribution<double> standard_normal(0.0, 1.0); // 표준 정규 분포 N(0, 1)

        // 매 세대(Generation)마다 생성할 M개의 샘플 데이터를 저장할 캐시 메모리 (O(M) 공간 복잡도)
        // thread_local 정적 할당으로 런타임 동적 할당(new/malloc) 비용 제거
        static thread_local std::array<std::array<double, N>, M> z_cache; // 정규 분포에서 뽑은 노이즈 z 저장
        static thread_local std::array<double, M> y_cache; // 각 샘플 위치에서의 함수 평가값 f(x) 저장

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🧬 True Natural Evolution Strategies Started\n";
            std::cout << "========================================================\n";
        }

        double best_y_global = 1e99; // 전체 세대를 통틀어 가장 좋은(작은) 함수값 기록
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            double sum_y = 0.0;
            double sum_y_sq = 0.0;
            double best_y_iter = 1e99; // 현재 세대에서의 최고 함수값

            // -------------------------------------------------------------
            // 1. 샘플 생성 및 목적 함수 평가 (Pass 1)
            // 현재의 확률 분포 N(mu, sigma^2)를 바탕으로 M개의 자식(개체)들을 생성합니다.
            // -------------------------------------------------------------
            for (size_t p = 0; p < M; ++p) {
                alignas(64) std::array<double, N> x = {0.0};
                
// OpenMP SIMD 지시어를 통해 벡터 연산을 가속화
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    // 표준 정규 분포에서 노이즈 z 추출
                    z_cache[p][i] = standard_normal(gen);
                    // 실제 위치 x = mu + z * sigma
                    // fma(Fused Multiply-Add)를 사용하여 정밀도를 유지하며 빠르게 연산
                    x[i] = std::fma(std::sqrt(sigma_sq[i]), z_cache[p][i], mu[i]);
                }

                y_cache[p] = f(x); // 목적 함수 평가
                
                // 정규화(Normalization)를 위한 기초 통계량(합, 제곱합) 누적
                sum_y += y_cache[p];
                sum_y_sq += y_cache[p] * y_cache[p];
                
                // 현재 세대의 최솟값 갱신
                if (y_cache[p] < best_y_iter) best_y_iter = y_cache[p];
            }

            // 전체(Global) 최솟값 갱신
            if (best_y_iter < best_y_global) best_y_global = best_y_iter;

            // -------------------------------------------------------------
            // [핵심 1] Fitness Normalization (Z-Score 정규화)
            // 함수값의 스케일이 매우 크거나 작더라도 일정한 크기의 업데이트를 보장하기 위해
            // 현재 세대 함수값들의 평균과 표준편차를 구하여 Z-Score로 변환합니다.
            // 이를 통해 극단적인 목적 함수 값으로 인한 파라미터 폭주(Overshooting)를 방지합니다.
            // -------------------------------------------------------------
            double mean_y = sum_y / static_cast<double>(M);
            double var_y = (sum_y_sq / static_cast<double>(M)) - (mean_y * mean_y);
            // 분산이 0이 되어 0으로 나누는 오류를 방지하기 위해 최소값(1e-12) 보장
            double std_y = std::sqrt(std::max(var_y, 1e-12));

            // 자연 기울기 누적을 위한 배열
            alignas(64) std::array<double, N> grad_mu = {0.0};
            alignas(64) std::array<double, N> grad_sigma_sq = {0.0};

            // -------------------------------------------------------------
            // 2. Natural Gradient 계산 및 누적 (Pass 2)
            // 피셔 정보 행렬(F)의 역행렬이 이미 곱해진 형태의 기울기를 도출합니다.
            // -------------------------------------------------------------
            for (size_t p = 0; p < M; ++p) {
                // Z-Score로 정규화된 유틸리티(Utility) 값
                // 주의: 최솟값을 찾는 문제이므로, 함수값이 평균(mean_y)보다 작을수록 
                // normalized_y는 음수가 되며, 업데이트 방향을 유리한 쪽으로 이끕니다.
                double normalized_y = (y_cache[p] - mean_y) / std_y;

#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    double var = sigma_sq[i];
                    double std_dev = std::sqrt(var);

                    // [핵심 2] Natural Gradient 도출
                    // 일반적인 Log-Likelihood의 기울기(Vanilla Gradient)에 Fisher Information Matrix의 역행렬(F^{-1})을 곱한 결과입니다.
                    // 놀랍게도 복잡한 나눗셈 항들이 상쇄되어 매우 우아하고 단순한 곱셈 형태가 됩니다.
                    // - 평균에 대한 자연 기울기: z * sigma
                    // - 분산에 대한 자연 기울기: (z^2 - 1) * sigma^2
                    double nat_grad_mu = z_cache[p][i] * std_dev;
                    double nat_grad_sigma = (z_cache[p][i] * z_cache[p][i] - 1.0) * var;

                    // 정규화된 함수값(Utility)을 가중치로 삼아 기울기를 누적합니다.
                    // fma 연산 : grad = normalized_y * nat_grad + grad
                    grad_mu[i] = std::fma(normalized_y, nat_grad_mu, grad_mu[i]);
                    grad_sigma_sq[i] = std::fma(normalized_y, nat_grad_sigma, grad_sigma_sq[i]);
                }
            }

            // -------------------------------------------------------------
            // 3. 파라미터 업데이트 (경사 하강)
            // 누적된 기울기의 평균을 구하고 학습률(alpha)을 적용해 mu와 sigma_sq를 업데이트합니다.
            // 최솟값을 찾는 문제이므로 기울기의 반대 방향(-alpha)으로 이동합니다.
            // -------------------------------------------------------------
            double inv_M = 1.0 / static_cast<double>(M); // 나눗셈 대신 곱셈을 위해 역수 사전 계산
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                // mu_new = mu_old - alpha * (grad_mu / M)
                mu[i] = std::fma(-alpha, grad_mu[i] * inv_M, mu[i]);
                // sigma_sq_new = sigma_sq_old - alpha * (grad_sigma_sq / M)
                sigma_sq[i] = std::fma(-alpha, grad_sigma_sq[i] * inv_M, sigma_sq[i]);

                // 분산이 0 또는 음수가 되면 더 이상 탐색이 불가능하므로 (분산 붕괴 방지)
                // 하드 리미트(1e-4)를 설정하여 최소한의 탐색 반경을 유지합니다.
                if (sigma_sq[i] < 1e-4) sigma_sq[i] = 1e-4;
            }

            // 진행 상태 출력: 첫 세대이거나 매 10세대마다 한 번씩 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(5) << best_y_global << " | mu: [" << mu[0] << ", "
                          << mu[1] << "]\n";
            }
        }

        // 최적화 종료 및 소요 시간 계산
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
        
        // 탐색이 완료된 확률 분포의 평균 위치(mu)를 최적해로 반환합니다.
        return {mu, best_y_global, iter, duration.count()};
    }
};

}  // namespace Optimization
#endif  // OPTIMIZATION_NATURAL_EVOLUTION_STRATEGIES_HPP_