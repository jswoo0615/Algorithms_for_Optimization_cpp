#ifndef OPTIMIZATION_GENERALIZED_PATTERN_SEARCH_HPP_
#define OPTIMIZATION_GENERALIZED_PATTERN_SEARCH_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과 구조체
 * @details 최적화 알고리즘의 수행 결과를 저장하고 반환하는 데 사용됩니다.
 * 
 * @tparam N 변수의 차원 (크기)
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt; ///< 탐색이 완료된 최적의 매개변수 (위치) 배열
    double f_opt;                ///< 최적의 위치에서 평가된 목적 함수의 값
    size_t iterations;           ///< 최적화 완료까지 수행된 반복 횟수
    long long elapsed_ns;        ///< 알고리즘 수행에 걸린 시간 (나노초 단위)
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @brief 일반화된 패턴 탐색 (Generalized Pattern Search, GPS) 알고리즘 클래스
 * @details 
 * 기울기(Gradient) 정보를 사용하지 않는(Derivative-free) 직접 탐색(Direct Search) 방법 중 하나입니다.
 * 주어진 위치 주변을 특정 방향 집합(Positive Spanning Set)을 따라 탐색하며, 더 나은 지점(목적 함수값이 작은 지점)을
 * 찾으면 즉시 해당 지점으로 이동(기회주의적 업데이트)합니다.
 * 만약 모든 방향을 다 뒤져도 더 나은 지점을 찾지 못하면 보폭(Step size, alpha)을 줄여 더 세밀하게 탐색합니다.
 * 
 * @note 본 구현은 두 가지 최적화 기법을 도입하여 실시간 제어 환경에서의 속도와 강건성을 확보했습니다.
 * 1. 기회주의적 탐색 (Opportunistic Search): 
 *    모든 M개의 방향을 다 계산한 뒤 제일 좋은 곳으로 가는 것이 아니라, 현재보다 조금이라도 좋아지는 
 *    방향을 발견하면 계산을 즉시 중단하고 위치를 업데이트합니다.
 * 2. 동적 정렬 (Dynamic Ordering):
 *    성공적으로 업데이트를 이끌어낸 방향 벡터를 배열의 맨 앞(index 0)으로 끌어옵니다(Shift).
 *    일반적으로 볼록한 함수에서 한 번 내려가기 시작한 방향은 다음 스텝에서도 또 내려갈 확률이 높기 때문입니다.
 *    이로 인해 평균적인 목적 함수 평가 횟수를 크게 줄일 수 있습니다.
 * 3. MISRA C++ 준수 및 SIMD/FMA 가속: 
 *    정적 배열(std::array)을 사용해 동적 메모리 할당(힙)을 원천 차단하고 메모리 단편화를 방지합니다.
 */
class GeneralizedPatternSearch {
   public:
    GeneralizedPatternSearch() = delete; // 유틸리티 클래스이므로 인스턴스화 방지

    /**
     * @brief GPS 알고리즘을 실행하여 최적해를 탐색합니다.
     * 
     * @tparam N 상태 공간(변수)의 차원 수
     * @tparam M 탐색 방향 집합(Positive Spanning Set)의 방향 개수 (항상 N + 1 이상이어야 함)
     * @tparam Func 목적 함수 타입
     * @param f 최소화하려는 목적 함수
     * @param x_init 탐색을 시작할 초기 위치 벡터
     * @param D 탐색 방향들을 담은 2차원 정적 배열 (Positive Spanning Set 행렬).
     *          이 벡터들은 공간의 모든 방향을 양의 계수로 조합해낼 수 있어야 합니다. (예: 축 방향 및 그 반대 방향)
     * @param alpha 초기 탐색 보폭 (Step size). (기본값: 1.0)
     * @param epsilon 수렴 판정 기준치. 보폭(alpha)이 이 값보다 작아지면 최적점에 도달했다고 판단하고 종료합니다. (기본값: 1e-5)
     * @param gamma 보폭 축소 비율. 주변 탐색에 실패했을 때 보폭을 얼마나 줄일 것인지 결정합니다. (기본값: 0.5)
     * @param max_iter 최대 반복 횟수 (기본값: 10000)
     * @param verbose 실행 과정 및 상태 출력 여부 (기본값: false)
     * @return 최적화 결과가 담긴 OptimizationResultND 구조체
     */
    template <size_t N, size_t M, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          std::array<std::array<double, N>, M> D,
                                                          double alpha = 1.0, double epsilon = 1e-5,
                                                          double gamma = 0.5,
                                                          size_t max_iter = 10000,
                                                          bool verbose = false) noexcept {
        // 컴파일 타임 검증: 차원은 0보다 커야 하며,
        // 공간 전체를 양수로 스패닝(Positive Spanning)하려면 최소 N+1개의 방향 벡터가 필수입니다.
        static_assert(N > 0, "Dimension N must be greater than 0");
        static_assert(M >= N + 1, "A positive spanning set must have at least N+1 directions");

        // 실행 시간 측정을 위한 타이머 시작
        auto start_clock = std::chrono::high_resolution_clock::now();

        // 64바이트 캐시 라인에 맞게 정렬하여 메모리 접근 속도 및 SIMD 벡터화 효율을 극대화
        alignas(64) std::array<double, N> x = x_init;
        double y = f(x); // 초기 위치에서의 목적 함수 값
        size_t iter = 0;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🕸️ Generalized Pattern Search Started (alpha0=" << alpha << ")\n";
            std::cout << "========================================================\n";
        }

        // 메인 최적화 루프
        // 보폭(alpha)이 허용 오차(epsilon)보다 크고, 최대 반복 횟수에 도달하지 않을 때까지 반복
        while (alpha > epsilon && iter < max_iter) {
            iter++;
            bool improved = false; // 이번 iteration에서 업데이트(성공)가 일어났는지 추적

            // 주어진 방향 집합(M개)을 순서대로 탐색
            for (size_t i = 0; i < M; ++i) {
                alignas(64) std::array<double, N> x_prime = {0.0};

                // 탐색 위치 계산: x_prime = x + alpha * D[i]
                // SIMD 지시어와 FMA(Fused Multiply-Add) 연산을 사용하여 부동소수점 오차를 줄이고 병렬로 빠르게 연산
#pragma omp simd
                for (size_t j = 0; j < N; ++j) {
                    x_prime[j] = std::fma(alpha, D[i][j], x[j]);
                }

                // 새로운 위치에서의 목적 함수값 평가
                double y_prime = f(x_prime);

                // 1. 기회주의적 업데이트 (Opportunistic Update)
                // 만약 단 하나라도 현재(y)보다 더 나은(목적 함수값이 작은) 위치를 발견하면,
                // 나머지 방향은 더 이상 평가하지 않고 즉시 위치를 이동합니다.
                if (y_prime < y) {
                    x = x_prime;
                    y = y_prime;
                    improved = true;

                    // 2. 동적 정렬 (Dynamic Ordering)
                    // 방금 성공한 방향(D[i])이 다음번 탐색에서도 또 성공할 확률이 높다는 가정 하에
                    // 이 방향 벡터를 탐색 우선순위 1위(배열의 맨 앞, index 0)로 재배치(Shift)합니다.
                    // 이는 불필요한 함수 평가를 획기적으로 줄여주는 휴리스틱 기법입니다.
                    if (i > 0) {
                        alignas(64) std::array<double, N> best_d = D[i];
                        // 성공한 방향을 빼내고, 그 앞의 요소들을 한 칸씩 뒤로 밉니다.
                        for (size_t k = i; k > 0; --k) {
                            D[k] = D[k - 1];
                        }
                        D[0] = best_d; // 성공한 방향을 배열 맨 앞에 삽입
                    }
                    
                    // 기회주의적 탐색이므로 한 번 성공하면 현재 루프를 빠져나가고 다음 step으로 넘어감
                    break;
                }
            }

            // 3. 패턴 탐색 실패 처리 (Step size 축소)
            // M개의 방향을 모두 탐색했지만 어느 곳으로 가도 목적 함수값이 줄어들지 않은 경우,
            // 현재 지점이 골짜기의 바닥 근처이거나 국소 최적점(Local Minimum)에 가까워졌다는 뜻입니다.
            // 이때는 보폭(alpha)에 축소 비율(gamma, 주로 0.5)을 곱해 탐색 반경을 세밀하게 좁힙니다.
            if (!improved) {
                alpha *= gamma;
            }
            
            // 상태 출력
            // 실패해서 보폭이 줄어들었을 때 또는 특정 주기(100번)마다 진행 상황을 알립니다.
            if (verbose && (!improved || iter % 100 == 0)) {
                std::cout << "[Iter " << std::setw(4) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << y << " | alpha: " << alpha << "\n";
            }
        }

        // 최적화 루프 종료 및 소요 시간 계산
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

#endif  // OPTIMIZATION_GENERALIZED_PATTERN_SEARCH_HPP_