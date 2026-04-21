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

// 최적화 결과를 담는 구조체 중복 정의 방지용 매크로
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 저장하는 구조체
 * @tparam N 최적화 대상 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt;  ///< 최적해 (Optimal solution)
    double f_opt;                 ///< 최적 함수값 (Optimal function value)
    size_t iterations;            ///< 총 반복 횟수 (Total iterations)
    long long elapsed_ns;         ///< 소요 시간 (나노초 단위)
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @class MeshAdaptiveDirectSearch
 * @brief MADS (Mesh Adaptive Direct Search) 알고리즘 정적 클래스
 *
 * @details MADS 알고리즘은 함수의 미분 정보(기울기 등)를 사용하지 않는 Derivative-Free
 * Optimization(DFO) 기법입니다. 목적 함수의 평가(Evaluation)만을 기반으로 탐색을 진행하므로, 미분
 * 불가능한 지점이 있거나 노이즈가 많은 환경에서도 강건하게 동작합니다. 본 구현체는 매 스텝마다 탐색
 * 방향으로 무작위 'Positive Spanning Set(양의 생성 집합)'을 생성하여 지역 최솟값에 빠지는 것을
 * 방지하며 탐색 범위를 조절합니다.
 *
 * @note 탐색 기저 생성에 무작위성(Randomness)이 포함되어 있으므로 결과가 비결정론적일 수 있습니다.
 *       따라서 엄격한 시간 제약이 있는 실시간(Real-Time) 제어 루프 내부보다는 오프라인 궤적 생성 및
 * 파라미터 튜닝용으로 적합합니다.
 */
class MeshAdaptiveDirectSearch {
   private:
    /**
     * @brief 랜덤 양의 생성 집합(Positive Spanning Set)을 생성하는 함수
     *
     * @details N차원 공간을 완전히 탐색하기 위해 최소 N+1개의 기저 벡터(방향)가 필요합니다.
     * 이 함수는 난수를 활용하여 랜덤한 하삼각 행렬(L)을 만들고, 행과 열을 섞은 후,
     * 마지막 N+1번째 벡터는 나머지 모든 기저 벡터들의 역방향 합으로 설정하여 공간 전체를 커버할 수
     * 있도록 만듭니다. 정적 배열(std::array)을 사용하여 O(1) 복잡도로 메모리 지연 없이 빠르게
     * 동작합니다.
     *
     * @param alpha 현재 스텝 크기(Mesh Size) 파라미터. 이 값에 따라 벡터들의 크기(delta)가 결정됨
     * @param gen 난수 생성기 참조 (성능을 위해 thread_local 엔진 사용)
     * @return N차원 벡터 N+1개로 구성된 방향 행렬 D
     */
    template <size_t N, typename Generator>
    [[nodiscard]] static std::array<std::array<double, N>, N + 1> generate_positive_spanning_set(
        double alpha, Generator& gen) {
        // 캐시 라인(64 byte) 정렬을 통해 SIMD 연산 및 메모리 접근 속도 최적화
        alignas(64) std::array<std::array<double, N>, N + 1> D = {0.0};
        alignas(64) std::array<std::array<double, N>, N> L = {0.0};

        // 1. 델타 (Delta) 크기 결정
        // alpha 값이 작아질수록 델타가 커져 더 넓은 범위의 기저 방향들을 생성하게 됨
        int delta = static_cast<int>(std::round(1.0 / std::sqrt(alpha)));
        if (delta < 1) {
            delta = 1;
        }

        // 난수 분포 설정
        std::uniform_int_distribution<int> sign_dist(0, 1);  // 0 또는 1 반환 (부호 결정용)
        std::uniform_int_distribution<int> lower_dist(-delta + 1, delta - 1);  // 하삼각 요소 범위

        // 2. 하삼각 행렬 (Lower Triangular Matrix) L 생성
        for (size_t i = 0; i < N; ++i) {
            // 대각 성분은 ±delta 로 강제하여 선형 독립성을 보장
            L[i][i] = (sign_dist(gen) == 1) ? delta : -delta;
            for (size_t j = 0; j < i; ++j) {
                // 대각선 아래 값들은 범위 내의 임의의 정수 할당
                L[i][j] = lower_dist(gen);
            }
        }

        // 3. 행과 열을 랜덤하게 섞기 (Permutation)
        // 이 과정을 통해 하삼각 행렬 구조에 편향되지 않은 균일한 방향 탐색을 수행함
        std::array<size_t, N> row_p, col_p;
        std::iota(row_p.begin(), row_p.end(), 0);  // 0, 1, 2, ... N-1 할당
        std::iota(col_p.begin(), col_p.end(), 0);

        // 배열을 섞어서 무작위 순열 생성
        std::shuffle(row_p.begin(), row_p.end(), gen);
        std::shuffle(col_p.begin(), col_p.end(), gen);

        // 4. 방향 행렬 D 계산 및 N + 1 번째 방향 (Negative Sum) 추가
        for (size_t d_idx = 0; d_idx < N; ++d_idx) {
            for (size_t dim = 0; dim < N; ++dim) {
                // 섞인 순열 인덱스를 활용하여 L 행렬에서 값 복사
                double val = L[row_p[dim]][col_p[d_idx]];
                D[d_idx][dim] = val;

                // N+1 번째 벡터 (인덱스 N)는 나머지 N개 벡터의 역방향 합산 (Negative Sum)
                // 이렇게 해야 N+1개의 벡터가 공간의 모든 방향을 양의 조합(Positive Span)으로 표현할
                // 수 있음
                D[N][dim] -= val;
            }
        }
        return D;
    }

   public:
    MeshAdaptiveDirectSearch() = delete;  // 정적 메서드만 제공하므로 인스턴스화 방지

    /**
     * @brief MADS 알고리즘을 수행하여 최적해를 찾는 함수
     *
     * @param f 최적화하려는 목적 함수 (미분 가능할 필요 없음)
     * @param x_init 탐색을 시작할 초기 위치 (N차원 벡터)
     * @param tol 종료 조건 허용 오차 (alpha가 이 값 이하로 떨어지면 탐색 종료)
     * @param max_iter 최대 반복 횟수 (기본값: 10000)
     * @param verbose 콘솔에 진행 과정을 출력할지 여부
     * @return 최적해(x_opt), 최적값(f_opt), 반복 횟수, 연산 시간을 포함하는 OptimizationResultND
     * 구조체
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          double tol = 1e-5,
                                                          size_t max_iter = 10000,
                                                          bool verbose = false) {
        static_assert(N > 0, "Dimension N must be greater than 0");  // 컴파일 타임 차원 검사

        // 연산 소요 시간 측정을 위한 타이머 시작
        auto start_clock = std::chrono::high_resolution_clock::now();

        double alpha = 1.0;  // 초기 스텝 사이즈 (Mesh 파라미터)
        alignas(64) std::array<double, N> x = x_init;  // 캐시 정렬된 현재 탐색 위치 변수
        double y = f(x);                               // 현재 위치의 목적 함수값

        // [핵심] 정적 thread_local 난수 생성기로 지연 (Latency) 원천 차단
        // 매 반복마다 생성기를 다시 인스턴스화하지 않아 난수 생성 오버헤드를 극도로 줄임
        static thread_local std::mt19937 gen(std::random_device{}());

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🕸️ MADS (Mesh Adaptive Direct Search) Started \n";
            std::cout << "========================================================\n";
        }

        size_t iter = 0;
        // alpha 값이 허용 오차보다 크고, 최대 반복 횟수를 넘지 않을 동안 탐색 지속
        while (alpha > tol && iter < max_iter) {
            iter++;
            bool improved = false;  // 현재 이터레이션에서 해의 개선이 있었는지 여부

            // alpha에 의존적인 N + 1개의 무작위 탐색 방향 세트(Positive Spanning Set) 획득
            auto D = generate_positive_spanning_set<N>(alpha, gen);

            // [기회주의적 탐색 (Opportunistic Polling)]
            // 모든 방향을 다 평가하는 것이 아니라, 더 좋은 해를 찾는 즉시 방향 탐색을 멈추고 이동함
            for (size_t i = 0; i <= N; ++i) {
                alignas(64) std::array<double, N> x_prime = {0.0};

// OpenMP SIMD 프라그마를 사용하여 벡터 연산 가속 (컴파일러 지원 시)
#pragma omp simd
                for (size_t j = 0; j < N; ++j) {
                    // x_prime = x + alpha * D[i]
                    // std::fma(Fused Multiply-Add)를 사용하여 하드웨어 레벨에서 단일 명령어로
                    // 고속/고정밀 계산
                    x_prime[j] = std::fma(alpha, D[i][j], x[j]);
                }

                double y_prime = f(x_prime);  // 새로운 위치에서 함수 평가

                // 만약 새로운 위치가 기존보다 더 낫다면 (최소화 문제)
                if (y_prime < y) {
                    x = x_prime;
                    y = y_prime;
                    improved = true;

                    // [가속 스텝 (Accelerated Step)]
                    // 현재 방향으로의 성과가 좋으므로, 같은 방향으로 더 멀리(3배) 뻗어봄
                    alignas(64) std::array<double, N> x_exp = {0.0};
#pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        x_exp[j] = std::fma(3.0 * alpha, D[i][j], x[j]);
                    }

                    double y_exp = f(x_exp);  // 가속 지점에서 함수 평가

                    // 더 멀리 갔더니 결과가 더 좋으면 가속 위치 채택
                    if (y_exp < y) {
                        x = x_exp;
                        y = y_exp;
                    }

                    // 더 나은 점을 하나라도 발견했으므로, 나머지 방향 평가는 건너뛰고 바로 루프
                    // 탈출 (기회주의적 탐색 방식이 탐색 횟수를 획기적으로 줄여줌)
                    break;
                }
            }

            // 성과(improved)에 따른 Mesh 크기 및 스텝 폭 조절
            if (improved) {
                // 탐색에 성공했다면 방향의 해상도(알파)를 키움
                // 다만 최대 alpha 값을 1.0으로 제한하여 스텝이 무한정 커지는 것 방지
                alpha = std::min(4.0 * alpha, 1.0);
            } else {
                // 어떤 방향으로도 개선하지 못했다면 현재 점이 골짜기 내부에 있을 수 있음
                // 탐색 반경(스텝 크기)을 1/4로 줄여서 더 촘촘하게 국소적 탐색을 진행함
                alpha /= 4.0;
            }

            // 진행 상태 출력: 개선에 실패했거나(구간이 축소될 때), 10회 주기마다 출력
            if (verbose && (!improved || iter % 10 == 0)) {
                std::cout << "[Iter " << std::setw(4) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << y << " | alpha: " << alpha << "\n";
            }
        }

        // 탐색 완료 후 시간 측정 종료
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

        // 최적해와 부가 정보를 결과 구조체에 담아 반환
        return {x, y, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_MADS_HPP_