#ifndef OPTIMIZATION_DIRECT_HPP_
#define OPTIMIZATION_DIRECT_HPP_

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
    std::array<double, N> x_opt;  ///< 탐색이 완료된 최적의 매개변수 (위치) 배열
    double f_opt;                 ///< 최적의 위치에서 평가된 목적 함수의 값
    size_t iterations;            ///< 최적화 완료까지 수행된 반복 횟수
    long long elapsed_ns;         ///< 알고리즘 수행에 걸린 시간 (나노초 단위)
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @brief DIRECT (DIviding RECTangles) 전역 최적화 알고리즘 클래스
 * @details
 * DIRECT 알고리즘은 함수의 기울기(미분)를 필요로 하지 않는 Derivative-free 전역 최적화 기법입니다.
 * 립시츠 상수(Lipschitz constant)를 미리 알 필요 없이, 탐색 공간을 여러 개의
 * 하이퍼큐브(초직육면체)로 분할하며 전역 최솟값을 찾습니다.
 *
 * 탐색 공간은 항상 정규화된 [0, 1]^N 공간에서 다루어지며, 목적 함수를 평가할 때만 원래의 범위 [a,
 * b]로 변환(denormalize)합니다.
 *
 * @note 본 구현은 동적 메모리 할당(`std::vector` 등)을 배제하고 `std::array`를 활용한 정적 메모리
 * 풀(Static Pool) 패턴을 적용하여 실시간(Real-Time) 시스템 및 임베디드 환경에서의 안전성과 MISRA
 * C++ 준수를 고려하였습니다.
 *
 * @tparam N 최적화 문제의 차원 수
 * @tparam MAX_INTERVALS 분할 가능한 최대 하이퍼큐브(구간)의 개수 (기본값: 10000, 메모리 가용량에
 * 따라 조절 필요)
 */
template <size_t N, size_t MAX_INTERVALS = 10000>
class DIRECT {
   private:
    /**
     * @brief 탐색 공간을 구성하는 단일 하이퍼큐브 구간(Interval)을 나타내는 내부 구조체
     */
    struct Interval {
        alignas(64) std::array<double, N> c;  ///< 하이퍼큐브의 중앙점 좌표 (정규화된 [0, 1]^N 공간
                                              ///< 내의 위치)
        double y;                      ///< 해당 중앙점 c에서의 목적 함수 평가값 f(c)
        std::array<size_t, N> depths;  ///< 각 차원(축)별로 분할된 횟수 (분할될 때마다 +1)
        bool active;  ///< 정적 메모리 풀에서 해당 구간이 현재 유효한지(사용 중인지) 여부

        /**
         * @brief 현재 구간이 각 차원 중 가장 적게 분할된 횟수를 반환합니다.
         * @details DIRECT 알고리즘은 하이퍼큐브를 분할할 때, 가장 적게 분할된 차원(가장 긴 변)을
         * 우선적으로 분할합니다.
         * @return 모든 차원 중 최소 분할 횟수
         */
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

    /**
     * @brief 정규화된 [0, 1]^N 공간의 좌표 x를 실제 문제의 탐색 범위 [a, b]로 변환(역정규화)합니다.
     *
     * @param x 정규화된 공간 내의 좌표
     * @param a 실제 탐색 범위의 하한 (각 차원별 최소값)
     * @param b 실제 탐색 범위의 상한 (각 차원별 최대값)
     * @return 실제 공간 범위로 스케일링된 좌표
     */
    [[nodiscard]] static constexpr std::array<double, N> denormalize(
        const std::array<double, N>& x, const std::array<double, N>& a,
        const std::array<double, N>& b) noexcept {
        std::array<double, N> real_x = {0.0};
        // SIMD 지시어를 통해 벡터화 연산 수행
#pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            // real_x[i] = x[i] * (b[i] - a[i]) + a[i]
            // FMA(Fused Multiply-Add)를 사용하여 부동소수점 오차를 줄이고 연산 속도 향상
            real_x[i] = std::fma(x[i], b[i] - a[i], a[i]);
        }
        return real_x;
    }

   public:
    DIRECT() = delete;  // 유틸리티 클래스이므로 인스턴스화를 방지

    /**
     * @brief DIRECT 알고리즘을 수행하여 주어진 범위 내에서 전역 최적해를 탐색합니다.
     *
     * @tparam Func 목적 함수 타입
     * @param f 최소화하려는 목적 함수
     * @param a 탐색 범위의 하한 벡터 (Lower bound)
     * @param b 탐색 범위의 상한 벡터 (Upper bound)
     * @param epsilon 잠재적 최적 구간(Potentially Optimal Interval)을 판단하기 위한 허용 오차
     * (기본값: 1e-4)
     * @param max_iter 알고리즘의 최대 반복 횟수 (기본값: 100)
     * @param verbose 실행 과정 및 상태 출력 여부 (기본값: false)
     * @return 전역 최적해와 관련 정보가 담긴 OptimizationResultND 구조체
     */
    template <typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> a,
                                                          std::array<double, N> b,
                                                          double epsilon = 1e-4,
                                                          size_t max_iter = 100,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");  // 최소 1차원 이상 필요
        auto start_clock = std::chrono::high_resolution_clock::now();  // 타이머 시작

        // 정적 메모리 풀(Static Memory Pool)
        // 동적 메모리 할당(new, malloc)으로 인한 파편화나 지연을 막기 위해 thread_local로 큰 배열을
        // 스택/데이터 영역에 할당
        static thread_local std::array<Interval, MAX_INTERVALS> pool;
        for (size_t i = 0; i < MAX_INTERVALS; ++i) {
            pool[i].active = false;  // 풀 초기화
        }
        size_t pool_size = 0;  // 현재 풀에 활성화된(분할된) 구간의 개수

        // 1. 초기 하이퍼큐브 설정: 정규화된 공간의 정중앙 점(0.5, 0.5, ...)
        std::array<double, N> c_init = {0.0};
        for (size_t i = 0; i < N; ++i) {
            c_init[i] = 0.5;
        }

        // 첫 번째 구간을 풀에 등록하고 중앙점 함수값을 평가
        pool[0].c = c_init;
        pool[0].y = f(denormalize(c_init, a, b));
        for (size_t i = 0; i < N; ++i) {
            pool[0].depths[i] = 0;  // 아직 어느 차원으로도 분할되지 않음
        }
        pool[0].active = true;
        pool_size = 1;

        // 현재까지 발견된 가장 작은 함수값과 그 때의 좌표 기록
        double y_best = pool[0].y;
        alignas(64) std::array<double, N> c_best = pool[0].c;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🌐 DIRECT (Divided Rectangles) Search Started \n";
            std::cout << "========================================================\n";
        }

        // 잠재적 최적 구간의 인덱스들을 저장할 버퍼 (std::vector 대신 정적 배열 사용)
        static thread_local std::array<size_t, MAX_INTERVALS> S;
        size_t iter = 0;

        // 메인 최적화 루프
        for (iter = 1; iter <= max_iter; ++iter) {
            size_t S_size = 0;
            double min_y_val = y_best;

            // 2. 잠재적 최적 구간 (Potentially Optimal Intervals) 탐색 및 식별
            // 간략화된 구현: 현재 최고 기록(y_best)에 가까운(epsilon 내에 있는) 구간들을 후보로
            // 선정합니다. (정석적인 DIRECT는 립시츠 상수 하한을 이용해 볼록 껍질(Convex Hull)을
            // 구하여 식별합니다)
            for (size_t i = 0; i < pool_size; ++i) {
                if (pool[i].active && pool[i].y <= min_y_val + epsilon) {
                    S[S_size++] = i;
                }
            }

            // 3. 식별된 잠재적 최적 구간들을 분할(Division)
            for (size_t s_idx = 0; s_idx < S_size; ++s_idx) {
                size_t idx = S[s_idx];  // 분할할 부모 구간의 인덱스

                // 한 번의 분할로 차원 당 2개(좌/우)의 새 구간이 생기므로, 메모리 풀 용량을 사전
                // 검사
                if (pool_size + 2 * N > MAX_INTERVALS) {
                    break;
                }

                Interval& parent = pool[idx];
                size_t d =
                    parent.min_depth();  // 분할을 진행할 깊이 (가장 적게 분할된 차원부터 분할)

                // 각 차원을 순회하며 해당 차원이 최소 분할 상태(d)라면 분할을 수행
                for (size_t dim = 0; dim < N; ++dim) {
                    if (parent.depths[dim] == d) {
                        // 3-등분할 때 중앙점의 이동 거리 계산
                        // 깊이 d일 때 한 변의 길이는 3^(-d)에 비례하므로 중앙점 간의 간격은
                        // 3^(-(d+1))이 됨
                        double delta = std::pow(3.0, -static_cast<double>(d + 1));

                        // 3.1 왼쪽(Left) 구간 생성 및 평가
                        Interval left = parent;
                        left.c[dim] -= delta;   // 왼쪽으로 중심점 이동
                        left.depths[dim] += 1;  // 해당 차원 분할 횟수 증가
                        left.y = f(denormalize(left.c, a, b));  // 목적 함수 평가

                        // 3.2 오른쪽(Right) 구간 생성 및 평가
                        Interval right = parent;
                        right.c[dim] += delta;   // 오른쪽으로 중심점 이동
                        right.depths[dim] += 1;  // 해당 차원 분할 횟수 증가
                        right.y = f(denormalize(right.c, a, b));  // 목적 함수 평가

                        // 새 구간들을 풀에 추가
                        pool[pool_size++] = left;
                        pool[pool_size++] = right;

                        // 부모 구간도 해당 차원에 대해 분할되었으므로 깊이 증가
                        parent.depths[dim] += 1;

                        // 최고 기록(y_best) 갱신 여부 확인
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

            // 상태 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(6) << y_best << " | Active Intervals: " << pool_size
                          << "\n";
            }

            // 4. 종료 조건 검사: 메모리 풀이 가득 찼으면 조기 종료
            if (pool_size >= MAX_INTERVALS - 2 * N) {
                if (verbose) std::cout << "  ↳ ⚠️ MAX_INTERVALS capacity reached!\n";
                break;
            }
        }

        // 5. 최종 결과 반환
        // 최고 기록을 달성한 정규화 좌표를 실제 범위 좌표로 변환
        std::array<double, N> final_x = denormalize(c_best, a, b);

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

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
}  // namespace Optimization
#endif  // OPTIMIZATION_DIRECT_HPP_