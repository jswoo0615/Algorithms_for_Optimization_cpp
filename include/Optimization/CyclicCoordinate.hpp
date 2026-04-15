#ifndef OPTIMIZATION_CYCLIC_COORDINATE_HPP_
#define OPTIMIZATION_CYCLIC_COORDINATE_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/LineSearch.hpp"

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

/**
 * @brief 선 탐색(Line Search)을 순환 좌표 탐색에 연결하는 전략 구조체
 * @details 
 * 최적화 과정에서 주어진 방향(d)으로 얼마나 이동할지(step size, alpha)를 결정하는 역할을 합니다.
 * 본 구조체는 다음 두 단계를 거쳐 최적의 이동 보폭을 찾습니다.
 * 1. Bracket Minimum (구간 추정): 최솟값이 포함될 것으로 예상되는 초기 구간 [a, b]를 찾습니다.
 * 2. Golden Section Search (황금 분할 탐색): 해당 구간 내에서 황금비를 이용해 구간을 좁혀가며 정밀하게 최솟값을 찾습니다.
 */
struct GoldenSectionStrategy {
    /**
     * @brief 함수 f에 대해 위치 x에서 방향 d로 이동할 때의 최적 위치를 반환합니다.
     * 
     * @tparam N 변수의 차원
     * @tparam Func 목적 함수의 타입
     * @param f 최적화하려는 목적 함수
     * @param x 현재 위치 벡터
     * @param d 탐색 방향 벡터 (통상적으로 좌표축 중 하나를 나타내는 단위 벡터)
     * @return 최적의 step size를 적용하여 이동한 새로운 위치 벡터
     */
    template <size_t N, typename Func>
    [[nodiscard]] std::array<double, N> operator()(Func f, const std::array<double, N>& x,
                                                   const std::array<double, N>& d) const noexcept {
        // 1. 최솟값이 존재하는 구간 [a, b] 탐색
        // 초기 step size는 1e-2, 확장 비율은 2.0으로 설정하여 최솟값이 포함된 구간을 찾습니다.
        auto [a, b] = LineSearch::bracket_minimum<N>(f, x, d, 1e-2, 2.0, false);

        // 2. 황금 분할 탐색으로 최적의 이동 보폭 (alpha) 계산
        // 앞서 찾은 구간 [a, b] 내에서 허용 오차 1e-5 내로 들어올 때까지 황금 분할 탐색을 수행합니다.
        double alpha = LineSearch::golden_section_search<N>(f, x, d, a, b, 1e-5, false);

        // 3. 최적 위치로 이동 (FMA 연산 적용)
        // x_new = x + alpha * d 공식을 FMA(Fused Multiply-Add)를 사용하여 빠르고 정확하게 계산합니다.
        // 메모리 정렬(alignas(64))과 OpenMP SIMD 지시어를 사용하여 벡터화 연산을 최적화합니다.
        alignas(64) std::array<double, N> x_new = {0.0};
#pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            x_new[i] = std::fma(alpha, d[i], x[i]);
        }
        return x_new;
    }
};

/**
 * @brief Cyclic Coordinate Search (순환 좌표 탐색) 알고리즘 클래스
 * @details 
 * 기울기(미분값)를 사용하지 않고(Derivative-free) 최적화를 수행하는 알고리즘입니다.
 * N차원 공간에서 각 좌표축 방향을 따라 순차적으로 1차원 선 탐색(Line Search)을 수행하며 최적점을 찾습니다.
 */
class CyclicCoordinateSearch {
   public:
    CyclicCoordinateSearch() = delete;  // 유틸리티 클래스이므로 인스턴스화를 방지합니다.

    // ================================================
    // Algorithm 7.2 : 기본 순환 좌표 탐색 (Basic Cyclic Coordinate Search)
    // ================================================
    /**
     * @brief 기본 순환 좌표 탐색을 수행합니다.
     * @details
     * 각 반복(iteration)마다 0번 축부터 N-1번 축까지 각 차원별로 순차적으로 선 탐색을 수행하여 위치를 갱신합니다.
     * 축 방향으로만 이동하므로 등고선이 축에 나란하지 않은 경우 지그재그(zigzag) 형태로 느리게 수렴할 수 있습니다.
     * 
     * @tparam N 문제의 차원 수
     * @tparam Func 목적 함수 타입
     * @tparam LineSearchFunc 선 탐색을 수행할 함수 객체 타입 (기본값: GoldenSectionStrategy)
     * @param f 최소화하려는 목적 함수
     * @param x_init 탐색을 시작할 초기 위치 (N차원 배열)
     * @param line_search 선 탐색 전략 객체 (기본 제공 객체 사용)
     * @param tol 수렴 판정 기준치 (허용 오차, 기본값: 1e-5)
     * @param max_iter 최대 반복 횟수 (기본값: 500)
     * @param verbose 실행 과정 출력 여부 (기본값: false)
     * @return 최적화 결과가 담긴 OptimizationResultND 구조체
     */
    template <size_t N, typename Func, typename LineSearchFunc = GoldenSectionStrategy>
    [[nodiscard]] static OptimizationResultND<N> optimize(
        Func f, std::array<double, N> x_init, LineSearchFunc line_search = GoldenSectionStrategy{},
        double tol = 1e-5, size_t max_iter = 500, bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0"); // 1차원 이상의 문제만 허용

        // 실행 시간 측정을 위한 타이머 시작
        auto start_clock = std::chrono::high_resolution_clock::now();

        // SIMD 최적화를 위해 캐시 라인(64바이트)에 맞게 메모리 정렬된 배열 사용
        alignas(64) std::array<double, N> x = x_init;
        const double tol_sq = tol * tol; // 거리 비교 시 루트 연산을 피하기 위해 오차 기준의 제곱값 사용
        size_t iter = 0;

        // 진행 상황 출력
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🧭 Cyclic Coordinate Search Started \n";
            std::cout << "========================================================\n";
        }

        // 최대 허용 반복 횟수만큼 반복 수행
        for (iter = 1; iter <= max_iter; ++iter) {
            // 이번 반복의 시작 위치를 기록 (이동 거리 계산용)
            alignas(64) std::array<double, N> x_prime = x;
            
            // 각 좌표축 단위로 순환하며 1차원 선 탐색 수행
            for (size_t i = 0; i < N; ++i) {
                // 탐색할 방향 벡터 초기화 (모든 성분 0.0)
                alignas(64) std::array<double, N> d = {0.0};
                d[i] = 1.0; // i번째 축 방향으로만 단위 벡터 설정
                
                // 해당 축 방향으로 선 탐색을 수행하여 위치 x를 갱신
                x = line_search(f, x, d);
            }

            // 이번 반복에서 N개 축을 모두 순환한 후 이동한 거리의 제곱 계산
            double delta_sq = 0.0;
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                const double diff = x[i] - x_prime[i]; // (현재 위치 - 시작 위치)
                delta_sq = std::fma(diff, diff, delta_sq); // delta_sq += diff * diff
            }

            // 지정된 주기(10번)마다 혹은 첫 번째 반복일 때 상태 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f(x) << " | ||Δ||: " << std::sqrt(delta_sq)
                          << "\n";
            }

            // 이동한 거리가 설정한 허용 오차보다 작으면 충분히 수렴했다고 판단하여 종료
            if (delta_sq < tol_sq) {
                break;
            }
        }

        // 실행 시간 측정 종료 및 결과 반환
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        return {x, f(x), iter, duration.count()};
    }

    // ================================================
    // Algorithm 7.3 : 가속 스텝을 포함한 순환 좌표 탐색 (Accelerated Cyclic Coordinate Search)
    // ================================================
    /**
     * @brief 가속 스텝(패턴 탐색)이 추가된 순환 좌표 탐색을 수행합니다.
     * @details
     * 기본 순환 좌표 탐색의 단점(협곡과 같은 지형에서 지그재그로 매우 느리게 수렴)을 극복하기 위해 제안된 방법입니다.
     * 모든 축에 대해 한 바퀴 탐색을 마친 후, (시작점 -> 현재점)을 잇는 방향(Pattern Direction)으로
     * 추가적인 선 탐색을 한 번 더 수행하여 수렴 속도를 높입니다. 이를 Powell의 방법의 기초 아이디어로도 볼 수 있습니다.
     * 
     * @tparam N 문제의 차원 수
     * @tparam Func 목적 함수 타입
     * @tparam LineSearchFunc 선 탐색 함수 객체 타입
     * @param f 최소화하려는 목적 함수
     * @param x_init 탐색 시작 초기 위치
     * @param line_search 선 탐색 전략 객체
     * @param tol 수렴 판정 기준치
     * @param max_iter 최대 반복 횟수
     * @param verbose 실행 과정 출력 여부
     * @return 최적화 결과가 담긴 OptimizationResultND 구조체
     */
    template <size_t N, typename Func, typename LineSearchFunc = GoldenSectionStrategy>
    [[nodiscard]] static OptimizationResultND<N> optimize_accelerated(
        Func f, std::array<double, N> x_init, LineSearchFunc line_search = GoldenSectionStrategy{},
        double tol = 1e-5, size_t max_iter = 500, bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");
        
        // 타이머 시작
        auto start_clock = std::chrono::high_resolution_clock::now();

        alignas(64) std::array<double, N> x = x_init;
        const double tol_sq = tol * tol;
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            // 사이클 시작 시점의 위치 기록
            alignas(64) std::array<double, N> x_prime = x;
            
            // 1단계: 각 축 방향별로 순환하며 선 탐색 수행 (기본 Cyclic Coordinate Search와 동일)
            for (size_t i = 0; i < N; ++i) {
                alignas(64) std::array<double, N> d = {0.0};
                d[i] = 1.0;
                x = line_search(f, x, d);
            }

            // 2단계: 가속 방향(패턴 방향, Pattern Direction) 계산
            // 이전 사이클 시작점(x_prime)에서 현재 점(x)으로 향하는 벡터를 구합니다.
            alignas(64) std::array<double, N> d_acc = {0.0};
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                d_acc[i] = x[i] - x_prime[i];
            }
            
            // 3단계: 가속 방향으로 추가적인 선 탐색 수행
            // 전체 진행 방향으로 크게 점프(이동)할 수 있어 좁은 협곡 등에서 빠른 수렴을 돕습니다.
            x = line_search(f, x, d_acc);

            // 4단계: 수렴 여부 확인을 위해 이동 거리 계산
            double delta_sq = 0.0;
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                const double diff = x[i] - x_prime[i]; // 한 사이클(가속 스텝 포함) 전체의 이동 거리
                delta_sq = std::fma(diff, diff, delta_sq);
            }

            // 수렴 조건을 만족하면 루프 탈출
            if (delta_sq < tol_sq) {
                break;
            }
        }

        // 실행 시간 측정 종료 및 반환
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        return {x, f(x), iter, duration.count()};
    }
};
}  // namespace Optimization
#endif  // OPTIMIZATION_CYCLIC_COORDINATE_HPP_