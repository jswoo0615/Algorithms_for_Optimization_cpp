#ifndef OPTIMIZATION_POWELL_HPP_
#define OPTIMIZATION_POWELL_HPP_

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/CyclicCoordinate.hpp"

namespace Optimization {
/**
 * @brief Powell's Method 최적화 알고리즘 (교재 Algorithm 7.4)
 * @note 미분 정보 없이 선탐색만 사용하여 성공적인 이동 방향을 학습
 * MISRA C++ 준수, O(1) 메모리 및 컴파일 타임 SIMD 벡터화
 *
 * 파월(Powell) 방법은 목적 함수의 기울기(Gradient) 계산 없이 최적값을 찾는
 * 방향성 탐색(Directional Search) 기법 중 하나입니다.
 * N차원 공간에서 주어진 일련의 탐색 방향을 따라 1차원 선탐색(Line Search)을
 * 순차적으로 수행하고, 그 결과 얻어진 알짜 이동 방향(Net displacement)을
 * 새로운 탐색 방향으로 기저(Basis) 집합에 업데이트하는 방식으로 동작합니다.
 */
class PowellMethod {
   private:
    /**
     * @brief 탐색 방향(기저 벡터 집합) U를 단위 행렬(Identity matrix)로 초기화합니다.
     *
     * 최적화 초기에는 축 방향(Standard basis vectors)을 기본 탐색 방향으로 사용합니다.
     *
     * @tparam N 문제의 차원 수
     * @param U 탐색 방향 벡터들을 저장하는 N x N 배열
     */
    template <size_t N>
    static constexpr void reset_basis(std::array<std::array<double, N>, N>& U) noexcept {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                U[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

   public:
    PowellMethod() = delete;  // 유틸리티 클래스이므로 인스턴스화 방지

    /**
     * @brief Powell's Method를 사용하여 주어진 목적 함수 f의 최솟값을 탐색합니다.
     *
     * @tparam N 변수의 차원 (컴파일 타임 상수)
     * @tparam Func 목적 함수 타입 (std::array<double, N>을 입력받아 double 반환)
     * @tparam LineSearchFunc 1차원 선탐색 기법 타입 (기본: GoldenSectionStrategy)
     *
     * @param f 최적화할 목적 함수 (Objective function)
     * @param x_init 탐색을 시작할 초기 지점 (Initial point)
     * @param line_search 선탐색 알고리즘 객체 (기본값: 황금분할탐색)
     * @param tol 수렴 허용 오차 (최적점에 도달했다고 판단할 거리 임계값)
     * @param max_iter 최대 반복 횟수 제한 (무한 루프 방지)
     * @param verbose 콘솔에 진행 상황 및 로그를 출력할지 여부
     * @return OptimizationResultND<N> 최적해의 위치, 함숫값, 반복 횟수, 소요 시간을 담은 결과
     * 구조체
     */
    template <size_t N, typename Func, typename LineSearchFunc = GoldenSectionStrategy>
    [[nodiscard]] static OptimizationResultND<N> optimize(
        Func f, std::array<double, N> x_init, LineSearchFunc line_search = GoldenSectionStrategy{},
        double tol = 1e-5, size_t max_iter = 1000, bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");

        // 성능 측정을 위한 타이머 시작
        auto start_clock = std::chrono::high_resolution_clock::now();

        // 현재 탐색 위치 x (캐시 성능 향상을 위한 메모리 정렬 적용)
        alignas(64) std::array<double, N> x = x_init;

        // 수렴 조건 비교 시 sqrt() 연산을 피하기 위해 허용 오차 제곱값 계산
        const double tol_sq = tol * tol;

        // 탐색 방향 (Search directions)을 저장할 캐시 정렬된 정적 배열 U
        // 초기에는 기본 좌표축 방향(단위 행렬)으로 설정
        alignas(64) std::array<std::array<double, N>, N> U = {0.0};
        reset_basis<N>(U);

        size_t iter = 0;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🌪️ Powell's Method Started \n";
            std::cout << "========================================================\n";
        }

        // 최대 허용 반복 횟수만큼 최적화 루프 실행
        for (iter = 1; iter <= max_iter; ++iter) {
            // 이번 이터레이션 시작점 백업 (각 방향별 선탐색 시작점)
            alignas(64) std::array<double, N> x_prime = x;

            // 1. 현재 유지 중인 방향 세트 U의 모든 방향(d)에 대해 순차적으로 1차원 선탐색 수행
            for (size_t i = 0; i < N; ++i) {
                alignas(64) std::array<double, N> d = U[i];
                // 각 방향으로 선탐색하여 x_prime 위치 업데이트
                x_prime = line_search(f, x_prime, d);
            }

            // --- Powell 방법의 핵심: 새로운 켤레 방향(Conjugate Direction) 학습 ---

            // 2. 가장 오래된 방향(맨 앞, U[0])을 버리고 배열을 한 칸씩 앞으로 이동 (Shift)
            for (size_t i = 0; i < N - 1; ++i) {
                U[i] = U[i + 1];
            }

            // 3. 1 사이클(N번의 선탐색) 전체를 통해 이동한 "알짜 이동 방향" (d_new) 계산
            alignas(64) std::array<double, N> d_new = {0.0};
#pragma omp simd  // 벡터화(SIMD) 가속 지시어
            for (size_t i = 0; i < N; ++i) {
                d_new[i] = x_prime[i] - x[i];  // (사이클 후 위치) - (사이클 전 위치)
            }
            // 이 알짜 이동 방향을 새로운 마지막 탐색 방향으로 등록
            U[N - 1] = d_new;

            // 4. 새롭게 학습한 방향(d_new)을 따라 원래 시작점 (x)에서부터 다시 한 번 선탐색 수행
            alignas(64) std::array<double, N> x_next = line_search(f, x, d_new);

            // 5. 이번 이터레이션 동안의 총 이동 거리의 제곱(Norm squared) 계산
            //    (수렴 여부 판별 용도)
            double delta_sq = 0.0;
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                const double diff = x_next[i] - x[i];
                // Fused Multiply-Add 연산 (delta_sq = diff * diff + delta_sq) 최적화
                delta_sq = std::fma(diff, diff, delta_sq);
            }

            // 로그 출력
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f(x_next)
                          << " | ||Δ||: " << std::sqrt(delta_sq) << "\n";
            }

            // 현재 위치를 새 위치(x_next)로 갱신
            x = x_next;

            // 6. 종료 조건 (수렴) 확인: 이동 거리가 허용 오차 이하로 줄어들었으면 종료
            if (delta_sq < tol_sq) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 7. [안전성 보장] 방향 벡터의 선형 종속(Linear dependence) 방지
            // Powell 방법은 반복하다 보면 새 방향들이 선형 종속적으로 변하여 탐색 공간이 축소될 수
            // 있음. 이를 방지하기 위해 주기적으로 N+1 번째 이터레이션마다 기저 방향 U를 단위 행렬로
            // 초기화시킴.
            if (iter % (N + 1) == 0) {
                reset_basis<N>(U);
            }
        }

        // 타이머 종료 및 소요 시간 계산
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        // 최종 결과 로그 출력
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        // 반환 구조체 생성 및 리턴 (최적해, 최솟값, 반복 횟수, 걸린 시간)
        return {x, f(x), iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_POWELL_HPP_