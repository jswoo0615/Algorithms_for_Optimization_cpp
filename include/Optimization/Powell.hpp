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
 */
class PowellMethod {
   private:
    // 기저 벡터 집합 U를 단위 행렬로 초기화
    template <size_t N>
    static constexpr void reset_basis(std::array<std::array<double, N>, N>& U) noexcept {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                U[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

   public:
    PowellMethod() = delete;  // 인스턴스화 방지
    template <size_t N, typename Func, typename LineSearchFunc = GoldenSectionStrategy>
    [[nodiscard]] static OptimizationResultND<N> optimize(
        Func f, std::array<double, N> x_init, LineSearchFunc line_search = GoldenSectionStrategy{},
        double tol = 1e-5, size_t max_iter = 1000, bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        alignas(64) std::array<double, N> x = x_init;
        const double tol_sq = tol * tol;

        // 탐색 방향 (Search directions)을 저장할 캐시 정렬된 정적 배열 U
        alignas(64) std::array<std::array<double, N>, N> U = {0.0};
        reset_basis<N>(U);

        size_t iter = 0;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🌪️ Powell's Method Started \n";
            std::cout << "========================================================\n";
        }

        for (iter = 1; iter <= max_iter; ++iter) {
            alignas(64) std::array<double, N> x_prime = x;

            // 1. 현재 방향 세트 U의 모든 방향에 대해 순차적으로 선탐색 수행
            for (size_t i = 0; i < N; ++i) {
                alignas(64) std::array<double, N> d = U[i];
                x_prime = line_search(f, x_prime, d);
            }

            // 2. 가장 오래된 방향 U를 버리고 배열을 한 칸씩 앞으로 당김 (Shift)
            for (size_t i = 0; i < N - 1; ++i) {
                U[i] = U[i + 1];
            }

            // 3. 1 사이클 전체의 알짜 이동 방향 (d_new) 도출
            alignas(64) std::array<double, N> d_new = {0.0};
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                d_new[i] = x_prime[i] - x[i];
            }
            U[N - 1] = d_new;  // 새로운 이동 방향을 마지막 탐색 방향으로 등록

            // 4. 새로운 방향을 따라 시작점 (x)에서부터 다시 한 번 선탐색 수행
            alignas(64) std::array<double, N> x_next = line_search(f, x, d_new);

            // 이동 거리 (Norm) 제곱 계산 및 FMA 적용
            double delta_sq = 0.0;
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                const double diff = x_next[i] - x[i];
                delta_sq = std::fma(diff, diff, delta_sq);
            }

            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f(x_next)
                          << " | ||Δ||: " << std::sqrt(delta_sq) << "\n";
            }

            x = x_next;  // 위치 갱신

            // 종료 조건 확인
            if (delta_sq < tol_sq) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // [안전성 보장] 방향 벡터의 선형 종속(Linear dependence) 방지
            if (iter % (N + 1) == 0) {
                reset_basis<N>(U);
            }
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            // [버그 수정] std::cout << x 와 x[5] 하드코딩 오류 제거
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        return {x, f(x), iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_POWELL_HPP_