#ifndef OPTIMIZATION_BFGS_HPP_
#define OPTIMIZATION_BFGS_HPP_

#include <array>
#include <chrono>
#include <cmath>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/BacktrackingLineSearch.hpp"

// 최적화 결과 반환용 구조체 일관성 유지 (중복 정의 방지)
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 담는 구조체
 * @tparam N 변수의 차원 수
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
 * @brief BFGS (Broyden–Fletcher–Goldfarb–Shanno) 최적화 알고리즘
 *
 * 준-뉴턴(Quasi-Newton) 기법 중 가장 널리 쓰이고 강력한 방법입니다.
 * 순수 뉴턴법(Newton's Method)은 매 반복마다 값비싼 헤시안(2차 미분 행렬)의 계산과 그 역행렬을 직접
 * 구해야 하는 단점이 있습니다.
 *
 * 반면 BFGS 알고리즘은 1차 미분(Gradient) 정보만을 누적하여 **역헤시안(Inverse Hessian) 행렬을
 * 근사**해 나갑니다. 이로 인해 역행렬을 구하는 과정 없이(O(N^2) 연산만으로) 뉴턴법에 가까운 2차
 * 수렴 속도(Quadratic Convergence)를 달성합니다.
 *
 * @note 동적 메모리(힙) 할당을 전면 배제하고, std::array 기반의 정적 메모리와 SIMD 가속을 사용하여
 *       실시간(RT) 환경 및 임베디드 제어 시스템에 적합하도록 구현되었습니다.
 */
class BFGS {
   private:
    /**
     * @brief 두 벡터의 내적 (Dot Product) 계산
     * @note 컴파일러 벡터화(#pragma omp simd)와 FMA 연산을 사용하여 하드웨어 최적화를 수행합니다.
     */
    template <size_t N>
    [[nodiscard]] static constexpr double dot(const std::array<double, N>& a,
                                              const std::array<double, N>& b) noexcept {
        double sum = 0.0;
#pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            sum = std::fma(a[i], b[i], sum);  // a[i] * b[i] + sum
        }
        return sum;
    }

    /**
     * @brief 정방 행렬과 벡터의 곱셈 (Matrix-Vector Multiplication)
     * @return std::array<double, N> 행렬 M과 벡터 v를 곱한 결과 벡터
     */
    template <size_t N>
    [[nodiscard]] static constexpr std::array<double, N> mat_vec_mul(
        const std::array<std::array<double, N>, N>& M, const std::array<double, N>& v) noexcept {
        std::array<double, N> res = {0.0};
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
#pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                sum = std::fma(M[i][j], v[j], sum);
            }
            res[i] = sum;
        }
        return res;
    }

   public:
    // 유틸리티 클래스로만 사용되도록 인스턴스화 금지
    BFGS() = delete;

    /**
     * @brief BFGS 알고리즘 메인 최적화 함수
     *
     * @tparam N 최적화 변수의 차원
     * @tparam Func 목적 함수 타입 (람다, 펑터 등)
     * @param f 최소화할 목적 함수
     * @param x_init 최적화 시작 지점 (초기 추정값)
     * @param tol 수렴 판정을 위한 허용 오차 (Gradient의 Norm이 이 값보다 작아지면 수렴으로 판단)
     * @param max_iter 무한 루프를 방지하기 위한 최대 허용 반복 횟수
     * @return OptimizationResultND<N> 최적해, 함수값, 반복 횟수, 소요 시간 정보
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          double tol = 1e-6,
                                                          size_t max_iter = 200) noexcept {
        auto start_clock = std::chrono::high_resolution_clock::now();

        std::array<double, N> x = x_init;
        const double tol_sq = tol * tol;  // 제곱 형태의 톨러런스 (루트 연산 방지용)

        // V : 역헤시안 근사 행렬 (Inverse Hessian Approximation Matrix)
        // 메모리 정렬(alignas 64)을 통해 캐시 히트율 및 SIMD 연산 효율을 높입니다.
        alignas(64) std::array<std::array<double, N>, N> V = {0.0};

        // 초기 역헤시안 근사 행렬을 단위 행렬(Identity Matrix)로 설정합니다.
        // 이는 초기에는 경사 하강법(Gradient Descent) 방향으로 시작함을 의미합니다.
        for (size_t i = 0; i < N; ++i) {
            V[i][i] = 1.0;
        }

        double f_val = 0.0;
        std::array<double, N> g = {0.0};

        // 초기 위치에서의 목적 함수 값과 기울기(Gradient) 평가
        AutoDiff::value_and_gradient<N>(f, x, f_val, g);

        size_t iter = 0;
        // 메인 최적화 루프
        for (iter = 1; iter <= max_iter; ++iter) {
            // 1. 수렴 검사 (기울기의 제곱합이 허용치보다 작으면 종료)
            double g_norm_sq = dot<N>(g, g);
            if (g_norm_sq < tol_sq) {
                break;  // 평고점(최적해) 도달
            }

            // 2. 탐색 방향(Search Direction, p) 계산
            // p = -V * g (역헤시안 근사 행렬과 기울기를 곱하여 뉴턴 방향 근사)
            std::array<double, N> p = mat_vec_mul<N>(V, g);
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                p[i] = -p[i];  // 방향을 음수(하강 방향)로 반전
            }

            // 3. 선 탐색 (Line Search)
            // 결정된 탐색 방향 p를 따라 얼마나 이동할지(alpha)를 결정합니다.
            // BFGS는 근사된 뉴턴 스텝을 따르므로 기본 스텝 사이즈 1.0부터 백트래킹을 시작합니다.
            double alpha =
                BacktrackingLineSearch::search<N>(f, x, p, f_val, g, 1.0, 0.5, 1e-4, false);

            // 4. 새로운 위치(x_new) 산출
            std::array<double, N> x_new = {0.0};
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                x_new[i] = std::fma(alpha, p[i], x[i]);  // x_new = x + alpha * p
            }

            // 새로운 위치에서의 함수 값과 기울기 평가
            double f_new = 0.0;
            std::array<double, N> g_new = {0.0};
            AutoDiff::value_and_gradient<N>(f, x_new, f_new, g_new);

            // 5. 역헤시안 근사 행렬(V) 업데이트를 위한 준비
            std::array<double, N> s = {0.0};  // 변위 벡터: s = x_new - x
            std::array<double, N> y = {0.0};  // 기울기 변화 벡터: y = g_new - g
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                s[i] = x_new[i] - x[i];
                y[i] = g_new[i] - g[i];
            }

            // 6. 곡률 조건 (Curvature Condition) 검사
            double y_dot_s = dot<N>(y, s);

            // y^T * s > 0 을 만족해야만 V 행렬이 양의 정부호(Positive Definite) 성질을 유지할 수
            // 있습니다.
            if (y_dot_s > 1e-10) {
                // Sherman-Morrison 공식을 전개한 역헤시안 업데이트 (Rank-2 Update)
                // 수학적 공식: V_new = V - rho*(V*y*s^T + s*y^T*V) + rho*(rho*y^T*V*y + 1)*s*s^T

                const double rho = 1.0 / y_dot_s;
                auto Vy = mat_vec_mul<N>(V, y);    // V * y
                const double yVy = dot<N>(y, Vy);  // y^T * V * y

                // 공통 스칼라 항 계산: rho * (1 + rho * y^T * V * y)
                const double scalar_term = rho * (1.0 + rho * yVy);

                // 행렬 V 업데이트 (대칭 행렬이므로 상삼각 부분만 계산하고 대칭 복사)
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = i; j < N; ++j) {
                        double updated_val = V[i][j] + scalar_term * s[i] * s[j] -
                                             rho * (Vy[i] * s[j] + s[i] * Vy[j]);

                        V[i][j] = updated_val;
                        if (i != j) {
                            V[j][i] = updated_val;  // 대칭성 유지
                        }
                    }
                }
            } else if (iter > 1) {
                // 수치적 이유로 곡률 조건을 만족하지 못한 경우 (y^T * s <= 0)
                // 행렬이 무너지는 것을 막기 위해 V를 안전한 초기 상태(단위 행렬)로 리셋합니다
                // (Restart).
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        V[i][j] = (i == j) ? 1.0 : 0.0;
                    }
                }
            }

            // 7. 상태 갱신 (다음 이터레이션을 위함)
            x = x_new;
            f_val = f_new;
            g = g_new;
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        // 결과 반환
        return {x, f_val, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_BFGS_HPP_