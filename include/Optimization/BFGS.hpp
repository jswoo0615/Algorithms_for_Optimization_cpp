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
 * @brief N차원 최적화 결과를 담는 범용 구조체
 *
 * 최적화 알고리즘의 수행 결과를 사용자에게 반환하기 위해 사용되는 데이터 컨테이너입니다.
 * 모든 최적화 기법에서 공통된 반환 형식을 갖도록 하여 코드의 재사용성과 일관성을 높입니다.
 *
 * @tparam N 최적화 대상 변수의 차원(dimension) 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt;  ///< 도출된 최종 최적해 (Optimal Point) 벡터
    double f_opt;  ///< 해당 최적해(x_opt)를 목적 함수에 대입했을 때의 함숫값 (최솟값)
    size_t iterations;  ///< 수렴 조건(허용 오차 이내 도달)을 만족하기까지 반복한 총 횟수
    long long elapsed_ns;  ///< 최적화 수행에 걸린 전체 시간 (나노초, nanoseconds 단위)
};
}  // namespace Optimization
#endif

namespace Optimization {

/**
 * @brief BFGS (Broyden–Fletcher–Goldfarb–Shanno) 알고리즘을 이용한 무제약 최적화(Unconstrained
 * Optimization) 클래스
 *
 * BFGS 알고리즘은 준-뉴턴(Quasi-Newton) 계열의 최적화 기법 중 가장 널리 쓰이며 강력하고 안정적인
 * 성능을 자랑합니다.
 *
 * [수학적 배경 및 원리]
 * 순수 뉴턴법(Newton's Method)은 최적화 시 2차 미분인 헤시안(Hessian) 행렬 H와 그 역행렬 H^{-1}을
 * 매번 정확하게 계산해야 합니다. 이는 변수의 차원 N이 커질수록 O(N^3)의 막대한 연산량과 메모리를
 * 요구하는 큰 단점이 있습니다.
 *
 * 이를 극복하기 위해 준-뉴턴 기법은 실제 역헤시안 행렬을 구하는 대신, 반복 과정에서 얻어지는
 * 1차 미분(Gradient) 정보의 변화량을 누적하여 **역헤시안의 근사 행렬(Inverse Hessian Approximation
 * Matrix, V)**을 점진적으로 업데이트해 나갑니다. 이렇게 함으로써 역행렬을 계산하는 과정을 생략하고,
 * 행렬-벡터 곱 연산인 O(N^2)의 복잡도만으로 순수 뉴턴법에 필적하는 초선형 수렴 속도(Superlinear
 * Convergence)를 달성할 수 있습니다.
 *
 * [Sherman-Morrison-Woodbury 공식 활용]
 * BFGS 업데이트는 Rank-2 업데이트 방식으로 이루어지며, 헤시안 행렬의 근사치를 업데이트한 뒤
 * 역행렬을 취하는 대신, 아예 역헤시안의 근사치 자체를 직접 업데이트하는 수식을 적용합니다. 이는 매
 * 반복마다 계산 효율성을 극대화합니다.
 *
 * @note 본 구현체는 동적 메모리 할당(예: std::vector)을 완전히 배제하고, `std::array`를 이용한
 *       스택(Stack) 또는 정적(Static) 메모리 기반으로 설계되었습니다. 이를 통해 힙 메모리 할당 지연
 * 및 단편화를 방지하고, 제어 주기가 매우 빠른 로보틱스 등의 실시간(Real-Time) 환경과 자원이 제한된
 * 임베디드(Embedded) 시스템에 매우 적합합니다. 또한, OpenMP SIMD 지시어를 활용하여 하드웨어
 * 가속(Vectorization)을 통한 성능 극대화를 도모하였습니다.
 */
class BFGS {
   private:
    /**
     * @brief N차원 공간에서 두 벡터 간의 내적(Dot Product)을 계산하는 인라인 유틸리티 함수
     *
     * 두 벡터 a와 b에 대해 Σ(a_i * b_i) 를 수행합니다.
     *
     * @tparam N 벡터의 차원 수
     * @param a 첫 번째 입력 벡터
     * @param b 두 번째 입력 벡터
     * @return 두 벡터의 내적 결과 (스칼라 값)
     *
     * @note `std::fma` (Fused Multiply-Add)를 사용하여 (a * b) + c 연산을 하드웨어 단일 클럭으로
     * 수행하여 수치적 안정성을 높이고 부동소수점 오차를 줄입니다.
     *       `#pragma omp simd` 지시어를 통해 컴파일러가 SIMD(Single Instruction Multiple Data)
     * 명령어로 루프를 병렬 최적화(예: AVX, SSE)하도록 유도합니다.
     */
    template <size_t N>
    [[nodiscard]] static constexpr double dot(const std::array<double, N>& a,
                                              const std::array<double, N>& b) noexcept {
        double sum = 0.0;
#pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            sum = std::fma(a[i], b[i], sum);  // 단일 인스트럭션으로 내적 연산 수행
        }
        return sum;
    }

    /**
     * @brief N x N 정방 행렬과 N x 1 열 벡터의 곱셈을 수행하는 함수
     *
     * 역헤시안 근사 행렬 V와 기울기 벡터 g를 곱하여 뉴턴 탐색 방향 (p = -Vg)을 찾을 때 주로
     * 사용됩니다.
     *
     * @tparam N 행렬의 행/열 크기 및 벡터의 차원 수
     * @param M 대상 정방 행렬 (2차원 `std::array`)
     * @param v 대상 열 벡터 (1차원 `std::array`)
     * @return 계산된 결과 벡터 (M * v)
     */
    template <size_t N>
    [[nodiscard]] static constexpr std::array<double, N> mat_vec_mul(
        const std::array<std::array<double, N>, N>& M, const std::array<double, N>& v) noexcept {
        std::array<double, N> res = {0.0};
        // 행렬의 각 행(row)에 대해 반복 계산
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
// 각 행의 열(column) 요소들과 벡터 요소를 내적합니다. SIMD 최적화 적용.
#pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                sum = std::fma(M[i][j], v[j], sum);
            }
            res[i] = sum;
        }
        return res;
    }

   public:
    // BFGS 클래스는 내부 상태를 가지지 않는 순수 알고리즘 집합체이므로 인스턴스화(객체 생성)를
    // 완전히 금지합니다.
    BFGS() = delete;

    /**
     * @brief 주어진 목적 함수를 최소화하는 해를 찾는 BFGS 알고리즘의 메인 함수
     *
     * 이 함수는 주어진 초기 추정치에서 시작하여, 목적 함수의 기울기가 허용치 이하로 수렴할 때까지
     * 변수의 위치를 반복적으로 업데이트합니다. 자동 미분(AutoDiff)을 사용하여 미분 오차 없이
     * 정확하게 기울기를 구하며, 백트래킹 선 탐색(Backtracking Line Search)으로 강건하게 스텝
     * 사이즈(이동 거리)를 조절합니다.
     *
     * @tparam N 최적화하려는 변수 벡터의 차원
     * @tparam Func 목적 함수의 타입. 람다 함수(Lambda), 함수 포인터, 펑터(Functor) 등을 모두
     * 허용합니다.
     * @param f 스칼라 값을 반환하는 최적화 목적 함수 (예: `double f(const std::array<double, N>&
     * x)`)
     * @param x_init 탐색을 시작할 초기 위치 벡터 (Initial Guess)
     * @param tol 허용 오차 (Tolerance). 현재 위치에서의 기울기 벡터의 크기(L2-norm의 제곱)가 이 값
     * 미만이면 극솟값에 충분히 가까워졌다고 판단하고 최적화 루프를 조기 종료합니다. 기본값은 1e-6.
     * @param max_iter 최적화가 수렴하지 않을 경우를 대비한 최대 반복 횟수(안전 장치). 무한 루프를
     * 방지합니다. 기본값은 200.
     * @return OptimizationResultND<N> 최적화가 끝난 후의 최종 상태(최적해, 함숫값, 반복 횟수, 소요
     * 시간)를 담은 구조체
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_init,
                                                          double tol = 1e-6,
                                                          size_t max_iter = 200) noexcept {
        // 성능 분석 및 최적화 수행 시간을 측정하기 위한 고해상도 타이머 시작
        auto start_clock = std::chrono::high_resolution_clock::now();

        std::array<double, N> x = x_init;  // 현재 추정 위치 변수 (초기값으로 세팅)
        const double tol_sq = tol * tol;  // 매 반복마다 발생하는 sqrt 연산을 방지하고자 허용 오차를
                                          // 제곱 형태로 변환하여 비교

        // V : 역헤시안 근사 행렬 (Inverse Hessian Approximation Matrix)
        // 메모리를 64바이트(일반적인 CPU L1 캐시 라인 크기) 단위로 정렬(alignas)하여 메모리 접근
        // 병목 현상을 줄이고, 캐시 히트율(Cache Hit Rate) 및 SIMD 벡터 연산 효율성을 극대화합니다.
        alignas(64) std::array<std::array<double, N>, N> V = {0.0};

        // 초기화 단계:
        // 알고리즘 초기 단계에서는 목적 함수의 곡률(2차 미분)에 대한 정보가 전혀 없으므로,
        // 역헤시안 근사 행렬 V를 항등 행렬(Identity Matrix, 대각 원소가 1인 행렬)로 초기화합니다.
        // 이는 첫 번째 반복 시 탐색 방향이 순수 경사 하강법(Steepest Gradient Descent, p = -g)과
        // 완전히 동일해지도록 만들어 초기 안정성을 확보하기 위함입니다.
        for (size_t i = 0; i < N; ++i) {
            V[i][i] = 1.0;
        }

        double f_val = 0.0;               // 현재 위치 x에서의 목적 함수 평가값 저장
        std::array<double, N> g = {0.0};  // 현재 위치 x에서의 1차 기울기(Gradient) 벡터 저장

        // 최적화 루프 진입 전, 초기 위치에서의 목적 함수 값과 1차 미분(Gradient) 값을 자동
        // 미분(AutoDiff) 기법으로 동시 계산
        AutoDiff::value_and_gradient<N>(f, x, f_val, g);

        size_t iter = 0;
        // ==========================================
        // 메인 최적화 반복(Iteration) 루프 시작
        // ==========================================
        for (iter = 1; iter <= max_iter; ++iter) {
            // ------------------------------------------
            // 1. 수렴 검사 (Convergence Check)
            // ------------------------------------------
            // 현재 기울기 벡터의 제곱합(L2-norm의 제곱)을 계산
            double g_norm_sq = dot<N>(g, g);
            // 기울기가 사실상 0에 수렴했다면(tol_sq 미만), 현재 위치가 평고점(극솟값 또는 안장점)에
            // 도달한 것으로 간주하고 루프 조기 탈출
            if (g_norm_sq < tol_sq) {
                break;
            }

            // ------------------------------------------
            // 2. 탐색 방향(Search Direction, p) 결정
            // ------------------------------------------
            // 순수 뉴턴법의 탐색 방향 공식은 p = -H^{-1} * g 입니다.
            // BFGS에서는 H^{-1} 대신 역헤시안의 근사 행렬인 V를 사용하므로 p = -V * g 가 됩니다.
            // 우선 행렬 V와 기울기 벡터 g를 곱합니다.
            std::array<double, N> p = mat_vec_mul<N>(V, g);

            // 구해진 결과의 부호를 반전시켜, 목적 함수값이 가장 가파르게 감소하는 하강 방향(Descent
            // Direction)을 취합니다.
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                p[i] = -p[i];
            }

            // ------------------------------------------
            // 3. 선 탐색 (Line Search)
            // ------------------------------------------
            // 찾은 탐색 방향 p를 따라 "얼마만큼의 크기(Step Size, alpha)"로 이동해야
            // 함수값이 충분히 감소하는지(Armijo Condition 만족 등)를 결정합니다.
            // 뉴턴 기반 기법은 스케일이 자체적으로 맞춰져 있으므로, 이론적으로 최적의 이동 거리
            // alpha는 1.0에 가깝습니다. 따라서 기본 스텝 사이즈 1.0에서 시작하여 백트래킹(조건을
            // 만족할 때까지 0.5씩 곱하여 스텝을 줄임)을 수행합니다. (파라미터: 초기값 1.0, 감소율
            // 0.5, c1 파라미터 1e-4, wolfe 조건 강제 여부 false)
            double alpha =
                BacktrackingLineSearch::search<N>(f, x, p, f_val, g, 1.0, 0.5, 1e-4, false);

            // ------------------------------------------
            // 4. 새로운 위치(x_new) 산출 및 함수 재평가
            // ------------------------------------------
            std::array<double, N> x_new = {0.0};
            // 결정된 이동 거리 alpha와 방향 p를 바탕으로 다음 위치를 업데이트 (x_new = x + alpha *
            // p)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                x_new[i] = std::fma(alpha, p[i], x[i]);
            }

            // 새롭게 이동한 위치(x_new)에서의 목적 함수 값(f_new)과 새로운 기울기 벡터(g_new)를
            // 계산
            double f_new = 0.0;
            std::array<double, N> g_new = {0.0};
            AutoDiff::value_and_gradient<N>(f, x_new, f_new, g_new);

            // ------------------------------------------
            // 5. 역헤시안 근사 행렬(V) 업데이트를 위한 벡터 변화량 계산
            // ------------------------------------------
            std::array<double, N> s = {0.0};  // 위치 변화 벡터 (Step vector): s = x_new - x
            std::array<double, N> y = {0.0};  // 기울기 변화 벡터 (Yield vector): y = g_new - g
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                s[i] = x_new[i] - x[i];
                y[i] = g_new[i] - g[i];
            }

            // ------------------------------------------
            // 6. 곡률 조건 (Curvature Condition) 검사 및 V 행렬 갱신
            // ------------------------------------------
            // 두 벡터의 내적 (y^T * s) 연산
            double y_dot_s = dot<N>(y, s);

            // [곡률 조건(Curvature Condition)]
            // BFGS 알고리즘이 안정적으로 동작하여 최적해로 수렴하기 위해서는
            // 근사 행렬 V가 항상 양의 정부호(Positive Definite, 모든 고윳값이 양수) 성질을 유지해야
            // 합니다. 이를 수학적으로 보장하기 위한 필요충분조건(Secant Equation 제약 조건)이 바로
            // y^T * s > 0 입니다. 컴퓨터의 부동소수점 수치적 오차를 고려하여 단순 0이 아닌
            // 1e-10(아주 작은 양수)보다 큰지 엄격하게 검사합니다.
            if (y_dot_s > 1e-10) {
                // 조건을 만족하면 Sherman-Morrison-Woodbury 공식을 전개한 형태의 역헤시안
                // 업데이트(Rank-2 Update)를 수행합니다. 정석적인 BFGS 역헤시안 업데이트 공식: V_new
                // = (I - rho * s * y^T) * V * (I - rho * y * s^T) + rho * s * s^T 이를 실제
                // 구현에서 행렬 곱셈 연산량을 줄이기 위해 수학적으로 전개하여 최적화된 공식: V_new
                // = V - rho*(V*y*s^T + s*y^T*V) + rho*(rho*y^T*V*y + 1)*s*s^T

                const double rho = 1.0 / y_dot_s;  // 곡률 스케일링 팩터 (rho)
                auto Vy = mat_vec_mul<N>(V, y);    // 중간 벡터 행렬 곱 연산: V * y
                const double yVy = dot<N>(y, Vy);  // 스칼라 값 연산: y^T * V * y

                // 모든 업데이트 항에 공통으로 곱해지는 스칼라 계수 항 계산: rho * (1 + rho * (y^T *
                // V * y))
                const double scalar_term = rho * (1.0 + rho * yVy);

                // 행렬 V 업데이트 로직
                // 근사 행렬 V는 본질적으로 대칭 행렬(Symmetric Matrix)입니다.
                // 따라서 N^2 번의 연산을 모두 수행하지 않고, 상삼각(Upper Triangular) 부분만 직접
                // 계산한 뒤 하삼각 부분은 대칭 복사(Symmetric Copy)하여 연산 속도를 크게
                // 향상시킵니다.
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = i; j < N; ++j) {
                        // 공식에 따른 각 행렬 원소(i, j)별 업데이트 계산
                        double updated_val = V[i][j] + scalar_term * s[i] * s[j] -
                                             rho * (Vy[i] * s[j] + s[i] * Vy[j]);

                        V[i][j] = updated_val;
                        // 대각 원소가 아닌 경우(상삼각 부분인 경우), 대칭되는 하삼각 위치(j, i)에도
                        // 동일한 값 할당
                        if (i != j) {
                            V[j][i] = updated_val;
                        }
                    }
                }
            } else if (iter > 1) {
                // [리스타트 기법 (Restart Strategy)]
                // 선 탐색이 부정확하거나 최적화하는 목적 함수의 비선형성 및 굴곡이 너무 강해서,
                // 수치적 이유로 곡률 조건(y^T * s > 0)을 만족하지 못하는 경우가 발생할 수 있습니다.
                // 이 상태로 강제로 업데이트를 진행하면 V 행렬의 양의 정부호 성질이 깨져(Not
                // Positive Definite) 탐색 방향이 망가집니다. 이를 방지하기 위해, 이런 이상
                // 상황에서는 V 행렬에 누적된 정보가 오염되었다고 판단하고 행렬을 초기 상태인 단위
                // 행렬(Identity Matrix)로 완전히 초기화하여 안전한 경사 하강법 방향부터 다시
                // 시작(Restart)합니다.
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        V[i][j] = (i == j) ? 1.0 : 0.0;
                    }
                }
            }

            // ------------------------------------------
            // 7. 상태 갱신 (State Update)
            // ------------------------------------------
            // 다음 반복(Iteration)을 위해 현재 상태 정보를 새롭게 갱신된 정보로 치환합니다.
            x = x_new;
            f_val = f_new;
            g = g_new;
        }

        // 메인 최적화 루프 완전 종료 후 소요 시간 측정 종료
        auto end_clock = std::chrono::high_resolution_clock::now();
        // 측정된 두 시간차를 나노초(ns) 단위로 변환
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        // 도출된 최적해 벡터, 그 때의 목적 함수 값, 최종 수행된 반복 횟수, 경과 시간(나노초)을
        // 구조체로 묶어 반환
        return {x, f_val, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_BFGS_HPP_
