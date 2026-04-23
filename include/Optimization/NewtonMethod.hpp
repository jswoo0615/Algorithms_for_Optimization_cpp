#ifndef OPTIMIZATION_NEWTON_METHOD_HPP_
#define OPTIMIZATION_NEWTON_METHOD_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief 최적화 결과를 담는 구조체 (N차원 지원)
 */
template <size_t N>
struct OptimizationResult {
    std::array<double, N> x_opt;   ///< 최적해 (Optimal solution)
    double f_opt;                  ///< 최적 함수값 (Optimal function value)
    size_t iterations;             ///< 총 반복 횟수 (Total iterations)
    long long elapsed_ns;          ///< 실제 연산 소요 시간 (나노초 단위)
};

/**
 * @class NewtonMethod
 * @brief 뉴턴 메서드(Newton's Method) 기반 최적화 알고리즘
 * 
 * @details 뉴턴 메서드는 목적 함수의 2차 테일러 근사(Taylor Approximation)를 활용하여 최솟값을 찾는 방법입니다.
 * 1차 미분인 기울기(Gradient, g)와 2차 미분인 헤시안 행렬(Hessian Matrix, H)을 모두 사용하여
 * 최적점까지의 방향과 거리를 한 번에 도출하므로, 수렴 속도(2차 수렴, Quadratic Convergence)가 매우 빠릅니다.
 * 
 * 수식:
 * x_{t+1} = x_t - H^{-1} * g
 * (실제 구현에서는 역행렬 H^{-1}을 직접 구하지 않고 선형 방정식 H * p = -g 를 풀어 이동 방향 p를 구합니다.)
 * 
 * @note 본 구현체는 다음과 같은 성능 최적화를 포함합니다:
 * 1. N=2일 때는 Cramer's Rule을 이용한 초고속 해 도출 (역행렬 직접 계산 가능)
 * 2. N>2일 때는 동적 메모리 할당(힙 할당) 없는 정적 가우스 소거법(Static Gaussian Elimination) 적용
 * 3. 평탄한 지형(Singular Matrix)을 만나면 std::nullopt를 반환하여 안전하게 조기 종료
 */
class NewtonMethod {
   private:
    /**
     * @brief 뉴턴 시스템 선형 방정식 해 풀이 (H * p = -g 를 만족하는 이동 방향 p 계산)
     * 
     * @details 역행렬 계산 부하를 피하기 위해 선형 방정식을 직접 풉니다.
     * @note N=2인 경우 Cramer's Rule로 초고속 처리, N>2인 경우 정적 가우스 소거법(O(N^3)) 적용
     * 
     * @param H 헤시안 행렬 (N x N)
     * @param g 기울기 벡터 (N차원)
     * @return 방정식의 해(이동 벡터 p). 단, 행렬이 특이(Singular)하여 해를 구할 수 없는 경우 std::nullopt 반환
     */
    template <size_t N>
    [[nodiscard]] static constexpr std::optional<std::array<double, N>> solve_newton_system(
        std::array<std::array<double, N>, N> H, std::array<double, N> g) noexcept {
        
        if constexpr (N == 2) {
            // [2차원 최적화] 역행렬 공식을 Cramer's rule 기반으로 풀이
            // 판별식(Determinant) 계산: H[0][0]*H[1][1] - H[0][1]*H[1][0]
            // std::fma(Fused Multiply-Add)를 사용하여 하드웨어 단일 명령어로 빠르고 정밀하게 처리
            const double det = std::fma(H[0][0], H[1][1], -H[0][1] * H[1][0]);

            // 판별식이 0에 매우 가까우면 역행렬이 존재하지 않음 (지형이 완전 평탄하거나 안장점(Saddle Point))
            if (std::abs(det) < 1e-15) {
                return std::nullopt;  // Singular Matrix 대처
            }

            const double inv_det = 1.0 / det;
            
            // 2x2 역행렬 공식: 
            // H^{-1} = inv_det * [ H[1][1]  -H[0][1] ]
            //                    [ -H[1][0]  H[0][0] ]
            // p = -H^{-1} * g 계산 후 바로 반환
            return std::array<double, 2>{-inv_det * std::fma(H[1][1], g[0], -H[0][1] * g[1]),
                                         -inv_det * std::fma(H[0][0], g[1], -H[1][0] * g[0])};
        } else {
            // [N차원 최적화] 정적 가우스 소거법 (Static Gaussian Elimination)
            // 동적 배열(std::vector) 할당(new/malloc)을 없애 속도 최적화
            for (size_t i = 0; i < N; ++i) {
                // Partial Pivoting: 수치적 안정성을 위해 절댓값이 가장 큰 원소를 피벗으로 선택
                size_t pivot = i;
                for (size_t j = i + 1; j < N; ++j) {
                    if (std::abs(H[j][i]) > std::abs(H[pivot][i])) pivot = j;
                }
                // 행 교환
                std::swap(H[i], H[pivot]);
                std::swap(g[i], g[pivot]);

                // 피벗 값이 0에 너무 가까우면 해를 구할 수 없음 (Singular)
                if (std::abs(H[i][i]) < 1e-15) return std::nullopt;

                // 전진 소거 (Forward Elimination)
                for (size_t j = i + 1; j < N; ++j) {
                    const double factor = H[j][i] / H[i][i];
                    for (size_t k = i; k < N; ++k) H[j][k] -= factor * H[i][k];
                    g[j] -= factor * g[i];
                }
            }

            // 후진 대입 (Backward Substitution)
            std::array<double, N> p = {0.0};
            for (size_t i = N; i-- > 0;) {
                double sum = 0.0;
                for (size_t j = i + 1; j < N; ++j) {
                    sum = std::fma(H[i][j], p[j], sum);
                }
                // 원래 풀어야 하는 방정식은 H * p_raw = g 였으나,
                // 업데이트 공식을 위해 방향 반전 (-H^-1 * g)이 필요하므로 부호를 뒤집음
                p[i] = -(g[i] - sum) / H[i][i];  
            }
            return p;
        }
    }

   public:
    NewtonMethod() = delete; // 인스턴스화 방지

    /**
     * @brief 뉴턴 메서드를 사용하여 목적 함수의 최적해를 찾습니다.
     * 
     * @param f 최적화할 목적 함수
     * @param x_init 탐색을 시작할 초기 위치 (N차원 벡터)
     * @param tol 허용 오차. 기울기의 크기(Norm)가 이 값 미만이 되면 수렴한 것으로 간주 (기본값: 1e-6)
     * @param max_iter 최대 반복 횟수 (뉴턴 메서드는 수렴이 빠르므로 기본값 50으로도 충분)
     * @return 최적해(x_opt), 최적값(f_opt), 반복 횟수, 연산 시간을 포함하는 구조체
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResult<N> optimize(Func f, std::array<double, N> x_init,
                                                        double tol = 1e-6,
                                                        size_t max_iter = 50) noexcept {
        auto start_clock = std::chrono::high_resolution_clock::now(); // 타이머 시작

        std::array<double, N> x = x_init;
        
        // 분산/크기 비교를 위해 매번 제곱근(sqrt) 연산을 하는 비용을 없애기 위해 제곱 형태로 비교
        const double tol_sq = tol * tol;
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            double f_val = 0.0;
            std::array<double, N> g = {0.0};

            // 1. 현재 위치에서의 목적 함수값과 1차 미분(Gradient) 획득
            // 리포지토리에 자체 구현된 AutoDiff 엔진을 활용하여 O(1) 정적 할당만으로 기울기를 구합니다.
            AutoDiff::value_and_gradient<N>(f, x, f_val, g);

            // 2. 기울기 노름의 제곱(Gradient Norm Squared) 계산
            double g_norm_sq = 0.0;
// OpenMP SIMD 지시어를 통해 벡터 누적 연산을 가속 (컴파일러 지원 시)
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq); // g[i]*g[i] + g_norm_sq
            }

            // 기울기가 허용 오차보다 작으면 최적점(임계점)에 도달한 것으로 판단하고 조기 종료
            if (g_norm_sq < tol_sq) {
                break;
            }

            // 3. 현재 위치에서의 2차 미분(Hessian Matrix) 획득
            // 리포지토리의 AutoDiff.hpp에서 제공하는 Central Difference 기반의 헤시안 행렬을 구합니다.
            auto H = AutoDiff::hessian<N>(f, x);
            
            // 4. 뉴턴 스텝 계산 (선형 시스템 H * p = -g 풀이)
            auto p = solve_newton_system<N>(H, g);

            // 헤시안이 특이 행렬이어서 방향을 계산할 수 없다면 (지형 평탄) 최적화를 멈춥니다.
            if (!p) {
                break;  // 역행렬(선형 방정식) 실패 시 즉시 중단
            }

            // 5. 위치 업데이트
            // p 내부에는 이미 방향 반전(-H^-1 * g)이 반영되어 있으므로 단순 덧셈(+) 처리
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                x[i] += (*p)[i];  
            }
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        // 최종 위치(x)에서의 목적 함수값을 다시 평가하여 반환 구조체에 담아 리턴합니다.
        return {x, AutoDiff::value<N>(f, x), iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_NEWTON_METHOD_HPP_