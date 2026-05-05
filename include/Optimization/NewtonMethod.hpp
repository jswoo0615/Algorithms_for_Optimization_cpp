#ifndef OPTIMIZATION_NEWTON_METHOD_HPP_
#define OPTIMIZATION_NEWTON_METHOD_HPP_

#include <chrono>

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {

/**
 * @brief 최적화 결과를 담는 구조체 (StaticVector 기반으로 업그레이드)
 */
template <size_t N>
struct OptimizationResult {
    StaticVector<double, N> x_opt;  ///< 최적해
    double f_opt;                   ///< 최적 함수값
    size_t iterations;              ///< 총 반복 횟수
    long long elapsed_ns;           ///< 연산 소요 시간 (나노초)
    SolverStatus status;            ///< 솔버 종료 상태
};

/**
 * @class NewtonMethod
 * @brief SIMD 및 Zero-Allocation 기반 뉴턴 최적화 알고리즘 (Unconstrained Optimizer)
 */
class NewtonMethod {
   private:
    /**
     * @brief [Architect's Engine] Hybrid Hessian & Gradient 계산기
     * 1차 미분(Gradient)은 DualVec(AD)로 완벽한 정밀도를 확보하고,
     * 2차 미분(Hessian)은 AD 결과의 중앙 차분(Central Difference)을 통해 고속 도출합니다.
     */
    template <size_t N, typename Func>
    static void compute_grad_and_hessian(const Func& f, const StaticVector<double, N>& x,
                                         double& f_val, StaticVector<double, N>& g,
                                         StaticMatrix<double, N, N>& H) noexcept {
        using ADVar = DualVec<double, N>;
        constexpr double h = 1e-5; // 차분 스텝
        constexpr double inv_2h = 1.0 / (2.0 * h);

        // 1. 현재 위치의 함수값 및 Gradient (AD 1회 평가)
        StaticVector<ADVar, N> x_dual;
        for (size_t i = 0; i < N; ++i) x_dual(i) = ADVar::make_variable(x(i), i);
        
        ADVar f_dual = f(x_dual);
        f_val = Optimization::get_value(f_dual);
        for (size_t i = 0; i < N; ++i) g(i) = f_dual.g[i];

        // 2. Hessian 계산 (AD + Central Difference)
        StaticVector<ADVar, N> x_plus_dual, x_minus_dual;
        
        for (size_t i = 0; i < N; ++i) {
            // x + h*e_i
            for (size_t k = 0; k < N; ++k) x_plus_dual(k) = ADVar::make_variable(x(k) + (k == i ? h : 0.0), k);
            ADVar f_plus = f(x_plus_dual);

            // x - h*e_i
            for (size_t k = 0; k < N; ++k) x_minus_dual(k) = ADVar::make_variable(x(k) - (k == i ? h : 0.0), k);
            ADVar f_minus = f(x_minus_dual);

            for (size_t j = 0; j < N; ++j) {
                H(j, i) = (f_plus.g[j] - f_minus.g[j]) * inv_2h;
            }
        }

        // 수치적 오차 방지를 위한 헤시안 대칭화 (Symmetrization)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < i; ++j) {
                double sym = 0.5 * (H(i, j) + H(j, i));
                H(i, j) = sym;
                H(j, i) = sym;
            }
        }
    }

   public:
    NewtonMethod() = delete;

    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResult<N> optimize(const Func& f, 
                                                        const StaticVector<double, N>& x_init,
                                                        double tol = 1e-6,
                                                        size_t max_iter = 50) noexcept {
        auto start_clock = std::chrono::high_resolution_clock::now();

        StaticVector<double, N> x = x_init;
        const double tol_sq = tol * tol;
        size_t iter = 0;
        
        // 메모리 호이스팅 (Zero-Allocation)
        StaticVector<double, N> g, neg_g, p;
        StaticMatrix<double, N, N> H;
        double f_val = 0.0;
        SolverStatus status = SolverStatus::MAX_ITERATION_REACHED;

        for (iter = 1; iter <= max_iter; ++iter) {
            // 1. 값, Gradient, Hessian 동시 도출
            compute_grad_and_hessian(f, x, f_val, g, H);

            // 2. Gradient Norm 검사 (수렴 판정)
            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g(i) * g(i);
                neg_g(i) = -g(i); // LDLT_solve를 위한 부호 반전
            }

            if (g_norm_sq < tol_sq) {
                status = SolverStatus::SUCCESS;
                break;
            }

            // 3. 선형 시스템 풀이 (H * p = -g)
            // 헤시안(H)은 대칭행렬이므로 LDLT 분해가 가장 빠르고 적합합니다.
            if (linalg::LDLT_decompose(H) != MathStatus::SUCCESS) {
                // 헤시안이 특이(Singular)하거나 음의 곡률(Negative Curvature)을 만나면 즉시 탈출
                status = SolverStatus::MATH_ERROR;
                break;
            }

            linalg::LDLT_solve(H, neg_g, p);

            // 4. Step Size (Stagnation) 검증
            double max_p = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double abs_p = MathTraits<double>::abs(p(i));
                if (abs_p > max_p) max_p = abs_p;
            }
            if (max_p < 1e-12) {
                status = SolverStatus::STEP_SIZE_TOO_SMALL;
                break;
            }

            // 5. 위치 업데이트 (x = x + p) SIMD 타격
            x += p;
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        return {x, f_val, iter, duration.count(), status};
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_NEWTON_METHOD_HPP_