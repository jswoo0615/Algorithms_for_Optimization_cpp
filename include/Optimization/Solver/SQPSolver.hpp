#ifndef OPTIMIZATION_SQP_SOLVER_HPP_
#define OPTIMIZATION_SQP_SOLVER_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Solver/QPSolver_IPM.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
namespace solver {

/**
 * @brief 고속 순차적 이차 계획법 (SQP) 솔버
 * @details 비선형 최적화 문제를 매 스텝 선형 QP 서브프라블럼으로 변환하여 풉니다.
 * 
 * @tparam Nx 최적화 변수의 개수
 * @tparam Nc 부등식 제약조건의 개수
 * @tparam CostFunctor 비용 함수 객체 타입
 * @tparam IneqFunctor 부등식 제약 함수 객체 타입 (h(x) <= 0 형태)
 */
template <size_t Nx, size_t Nc, typename CostFunctor, typename IneqFunctor>
class SQPSolver {
   private:
    /**
     * @brief 하이브리드 헤시안/그래디언트 계산 엔진
     */
    static void compute_cost_derivatives(const CostFunctor& f, const StaticVector<double, Nx>& x,
                                         StaticVector<double, Nx>& g, StaticMatrix<double, Nx, Nx>& H) {
        using ADVar = DualVec<double, Nx>;
        constexpr double h = 1e-5;
        constexpr double inv_2h = 1.0 / (2.0 * h);

        StaticVector<ADVar, Nx> x_dual;
        for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(x(i), i);
        
        ADVar f_dual = f(x_dual);
        for (size_t i = 0; i < Nx; ++i) g(i) = f_dual.g[i];

        // Central Difference for Exact Hessian
        StaticVector<ADVar, Nx> x_plus, x_minus;
        for (size_t i = 0; i < Nx; ++i) {
            for (size_t k = 0; k < Nx; ++k) x_plus(k) = ADVar::make_variable(x(k) + (k == i ? h : 0.0), k);
            ADVar f_plus = f(x_plus);
            
            for (size_t k = 0; k < Nx; ++k) x_minus(k) = ADVar::make_variable(x(k) - (k == i ? h : 0.0), k);
            ADVar f_minus = f(x_minus);

            for (size_t j = 0; j < Nx; ++j) {
                H(j, i) = (f_plus.g[j] - f_minus.g[j]) * inv_2h;
            }
        }

        // [Architect's Update] 대칭화 및 SPD 강제 (Regularization)
        // QP 솔버인 IPM이 무조건 풀 수 있도록 대각 성분에 미세한 양수를 더해줍니다.
        for (size_t i = 0; i < Nx; ++i) {
            for (size_t j = 0; j < i; ++j) {
                double sym = 0.5 * (H(i, j) + H(j, i));
                H(i, j) = sym;
                H(j, i) = sym;
            }
            H(i, i) += 1e-4; // Regularization Shield
        }
    }

   public:
    static SolverStatus solve(const CostFunctor& cost_func,
                              const IneqFunctor& ineq_func,
                              StaticVector<double, Nx>& x_opt,
                              int sqp_max_iter = 15, double tol = 1e-5) {
        
        using ADVar = DualVec<double, Nx>;
        StaticVector<ADVar, Nx> x_dual;
        StaticMatrix<double, Nx, Nx> H;
        StaticVector<double, Nx> g;
        StaticMatrix<double, Nc, Nx> C;
        StaticVector<double, Nc> d;
        StaticVector<double, Nx> dx;

        for (int sqp_iter = 0; sqp_iter < sqp_max_iter; ++sqp_iter) {
            // 1. 비선형 목적 함수 -> QP용 H, g 도출
            compute_cost_derivatives(cost_func, x_opt, g, H);

            // 2. 비선형 제약조건 -> QP용 선형 제약 도출 (C * dx <= d)
            // h(x + dx) \approx h(x) + J_h * dx <= 0  ==>  J_h * dx <= -h(x)
            for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(x_opt(i), i);
            StaticVector<ADVar, Nc> h_dual = ineq_func(x_dual);

            for (size_t i = 0; i < Nc; ++i) {
                d(i) = -Optimization::get_value(h_dual(i)); // d = -h(x)
                for (size_t j = 0; j < Nx; ++j) {
                    C(i, j) = h_dual(i).g[j];               // C = J_h
                }
            }

            // 3. QP 서브프라블럼 하청 (IPM 호출)
            dx.set_zero();
            SolverStatus qp_status = QPSolver_IPM<Nx, Nc>::solve(H, g, C, d, dx, 40, 1e-5);

            // 수학적 붕괴 시 즉시 제어권 반환
            if (qp_status == SolverStatus::MATH_ERROR || qp_status == SolverStatus::INFEASIBLE) {
                return qp_status;
            }

            // 4. 프라이멀 변수 업데이트 (x_opt += dx)
            double max_dx = 0.0;
            for (size_t i = 0; i < Nx; ++i) {
                x_opt(i) += dx(i);
                double abs_dx = MathTraits<double>::abs(dx(i));
                if (abs_dx > max_dx) max_dx = abs_dx;
            }

            // 5. SQP 수렴 판정 (더 이상 이동하지 않으면 최적해 도달)
            if (max_dx < tol) {
                return SolverStatus::SUCCESS;
            }
        }

        return SolverStatus::MAX_ITERATION_REACHED;
    }
};

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_SQP_SOLVER_HPP_