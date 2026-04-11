#ifndef OPTIMIZATION_RTI_SOLVER_HPP_
#define OPTIMIZATION_RTI_SOLVER_HPP_

#include "Optimization/AutoDiff.hpp"
#include "Optimization/IPMQPSolver.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class RTISolver {
   public:
    IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver;

    RTISolver() {}

    // [RTI Core] No while loop, No line search, No BFGS update
    // O(1) Deterministic Execution Time
    template <typename CostFunc, typename EqFunc, typename IneqFunc>
    bool solve(StaticVector<double, N_vars>& u, CostFunc cost_f, EqFunc eq_f, IneqFunc ineq_f) {
        // 1. Preparation Phase : Linearization & Gauss-Newton Hessian
        double cost_val = 0.0;
        StaticVector<double, N_vars> grad_f;
        AutoDiff::value_and_gradient<N_vars>(cost_f, u, cost_val, grad_f);

        StaticMatrix<double, N_vars, N_vars> H_GN;
        H_GN.set_zero();

        for (size_t i = 0; i < N_vars; ++i) {
            H_GN(static_cast<int>(i), static_cast<int>(i)) = 1.0;
        }

        if constexpr (N_eq > 0) {
            StaticVector<double, N_eq> eq_val = eq_f(u);
            StaticMatrix<double, N_eq, N_vars> J_eq = AutoDiff::jacobian<N_eq, N_vars>(eq_f, u);
            for (size_t i = 0; i < N_vars; ++i) {
                qp_solver.b_eq(static_cast<int>(i)) = -eq_val(static_cast<int>(i));
                for (size_t j = 0; j < N_vars; ++j) {
                    qp_solver.A_eq(static_cast<int>(i), static_cast<int>(j)) =
                        J_eq(static_cast<int>(i), static_cast<int>(j));
                    H_GN(static_cast<int>(j), static_cast<int>(j)) +=
                        std::abs(J_eq(static_cast<int>(i), static_cast<int>(j))) * 0.1;
                }
            }
        }

        if constexpr (N_ineq > 0) {
            StaticVector<double, N_ineq> ineq_val = ineq_f(u);
            StaticMatrix<double, N_ineq, N_vars> J_ineq =
                AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);
            for (size_t i = 0; i < N_ineq; ++i) {
                qp_solver.b_ineq(static_cast<int>(i)) = -ineq_val(static_cast<int>(i));
                for (size_t j = 0; j < N_vars; ++j) {
                    qp_solver.A_ineq(static_cast<int>(i), static_cast<int>(j)) =
                        J_ineq(static_cast<int>(i), static_cast<int>(j));
                    H_GN(static_cast<int>(j), static_cast<int>(j)) +=
                        std::abs(J_ineq(static_cast<int>(i), static_cast<int>(j))) * 0.1;
                }
            }
        }

        qp_solver.P = H_GN;
        qp_solver.q = grad_f;

        StaticVector<double, N_vars> p;
        p.set_zero();

        // 2. Feedback Phase : Fixed 15 Iterations in IPM
        bool qp_success = qp_solver.solve(p, 15, 1e-3);

        // 3. Update Phase : Full Step with Trust-Region Clamping
        if (qp_success) {
            for (size_t i = 0; i < N_vars; ++i) {
                if (std::isnan(p(static_cast<int>(i) || std::isinf(p(static_cast<int>(i)))))) {
                    p(static_cast<int>(i)) = 0.0;
                }
                // Clamping limits
                if (p(static_cast<int>(i)) > 3.0) {
                    p(static_cast<int>(i)) = 3.0;
                }
                if (p(static_cast<int>(i)) < -3.0) {
                    p(static_cast<int>(i)) = -3.0;
                }

                u(static_cast<int>(i)) += p(static_cast<int>(i));
            }
            return true;
        }
        return false;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_RTI_SOLVER_HPP_