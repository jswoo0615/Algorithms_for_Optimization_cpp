#ifndef OPTIMIZATION_RTI_SOLVER_HPP_
#define OPTIMIZATION_RTI_SOLVER_HPP_

#include <cmath>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/IPMQPSolver.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

template <size_t N_vars, size_t N_eq, size_t N_ineq, size_t N_res>
class RTISolver {
   public:
    IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver;

    template <typename ResidualFunc, typename EqFunc, typename IneqFunc>
    bool solve(StaticVector<double, N_vars>& u, ResidualFunc res_f, EqFunc eq_f, IneqFunc ineq_f) {
        StaticVector<double, N_res> r_val = res_f(u);
        StaticMatrix<double, N_res, N_vars> J_res = AutoDiff::jacobian<N_res, N_vars>(res_f, u);

        StaticMatrix<double, N_vars, N_vars> H_GN;
        StaticVector<double, N_vars> grad_f;
        H_GN.set_zero();
        grad_f.set_zero();

        for (size_t i = 0; i < N_vars; ++i) {
            for (size_t j = 0; j < N_vars; ++j) {
                double sum_h = 0.0;
                for (size_t k = 0; k < N_res; ++k) sum_h += J_res(k, i) * J_res(k, j);
                H_GN(i, j) = sum_h;
            }
            double sum_g = 0.0;
            for (size_t k = 0; k < N_res; ++k) sum_g += J_res(k, i) * r_val(k);
            grad_f(i) = sum_g;

            // [Architect's Tuning] 1.0 -> 5.0 (안정성 확보를 위한 댐핑 강화)
            H_GN(i, i) += 5.0;
        }

        // 제약 조건 선형화
        StaticVector<double, N_ineq> ineq_val = ineq_f(u);
        StaticMatrix<double, N_ineq, N_vars> J_ineq = AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);
        for (size_t i = 0; i < N_ineq; ++i) {
            qp_solver.b_ineq(i) = -ineq_val(i);
            for (size_t j = 0; j < N_vars; ++j) qp_solver.A_ineq(i, j) = J_ineq(i, j);
        }

        qp_solver.P = H_GN;
        qp_solver.q = grad_f;
        StaticVector<double, N_vars> p;
        p.set_zero();

        bool qp_success = qp_solver.solve(p, 15, 1e-3);

        if (qp_success) {
            for (size_t i = 0; i < N_vars; ++i) {
                if (std::isnan(p(i)) || std::isinf(p(i))) p(i) = 0.0;
                u(i) += p(i);

                // 최종 물리 한계 클램핑
                if (i % 2 == 0) {
                    if (u(i) > 3.0) u(i) = 3.0;
                    if (u(i) < -3.0) u(i) = -3.0;
                } else {
                    if (u(i) > 0.5) u(i) = 0.5;
                    if (u(i) < -0.5) u(i) = -0.5;
                }
            }
            return true;
        }
        return false;
    }
};

}  // namespace Optimization
#endif
