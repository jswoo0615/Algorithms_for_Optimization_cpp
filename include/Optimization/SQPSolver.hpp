#ifndef OPTIMIZATION_SQP_SOLVER_HPP_
#define OPTIMIZATION_SQP_SOLVER_HPP_

#include "Optimization/AutoDiff.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"
// [Architect's Upgrade] Active-Set 폐기, IPM 솔버 장착
#include <algorithm>
#include <cmath>

#include "Optimization/IPMQPSolver.hpp"

namespace Optimization {

template <size_t N_vars, size_t N_eq, size_t N_ineq>
class SQPSolver {
   public:
    // 하부 구조: 정적 메모리 기반 Primal-Dual IPM 엔진
    IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver;
    StaticMatrix<double, N_vars, N_vars> H;

    SQPSolver() {
        H.set_zero();
        for (size_t i = 0; i < N_vars; ++i) {
            H(static_cast<int>(i), static_cast<int>(i)) = 1.0;
        }
    }

    template <typename CostFunc, typename EqFunc, typename IneqFunc>
    bool solve(StaticVector<double, N_vars>& u, CostFunc cost_f, EqFunc eq_f, IneqFunc ineq_f,
               int max_iter = 50) {
        for (int iter = 0; iter < max_iter; ++iter) {
            double cost_val = 0.0;
            StaticVector<double, N_vars> grad_f;
            AutoDiff::value_and_gradient<N_vars>(cost_f, u, cost_val, grad_f);

            if constexpr (N_eq > 0) {
                StaticVector<double, N_eq> eq_val = eq_f(u);
                StaticMatrix<double, N_eq, N_vars> J_eq = AutoDiff::jacobian<N_eq, N_vars>(eq_f, u);
                for (size_t i = 0; i < N_eq; ++i) {
                    qp_solver.b_eq(static_cast<int>(i)) = -eq_val(static_cast<int>(i));
                    for (size_t j = 0; j < N_vars; ++j) {
                        qp_solver.A_eq(static_cast<int>(i), static_cast<int>(j)) =
                            J_eq(static_cast<int>(i), static_cast<int>(j));
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
                    }
                }
            }

            qp_solver.P = H;
            qp_solver.q = grad_f;

            StaticVector<double, N_vars> p;
            p.set_zero();

            // IPM 솔버 가동 (톨러런스를 1e-4로 설정하여 실시간성 확보)
            if (!qp_solver.solve(p, 50, 1e-4)) {
                p = grad_f * -0.05;
            }

            double p_norm = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                if (std::isnan(p(static_cast<int>(i))) || std::isinf(p(static_cast<int>(i)))) {
                    p(static_cast<int>(i)) = 0.0;
                }
                // 한 제어 주기당 변화할 수 있는 물리적 최대 한계치 부여 (가속도 +-3.0, 조향 +-3.0
                // 등)
                if (p(static_cast<int>(i)) > 3.0) p(static_cast<int>(i)) = 3.0;
                if (p(static_cast<int>(i)) < -3.0) p(static_cast<int>(i)) = -3.0;

                p_norm = std::max(p_norm, std::abs(p(static_cast<int>(i))));
            }

            if (p_norm < 1e-6) return true;

            double alpha = 1.0;
            const double rho = 10.0;

            double current_merit = cost_val;
            if constexpr (N_eq > 0) {
                StaticVector<double, N_eq> v = eq_f(u);
                for (size_t i = 0; i < N_eq; ++i)
                    current_merit += rho * std::abs(v(static_cast<int>(i)));
            }
            if constexpr (N_ineq > 0) {
                StaticVector<double, N_ineq> v = ineq_f(u);
                for (size_t i = 0; i < N_ineq; ++i)
                    current_merit += rho * std::max(0.0, v(static_cast<int>(i)));
            }

            StaticVector<double, N_vars> u_next;
            for (int ls_iter = 0; ls_iter < 10; ++ls_iter) {
                for (size_t i = 0; i < N_vars; ++i)
                    u_next(static_cast<int>(i)) =
                        u(static_cast<int>(i)) + alpha * p(static_cast<int>(i));

                double next_cost = cost_f(u_next);
                double next_merit = next_cost;

                if constexpr (N_eq > 0) {
                    StaticVector<double, N_eq> v = eq_f(u_next);
                    for (size_t i = 0; i < N_eq; ++i)
                        next_merit += rho * std::abs(v(static_cast<int>(i)));
                }
                if constexpr (N_ineq > 0) {
                    StaticVector<double, N_ineq> v = ineq_f(u_next);
                    for (size_t i = 0; i < N_ineq; ++i)
                        next_merit += rho * std::max(0.0, v(static_cast<int>(i)));
                }

                if (next_merit < current_merit) break;
                alpha *= 0.5;
            }

            // Damped BFGS Update
            StaticVector<double, N_vars> next_grad = AutoDiff::gradient<N_vars>(cost_f, u_next);
            StaticVector<double, N_vars> s;
            StaticVector<double, N_vars> y;
            double s_norm_sq = 0.0;

            for (size_t i = 0; i < N_vars; ++i) {
                int idx = static_cast<int>(i);
                s(idx) = u_next(idx) - u(idx);
                y(idx) = next_grad(idx) - grad_f(idx);
                s_norm_sq += s(idx) * s(idx);
            }

            if (s_norm_sq > 1e-12) {
                StaticVector<double, N_vars> Hs;
                Hs.set_zero();
                double sHs = 0.0;
                double ys = 0.0;

                for (size_t i = 0; i < N_vars; ++i) {
                    int r = static_cast<int>(i);
                    for (size_t j = 0; j < N_vars; ++j) {
                        Hs(r) += H(r, static_cast<int>(j)) * s(static_cast<int>(j));
                    }
                    sHs += s(r) * Hs(r);
                    ys += y(r) * s(r);
                }

                double theta = 1.0;
                if (ys < 0.2 * sHs) {
                    theta = (0.8 * sHs) / (sHs - ys + 1e-16);
                }

                StaticVector<double, N_vars> r_vec;
                double rs = 0.0;
                for (size_t i = 0; i < N_vars; ++i) {
                    int idx = static_cast<int>(i);
                    r_vec(idx) = theta * y(idx) + (1.0 - theta) * Hs(idx);
                    rs += r_vec(idx) * s(idx);
                }

                if (rs > 1e-12 && sHs > 1e-12) {
                    for (size_t i = 0; i < N_vars; ++i) {
                        for (size_t j = 0; j < N_vars; ++j) {
                            int row = static_cast<int>(i);
                            int col = static_cast<int>(j);
                            H(row, col) = H(row, col) - (Hs(row) * Hs(col)) / sHs +
                                          (r_vec(row) * r_vec(col)) / rs;
                        }
                    }
                }
            }

            u = u_next;
        }
        return false;
    }
};

}  // namespace Optimization

#endif