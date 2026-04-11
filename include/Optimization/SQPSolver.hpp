#ifndef OPTIMIZATION_SQP_SOLVER_HPP_
#define OPTIMIZATION_SQP_SOLVER_HPP_

#include <algorithm>
#include <cmath>

#include "Optimization/ActiveSetSolver.hpp"
#include "Optimization/AutoDiff.hpp"
#include "Optimization/LineSearch.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {
/**
 * @brief Sequential Quadratic Programming (SQP) Solver
 * @details 비선형 최적화 솔버
 */
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class SQPSolver {
   public:
    ActiveSetSolver<N_vars, N_eq, N_ineq> qp_solver;
    StaticMatrix<double, N_vars, N_vars> H;

    SQPSolver() {
        H.set_zero();
        for (size_t i = 0; i < N_vars; ++i) {
            H(static_cast<int>(i), static_cast<int>(i)) = 1.0;
        }
    }

    /**
     * @brief 비선형 최적화 메인 루프
     */
    template <typename CostFunc, typename EqFunc, typename IneqFunc>
    bool solve(StaticVector<double, N_vars>& u, CostFunc cost_f, EqFunc eq_f, IneqFunc ineq_f,
               int max_iter = 50) {
        for (int iter = 0; iter < max_iter; ++iter) {
            // 1. 현재 지점의 함수값 및 Gradient 추출
            double cost_val = 0.0;
            StaticVector<double, N_vars> grad_f;
            AutoDiff::value_and_gradient<N_vars>(cost_f, u, cost_val, grad_f);

            // 2. 등식 제약 조건 선형화 (StaticMatrix 규격 적용)
            if constexpr (N_eq > 0) {
                StaticVector<double, N_eq> eq_val = eq_f(u);
                StaticMatrix<double, N_eq, N_vars> J_eq = AutoDiff::jacobian<N_eq, N_vars>(eq_f, u);
                qp_solver.A_eq = J_eq;
                qp_solver.b_eq = eq_val * -1.0;
            }

            // 3. 부등식 제약 조건 선형화 및 Active Set 초기화
            if constexpr (N_ineq > 0) {
                StaticVector<double, N_ineq> ineq_val = ineq_f(u);
                StaticMatrix<double, N_ineq, N_vars> J_ineq =
                    AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);
                qp_solver.A_ineq = J_ineq;
                qp_solver.b_ineq = ineq_val * -1.0;

                // Infeasible Start 방지를 위한 제약 위반 감시
                for (size_t i = 0; i < N_ineq; ++i) {
                    qp_solver.working_set[i] = (qp_solver.b_ineq(static_cast<int>(i)) < -1e-6);
                }
            }

            // 4. QP 하위 문제 해결
            qp_solver.P = H;
            qp_solver.q = grad_f;
            StaticVector<double, N_vars> p;
            p.set_zero();

            if (!qp_solver.solve(p, 100)) {
                // QP 실패 시 Fallback : Steepest Descent 방향으로 비상 탈출
                p = grad_f * -0.05;
            }

            // 5. 수렴 판정
            double p_norm = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                p_norm = std::max(p_norm, std::abs(p(static_cast<int>(i))));
            }
            if (p_norm < 1e-6) {
                return true;
            }

            // 6. Merit Function 기반 Backtracking Line Search
            // LineSearch::ray_point 개념을 활용한 스텝 제어
            double alpha = 1.0;
            const double rho = 10.0;  // 제약 조건 패널티 가중치

            // 현재 Merit 계산
            double current_merit = cost_val;
            if constexpr (N_eq > 0) {
                StaticVector<double, N_eq> v = eq_f(u);
                for (size_t i = 0; i < N_eq; ++i) {
                    current_merit += rho * std::abs(v(static_cast<int>(i)));
                    ;
                }
            }
            if constexpr (N_ineq > 0) {
                StaticVector<double, N_ineq> v = ineq_f(u);
                for (size_t i = 0; i < N_ineq; ++i) {
                    current_merit += rho * std::max(0.0, v(static_cast<int>(i)) +
                                                             alpha * p(static_cast<int>(i)));
                }
            }
            StaticVector<double, N_vars> u_next;
            for (int ls_iter = 0; ls_iter < 10; ++ls_iter) {
                // u_next = u + alpha * p (LineSearch의 ray_point 로직과 동일)
                for (size_t i = 0; i < N_vars; ++i) {
                    u_next(static_cast<int>(i)) =
                        u(static_cast<int>(i)) + alpha * p(static_cast<int>(i));
                }
                double next_cost = cost_f(u_next);
                double next_merit = next_cost;

                if constexpr (N_eq > 0) {
                    StaticVector<double, N_eq> v = eq_f(u_next);
                    for (size_t i = 0; i < N_eq; ++i) {
                        next_merit += rho * std::abs(v(static_cast<int>(i)));
                    }
                }
                if constexpr (N_ineq > 0) {
                    StaticVector<double, N_ineq> v = ineq_f(u_next);
                    for (size_t i = 0; i < N_ineq; ++i) {
                        next_merit += rho * std::max(0.0, v(static_cast<int>(i)));
                    }
                }
                if (next_merit < current_merit) {
                    break;
                }
                alpha *= 0.5;  // 보폭 축소
            }

            // 7. Damped BFGS Hessian Update (Powell's Trick)
            StaticVector<double, N_vars> next_grad = AutoDiff::gradient<N_vars>(cost_f, u_next);
            StaticVector<double, N_vars> s;
            StaticVector<double, N_vars> y;
            double s_norm_sq = 0.0;

            for (size_t i = 0; i < N_vars; ++i) {
                int idx = static_cast<int>(i);
                s(idx) = u_next(idx) - u(idx);
                // SQP는 라그랑지안 미분을 써야 하나, 안정성 확보를 위해 우선 목적함수 미분으로 근사
                y(idx) = next_grad(idx) - grad_f(idx);
                s_norm_sq += s(idx) * s(idx);
            }

            if (s_norm_sq > 1e-12) {
                StaticVector<double, N_vars> Hs;
                Hs.set_zero();
                double sHs = 0.0;
                double ys = 0.0;

                // H * s 연산
                for (size_t i = 0; i < N_vars; ++i) {
                    int r = static_cast<int>(i);
                    for (size_t j = 0; j < N_vars; ++j) {
                        Hs(r) += H(r, static_cast<int>(j)) * s(static_cast<int>(j));
                    }
                    sHs += s(r) * Hs(r);
                    ys += y(r) * s(r);
                }

                // Powell's Damping : 제약 조건으로 인해 H가 양의 정부호 (Positivie Definite)를 잃는
                // 것을 방지
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

                // H_new = H - (Hs * HS^T)/sHs + (r * r^T)/rs
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

#endif  // OPTIMIZATION_SQP_SOLVER_HPP_