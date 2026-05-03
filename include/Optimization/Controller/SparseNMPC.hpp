#ifndef OPTIMIZATION_SPARSE_NMPC_HPP_
#define OPTIMIZATION_SPARSE_NMPC_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/Dual.hpp"
#include <array>
#include <cmath>

namespace Optimization {

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double r = 0.5;
};

namespace controller {

    template <size_t H, size_t Nx = 6, size_t Nu = 2>
    class SparseNMPC {
    public:
        StaticVector<double, Nx> Q, Qf;
        StaticVector<double, Nu> R, R_rate;
        double dt;

        std::array<StaticVector<double, Nu>, H> U_guess;
        std::array<StaticVector<double, Nx>, H + 1> X_pred;
        
        StaticVector<double, Nu> u_last;
        std::array<Obstacle, 10> obstacles;

        solver::RiccatiSolver<H, Nx, Nu> riccati;

        SparseNMPC() {
            Q.set_zero(); Q(0) = 15.0; Q(1) = 15.0; Q(2) = 5.0; Q(3) = 10.0;
            Qf.set_zero(); for (size_t i=0; i<Nx; ++i) Qf(i) = Q(i) * 10.0;
            R.set_zero(); R(0) = 0.5; R(1) = 10.0;
            R_rate.set_zero(); R_rate(0) = 50.0; R_rate(1) = 250.0; // Slew rate penalty
            u_last.set_zero();
            dt = 0.1;
            for (size_t k=0; k<H; ++k) U_guess[k].set_zero();
        }

        // =========================================================================
        // [Architect's Tactics] 각 노드(Node)별 비선형 잔차 함수 (M_res = 26)
        // 1. 상태 추종 (6) + 2. 제어 크기 (2) + 3. Slew Rate (2)
        // 4. 장애물 (10) + 5. 가속도/조향 한계 Soft Bounds (4+2)
        // =========================================================================
        template <typename T>
        StaticVector<T, 26> eval_node_residuals(const StaticVector<T, Nx>& x, 
                                                const StaticVector<T, Nu>& u, 
                                                const StaticVector<T, Nu>& u_prev,
                                                const StaticVector<double, Nx>& x_ref) 
        {
            StaticVector<T, 26> res; res.set_zero();
            int idx = 0;

            // 1. 상태 오차 잔차 (sqrt(Q) * error)
            for (size_t i = 0; i < Nx; ++i) {
                res(idx++) = T(std::sqrt(Q(i))) * (x(i) - T(x_ref(i)));
            }

            // 2. 제어 입력 크기 잔차
            for (size_t i = 0; i < Nu; ++i) {
                res(idx++) = T(std::sqrt(R(i))) * u(i);
            }

            // 3. Slew Rate (변화율) 잔차
            for (size_t i = 0; i < Nu; ++i) {
                res(idx++) = T(std::sqrt(R_rate(i))) * (u(i) - u_prev(i));
            }

            // 4. 장애물 회피망 (Soft Constraint)
            for (size_t i = 0; i < 10; ++i) {
                T dx = x(0) - T(obstacles[i].x);
                T dy = x(1) - T(obstacles[i].y);
                T dist_sq = dx * dx + dy * dy;
                T r_safe = T(obstacles[i].r + 0.8);
                T viol = r_safe * r_safe - dist_sq;
                
                // 거리가 안전반경 안으로 파고들었을 때만 엄청난 페널티 부여
                res(idx++) = (Optimization::get_value(viol) > 0.0) ? T(std::sqrt(8000.0)) * viol : T(0.0);
            }

            // 5. Hard Bounds의 Soft Penalty 근사 (IPM 적용 전 단계)
            // 가속도 한계 [-3.0, 3.0]
            T a_viol_upper = u(0) - T(3.0);
            T a_viol_lower = T(-3.0) - u(0);
            res(idx++) = (Optimization::get_value(a_viol_upper) > 0.0) ? T(100.0) * a_viol_upper : T(0.0);
            res(idx++) = (Optimization::get_value(a_viol_lower) > 0.0) ? T(100.0) * a_viol_lower : T(0.0);

            // 조향 한계 [-0.5, 0.5] rad
            T steer_viol_upper = u(1) - T(0.5);
            T steer_viol_lower = T(-0.5) - u(1);
            res(idx++) = (Optimization::get_value(steer_viol_upper) > 0.0) ? T(100.0) * steer_viol_upper : T(0.0);
            res(idx++) = (Optimization::get_value(steer_viol_lower) > 0.0) ? T(100.0) * steer_viol_lower : T(0.0);

            return res;
        }

        bool solve(const StaticVector<double, Nx>& x_curr, const StaticVector<double, Nx>& x_ref) {
            vehicle::DynamicBicycleModel model;

            // 1. Forward Rollout (Nominal Trajectory)
            X_pred[0] = x_curr;
            for (size_t k = 0; k < H; ++k) {
                X_pred[k + 1] = integrator::step_rk4<Nx, Nu>(model, X_pred[k], U_guess[k], dt);
            }

            // 2. Linearization & Gauss-Newton Cost Formulation
            for (size_t k = 0; k < H; ++k) {
                // 이전 제어 입력 (k=0 이면 u_last, 아니면 이전 스텝의 guess)
                StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k - 1];

                // --- A. 동역학 선형화 (A_k, B_k) ---
                using ADVar = DualVec<double, Nx + Nu>;
                StaticVector<ADVar, Nx> x_dual;
                StaticVector<ADVar, Nu> u_dual;
                StaticVector<ADVar, Nu> u_prev_dual; // 잔차 평가용 상수로 취급

                for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
                for (size_t i = 0; i < Nu; ++i) {
                    u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);
                    u_prev_dual(i) = ADVar(u_prev(i)); // Gradient 전파 안함
                }

                StaticVector<ADVar, Nx> x_next_dual = integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);
                for (size_t i = 0; i < Nx; ++i) {
                    for (size_t j = 0; j < Nx; ++j) riccati.A[k](i, j) = x_next_dual(i).g[j];
                    for (size_t j = 0; j < Nu; ++j) riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                }
                for (size_t i = 0; i < Nx; ++i) riccati.d[k](i) = 0.0; // Nominal trajectory 일치

                // --- B. 비용 함수 Gauss-Newton 선형화 (Q_k, R_k, q_k, r_k) ---
                StaticVector<ADVar, 26> res_dual = eval_node_residuals(x_dual, u_dual, u_prev_dual, x_ref);
                
                riccati.Q[k].set_zero(); riccati.R[k].set_zero();
                riccati.q[k].set_zero(); riccati.r[k].set_zero();

                // J^T * J 및 J^T * res 조립
                for (size_t res_idx = 0; res_idx < 26; ++res_idx) {
                    double r_val = res_dual(res_idx).v;
                    
                    // State 방향 (Q, q)
                    for (size_t i = 0; i < Nx; ++i) {
                        double J_xi = res_dual(res_idx).g[i];
                        riccati.q[k](i) += J_xi * r_val;
                        for (size_t j = 0; j < Nx; ++j) {
                            riccati.Q[k](i, j) += J_xi * res_dual(res_idx).g[j];
                        }
                    }
                    // Control 방향 (R, r)
                    for (size_t i = 0; i < Nu; ++i) {
                        double J_ui = res_dual(res_idx).g[Nx + i];
                        riccati.r[k](i) += J_ui * r_val;
                        for (size_t j = 0; j < Nu; ++j) {
                            riccati.R[k](i, j) += J_ui * res_dual(res_idx).g[Nx + j];
                        }
                    }
                }

                double lambda_q = 2.0;
                double lambda_r = 150.0;
                for (size_t i = 0; i < Nx; ++i) {
                    riccati.Q[k](i, i) += lambda_q;
                }
                for (size_t i = 0; i < Nu; ++i) {
                    riccati.R[k](i, i) += lambda_r;
                }
            }

            // Terminal Node (H)
            riccati.Q[H].set_zero(); riccati.q[H].set_zero();
            for (size_t i=0; i<Nx; ++i) {
                double err = X_pred[H](i) - x_ref(i);
                riccati.Q[H](i, i) = Qf(i);
                riccati.q[H](i) = Qf(i) * err;
            }

            // 3. Solve & Update
            SolverStatus status = riccati.solve();
            if (status != SolverStatus::SUCCESS) return false;

            for (size_t k = 0; k < H; ++k) {
                for (size_t i = 0; i < Nu; ++i) {
                    double step = riccati.du[k](i);

                    // 1 Iteration 당 최대 변화량 제한 (가속도 : 0.5m/s^2, 조향 : 0.05 rad (약 2.8도))
                    double max_step = (i == 0) ? 0.5 : 0.05;

                    if (step > max_step) {
                        step = max_step;
                    }
                    if (step < -max_step) {
                        step = -max_step;
                    }
                    U_guess[k](i) += step;
                }
            }
            
            // 다음 제어 주기를 위한 Warm Start 및 Slew rate 기준 갱신
            u_last = U_guess[0]; 

            return true;
        }
    };

} // namespace controller
} // namespace Optimization

#endif // OPTIMIZATION_SPARSE_NMPC_HPP_