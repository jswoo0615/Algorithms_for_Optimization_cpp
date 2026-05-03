#ifndef OPTIMIZATION_SPARSE_NMPC_HPP_
#define OPTIMIZATION_SPARSE_NMPC_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/Dual.hpp"
#include "Optimization/Utils/RigidTransform.hpp"
#include <array>
#include <cmath>
#include <string>
#include <iostream>
#include <algorithm> 

namespace Optimization {

struct Obstacle {
    double x = 0.0; double y = 0.0; double r = 0.5;
};

namespace controller {

    struct NMPCResult {
        bool success = false; bool fallback_triggered = false;
        double max_kkt_error = 0.0; int sqp_iterations = 0;
        std::string status_msg = "OK";
    };

    inline double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    template <size_t H, size_t Nx = 6, size_t Nu = 2>
    class SparseNMPC {
    public:
        StaticVector<double, Nx> Q, Qf;
        StaticVector<double, Nu> R, R_rate;
        double dt;
        std::array<StaticVector<double, Nu>, H> U_guess;
        std::array<StaticVector<double, Nx>, H + 1> X_pred;
        StaticVector<double, Nu> u_last;
        std::array<Obstacle, 10> global_obstacles;
        std::array<Obstacle, 10> local_obstacles;
        solver::RiccatiSolver<H, Nx, Nu> riccati;

        SparseNMPC() {
            // Q(2) 비중을 줄여 솔버가 각도에 너무 민감하게 반응하지 않도록 조절
            Q.set_zero(); Q(1) = 15.0; Q(2) = 100.0; Q(3) = 10.0; 
            Qf.set_zero(); for (size_t i=0; i<Nx; ++i) Qf(i) = Q(i) * 5.0;
            R.set_zero(); R(0) = 1.0; R(1) = 20.0; 
            R_rate.set_zero(); R_rate(0) = 50.0; R_rate(1) = 300.0;
            u_last.set_zero(); dt = 0.1;
            for (size_t k=0; k<H; ++k) U_guess[k].set_zero();
            for (size_t i = 0; i < 10; ++i) {
                global_obstacles[i].x = 10000.0; global_obstacles[i].y = 10000.0; global_obstacles[i].r = 0.1;
            }
        }

        template <typename T>
        StaticVector<T, 25> eval_node_residuals(const StaticVector<T, Nx>& x, 
                                                const StaticVector<T, Nu>& u, 
                                                const StaticVector<T, Nu>& u_prev,
                                                const StaticVector<double, Nx>& x_ref_local,
                                                const StaticVector<double, Nx>& x_curr_global,
                                                const StaticVector<double, Nx>& x_ref_global) 
        {
            using std::abs; using std::sqrt;

            StaticVector<T, 25> res; res.set_zero();
            int idx = 0;

            T theta_c = T(x_curr_global(2));
            T sin_t = T(std::sin(x_curr_global(2)));
            T cos_t = T(std::cos(x_curr_global(2)));
            T y_c = T(x_curr_global(1));
            T y_target = T(x_ref_global(1));
            
            // True global Y error projected approximately to local Y scale
            // Y_global = x(0)*sin_t + x(1)*cos_t + y_c
            // error = Y_global - y_target
            T y_err = x(0) * sin_t + x(1) * cos_t + y_c - y_target;

            // [Architect's New Law] X는 무시한다. 오직 차선 유지와 속도뿐.
            res(idx++) = T(0.0);                                          // X 오차 무시
            res(idx++) = T(sqrt(200.0)) * y_err;                          // Y (차선) 엄격히 유지
            res(idx++) = T(sqrt(500.0)) * (x(2) - T(x_ref_local(2)));     // Yaw 엄격히 정렬
            res(idx++) = T(sqrt(50.0))  * (x(3) - T(10.0));              // 속도 10m/s 유지

            // 조향 자체를 죄악시한다 (R 가중치 극대화)
            res(idx++) = T(sqrt(10.0))   * u(0); // Accel
            res(idx++) = T(sqrt(5000.0)) * u(1); // Steering (매우 무겁게)

            // 장애물 회피 (Margin 1.5m)
            for (size_t i = 0; i < 10; ++i) {
                T dx = x(0) - T(local_obstacles[i].x);
                T dy = x(1) - T(local_obstacles[i].y);
                T dist_sq = dx * dx + dy * dy;
                T r_safe = T(local_obstacles[i].r + 1.5);
                T viol = r_safe * r_safe - dist_sq;
                res(idx++) = (Optimization::get_value(viol) > 0.0) ? T(sqrt(20000.0)) * viol : T(0.0);
            }

            // [Hard Defense] Vx가 5m/s 아래로 떨어지면 기하급수적 처벌 (역주행 방지)
            T v_min_viol = T(5.0) - x(3);
            res(idx++) = (Optimization::get_value(v_min_viol) > 0.0) ? T(1000.0) * v_min_viol : T(0.0);

            return res;
        }

        void update_dynamic_weights(const StaticVector<double, Nx>& x_curr) {
            double vx = std::max(0.1, x_curr(3)); 
            R(1) = 20.0 + 1.0 * (vx * vx); // 고속 시 조향 더 억제
            R_rate(1) = 300.0 + 2.0 * (vx * vx);
        }

        void shift_sequence() {
            for (size_t k = 0; k < H - 1; ++k) U_guess[k] = U_guess[k + 1];
            U_guess[H-1](0) *= 0.5; U_guess[H-1](1) *= 0.5; // 더 가파른 감쇠
        }

        NMPCResult execute_fallback(NMPCResult& res, const std::string& reason) {
            for (size_t k = 0; k < H; ++k) { U_guess[k](0) = -4.0; U_guess[k](1) = 0.0; }
            u_last = U_guess[0]; res.success = false; res.fallback_triggered = true; res.status_msg = reason;
            return res;
        }

        NMPCResult solve(const StaticVector<double, Nx>& x_curr_global, const StaticVector<double, Nx>& x_ref_global, int max_sqp_iter = 1) {
            NMPCResult result;
            auto T_g2l = utils::SE2Transform::get_global_to_local(x_curr_global(0), x_curr_global(1), x_curr_global(2));
            for (size_t i = 0; i < 10; ++i) {
                local_obstacles[i].r = global_obstacles[i].r;
                utils::SE2Transform::transform_point(T_g2l, global_obstacles[i].x, global_obstacles[i].y, local_obstacles[i].x, local_obstacles[i].y);
            }

            StaticVector<double, Nx> x_curr_local; x_curr_local.set_zero();
            x_curr_local(3) = std::max(0.1, x_curr_global(3)); 
            x_curr_local(4) = x_curr_global(4); x_curr_local(5) = x_curr_global(5);

            StaticVector<double, Nx> x_ref_local;
            utils::SE2Transform::transform_point(T_g2l, x_ref_global(0), x_ref_global(1), x_ref_local(0), x_ref_local(1));
            x_ref_local(2) = normalize_angle(x_ref_global(2) - x_curr_global(2)); 
            x_ref_local(3) = x_ref_global(3); x_ref_local(4) = x_ref_global(4); x_ref_local(5) = x_ref_global(5);

            update_dynamic_weights(x_curr_local);
            vehicle::DynamicBicycleModel model;

            for (int sqp = 0; sqp < max_sqp_iter; ++sqp) {
                result.sqp_iterations = sqp + 1;
                X_pred[0] = x_curr_local;
                for (size_t k = 0; k < H; ++k) X_pred[k+1] = integrator::step_rk4<Nx, Nu>(model, X_pred[k], U_guess[k], dt);

                for (size_t k = 0; k < H; ++k) {
                    StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k - 1];
                    using ADVar = DualVec<double, Nx + Nu>;
                    StaticVector<ADVar, Nx> x_dual; StaticVector<ADVar, Nu> u_dual; StaticVector<ADVar, Nu> u_prev_dual;
                    for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
                    for (size_t i = 0; i < Nu; ++i) {
                        u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);
                        u_prev_dual(i) = ADVar(u_prev(i));
                    }

                    StaticVector<ADVar, Nx> x_next_dual = integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);
                    for (size_t i = 0; i < Nx; ++i) {
                        for (size_t j = 0; j < Nx; ++j) riccati.A[k](i, j) = x_next_dual(i).g[j];
                        for (size_t j = 0; j < Nu; ++j) riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                        riccati.d[k](i) = 0.0; 
                    }
                    StaticVector<ADVar, 25> res_dual = eval_node_residuals(x_dual, u_dual, u_prev_dual, x_ref_local, x_curr_global, x_ref_global);
                    riccati.Q[k].set_zero(); riccati.R[k].set_zero(); riccati.q[k].set_zero(); riccati.r[k].set_zero();
                    for (size_t res_idx = 0; res_idx < 25; ++res_idx) {
                        double r_val = res_dual(res_idx).v;
                        for (size_t i = 0; i < Nx; ++i) {
                            double J_xi = res_dual(res_idx).g[i]; riccati.q[k](i) += J_xi * r_val;
                            for (size_t j = 0; j < Nx; ++j) riccati.Q[k](i, j) += J_xi * res_dual(res_idx).g[j];
                        }
                        for (size_t i = 0; i < Nu; ++i) {
                            double J_ui = res_dual(res_idx).g[Nx + i]; riccati.r[k](i) += J_ui * r_val;
                            for (size_t j = 0; j < Nu; ++j) riccati.R[k](i, j) += J_ui * res_dual(res_idx).g[Nx + j];
                        }
                    }
                    // 강력한 정규화(Damping) 추가
                    for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += 5.0;
                    for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += 500.0;
                }
                riccati.Q[H].set_zero(); riccati.q[H].set_zero();
                double sin_t = std::sin(x_curr_global(2));
                double cos_t = std::cos(x_curr_global(2));
                for(size_t i=0; i<Nx; ++i) {
                    if (i == 1) {
                        double err_y = X_pred[H](0) * sin_t + X_pred[H](1) * cos_t + x_curr_global(1) - x_ref_global(1);
                        riccati.Q[H](1, 1) += Qf(1) * cos_t * cos_t;
                        riccati.Q[H](0, 0) += Qf(1) * sin_t * sin_t;
                        riccati.Q[H](0, 1) += Qf(1) * sin_t * cos_t;
                        riccati.Q[H](1, 0) += Qf(1) * sin_t * cos_t;
                        riccati.q[H](1) += Qf(1) * err_y * cos_t;
                        riccati.q[H](0) += Qf(1) * err_y * sin_t;
                    } else if (i != 0) { // i=0 is handled above, but original Q(0)=0 anyway
                        double err = X_pred[H](i) - x_ref_local(i);
                        riccati.Q[H](i, i) += Qf(i); 
                        riccati.q[H](i) += Qf(i) * err;
                    }
                }

                if (riccati.solve() != SolverStatus::SUCCESS) return execute_fallback(result, "Factorization Failed");

                double max_step = 0.0;
                for (size_t k = 0; k < H; ++k) {
                    for (size_t i = 0; i < Nu; ++i) {
                        // Trust Region: 더 보수적인 스텝 클램핑
                        riccati.du[k](i) = std::clamp(riccati.du[k](i), -0.5, 0.5);
                        max_step = std::max(max_step, std::abs(riccati.du[k](i)));
                    }
                }
                result.max_kkt_error = max_step;
                if (std::isnan(max_step) || max_step > 5.0) return execute_fallback(result, "Divergence Detected");

                double alpha = 0.3; // 극단적으로 보수적인 보폭
                for (size_t k = 0; k < H; ++k) {
                    for (size_t i = 0; i < Nu; ++i) U_guess[k](i) += alpha * riccati.du[k](i);
                }
                if (max_step < 1e-4) break; 
            }
            u_last = U_guess[0]; result.success = true; return result;
        }
    };
}
}
#endif