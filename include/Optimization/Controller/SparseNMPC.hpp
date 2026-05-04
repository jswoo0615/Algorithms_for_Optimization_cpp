#ifndef OPTIMIZATION_SPARSE_NMPC_HPP_
#define OPTIMIZATION_SPARSE_NMPC_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "Optimization/Dual.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

namespace Optimization {

// [Architect's Update: Frenet 좌표계 기준 장애물]
struct ObstacleFrenet {
    double s = 0.0;
    double d = 0.0;
    double r = 0.5;
    double vs = 0.0;  // 도로 중심선을 따라가는 속도
    double vd = 0.0;  // 차선을 넘나드는 속도
};

namespace controller {

struct NMPCResult {
    bool success = false;
    bool fallback_triggered = false;
    double max_kkt_error = 0.0;
    int sqp_iterations = 0;
    std::string status_msg = "OK";
};

struct NMPCTuningConfig {
    double Q_D = 200.0;   
    double Q_mu = 500.0;  
    double Q_Vx = 50.0;

    double R_Steer = 5000.0; // u(0)
    double R_Accel = 10.0;   // u(1)

    double R_Steer_Rate = 50000.0; // Jitter 방지를 위해 대폭 상향
    double R_Accel_Rate = 100.0;

    double Obstacle_Penalty = 20000.0;
    double Obstacle_Margin = 1.5;

    double damping_Q = 5.0;
    double damping_R = 500.0;

    double barrier_mu = 0.05;
    double u_min[2] = {-0.26, -5.0}; // [Steer_min, Accel_min]
    double u_max[2] = {0.26, 3.0};   // [Steer_max, Accel_max]
    double safety_fraction = 0.95;
    double step_alpha = 0.5;

    double kappa = 0.0;       
    double target_vx = 10.0;  
};

// [Architect's Decision: 잔차 벡터 크기 정의]
// 상태(4) + 제어(2) + 제어변화율(2) + 장애물(10) + 최소속도(1) = 19
// 넉넉하게 30으로 설정하여 solve_rt_qp와 동기화합니다.
constexpr size_t NUM_RESIDALS = 30;
constexpr size_t MAX_OBS = 10;

template <size_t H, size_t Nx = 6, size_t Nu = 2>
class SparseNMPC {
   public:
    double dt;
    std::array<StaticVector<double, Nu>, H> U_guess;
    std::array<StaticVector<double, Nx>, H + 1> X_pred;
    StaticVector<double, Nu> u_last;

    std::array<ObstacleFrenet, MAX_OBS> obstacles;
    solver::RiccatiSolver<H, Nx, Nu> riccati;

    SparseNMPC() {
        u_last.set_zero();
        dt = 0.1;
        for (size_t k = 0; k < H; ++k) U_guess[k].set_zero();
        for (size_t i = 0; i < MAX_OBS; ++i) {
            obstacles[i].s = 10000.0;
            obstacles[i].d = 10000.0;
            obstacles[i].r = 0.1;
            obstacles[i].vs = 0.0;
            obstacles[i].vd = 0.0;
        }
    }

    template <typename T>
    StaticVector<T, NUM_RESIDALS> eval_node_residuals(const StaticVector<T, Nx>& x, 
                                                     const StaticVector<T, Nu>& u, 
                                                     const StaticVector<T, Nu>& u_prev,
                                                     const NMPCTuningConfig& config,
                                                     int k) {
        using std::abs;
        using std::sqrt;

        StaticVector<T, NUM_RESIDALS> res; res.set_zero();
        int idx = 0;

        T s = x(0);
        T d = x(1);
        T mu = x(2);
        T vx = x(3);

        // 1. 상태 추종 페널티
        res(idx++) = T(0.0);                     // S
        res(idx++) = T(sqrt(config.Q_D)) * d;    // D
        res(idx++) = T(sqrt(config.Q_mu)) * mu;  // mu
        res(idx++) = T(sqrt(config.Q_Vx)) * (vx - T(config.target_vx));

        // 2. 제어 입력 페널티 (u0=Steer, u1=Accel)
        res(idx++) = T(sqrt(config.R_Steer)) * u(0);
        res(idx++) = T(sqrt(config.R_Accel)) * u(1);
        res(idx++) = T(sqrt(config.R_Steer_Rate)) * (u(0) - u_prev(0));
        res(idx++) = T(sqrt(config.R_Accel_Rate)) * (u(1) - u_prev(1));

        // 3. 다중 장애물 회피 (Frenet)
        double time_future = k * dt;
        for (size_t i = 0; i < MAX_OBS; ++i) {
            T obs_pred_s = T(obstacles[i].s + obstacles[i].vs * time_future);
            T obs_pred_d = T(obstacles[i].d + obstacles[i].vd * time_future);

            T ds = s - obs_pred_s;
            T dd = d - obs_pred_d;
            T dist_sq = ds * ds + dd * dd;
            
            T safety_margin = T(obstacles[i].r + config.Obstacle_Margin);
            T violation = safety_margin * safety_margin - dist_sq;

            // Soft Constraint
            if (Optimization::get_value(violation) > 0.0) {
                res(idx++) = T(sqrt(config.Obstacle_Penalty)) * violation;
            } else {
                res(idx++) = T(0.0);
            }
        }

        // 4. 최소 속도 보장 (특이점 방어)
        T v_min_viol = T(1.0) - vx; // 1.0m/s 이하로 떨어지지 않도록
        res(idx++) = (Optimization::get_value(v_min_viol) > 0.0) ? T(1000.0) * v_min_viol : T(0.0);

        return res;
    }

    void shift_sequence() {
        for (size_t k = 0; k < H - 1; ++k) U_guess[k] = U_guess[k + 1];
        U_guess[H - 1](0) *= 0.5;
        U_guess[H - 1](1) *= 0.5;
    }

    NMPCResult execute_fallback(NMPCResult& res, const std::string& reason) {
        for (size_t k = 0; k < H; ++k) {
            U_guess[k](0) = 0.0;  // 핸들 중립
            U_guess[k](1) = -4.0; // 강한 제동
        }
        u_last = U_guess[0];
        res.success = false;
        res.fallback_triggered = true;
        res.status_msg = reason;
        return res;
    }

    NMPCResult solve_rt_qp(const StaticVector<double, Nx>& x_curr_frenet,
                           const NMPCTuningConfig& config) {
        NMPCResult result;
        result.sqp_iterations = 1;

        vehicle::DynamicBicycleModel model;
        model.kappa = config.kappa;

        X_pred[0] = x_curr_frenet;
        X_pred[0](3) = std::max(0.1, x_curr_frenet(3)); 

        for (size_t k = 0; k < H; ++k) {
            X_pred[k + 1] = integrator::step_rk4<Nx, Nu>(model, X_pred[k], U_guess[k], dt);
        }

        for (size_t k = 0; k < H; ++k) {
            StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k - 1];
            using ADVar = DualVec<double, Nx + Nu>;
            StaticVector<ADVar, Nx> x_dual;
            StaticVector<ADVar, Nu> u_dual;
            StaticVector<ADVar, Nu> u_prev_dual;

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

            // [CRITICAL FIX: NUM_RESIDALS로 크기 동기화]
            StaticVector<ADVar, NUM_RESIDALS> res_dual = eval_node_residuals(x_dual, u_dual, u_prev_dual, config, k);

            riccati.Q[k].set_zero();
            riccati.R[k].set_zero();
            riccati.q[k].set_zero();
            riccati.r[k].set_zero();

            for (size_t res_idx = 0; res_idx < NUM_RESIDALS; ++res_idx) {
                double r_val = res_dual(res_idx).v;
                for (size_t i = 0; i < Nx; ++i) {
                    double J_xi = res_dual(res_idx).g[i];
                    riccati.q[k](i) += J_xi * r_val;
                    for (size_t j = 0; j < Nx; ++j)
                        riccati.Q[k](i, j) += J_xi * res_dual(res_idx).g[j];
                }
                for (size_t i = 0; i < Nu; ++i) {
                    double J_ui = res_dual(res_idx).g[Nx + i];
                    riccati.r[k](i) += J_ui * r_val;
                    for (size_t j = 0; j < Nu; ++j)
                        riccati.R[k](i, j) += J_ui * res_dual(res_idx).g[Nx + j];
                }
            }

            // Phase 2: Interior-Point Method
            for (size_t i = 0; i < Nu; ++i) {
                double u_val = U_guess[k](i);
                double s_upper = config.u_max[i] - u_val;
                double s_lower = u_val - config.u_min[i];

                s_upper = std::max(s_upper, 1e-3);
                s_lower = std::max(s_lower, 1e-3);

                double grad_barrier = config.barrier_mu * (1.0 / s_upper - 1.0 / s_lower);
                double hess_barrier = config.barrier_mu * (1.0 / (s_upper * s_upper) + 1.0 / (s_lower * s_lower));

                grad_barrier = std::clamp(grad_barrier, -100.0, 100.0);
                hess_barrier = std::min(hess_barrier, 500.0);

                riccati.r[k](i) += grad_barrier;
                riccati.R[k](i, i) += hess_barrier;
            }

            for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += config.damping_Q;
            for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += config.damping_R;
        }

        // Phase 3: Terminal Cost
        riccati.Q[H].set_zero();
        riccati.q[H].set_zero();

        for (size_t i = 0; i < Nx; ++i) {
            if (i == 1 || i == 2 || i == 3) {
                double err = (i == 3) ? (X_pred[H](i) - config.target_vx) : X_pred[H](i);
                double qf_i = (i == 1)   ? config.Q_D * 5.0
                              : (i == 2) ? config.Q_mu * 5.0
                                         : config.Q_Vx * 5.0;
                riccati.Q[H](i, i) += qf_i;
                riccati.q[H](i) += qf_i * err;
            }
        }

        // Phase 4: Riccati Solve & Update
        if (riccati.solve() != SolverStatus::SUCCESS)
            return execute_fallback(result, "Factorization Failed");

        double alpha = config.step_alpha;
        for (size_t k = 0; k < H; ++k) {
            for (size_t i = 0; i < Nu; ++i) {
                double du = riccati.du[k](i);
                if (du > 1e-4) {
                    double max_alpha = (config.u_max[i] - U_guess[k](i)) / du;
                    if (max_alpha > 0.0)
                        alpha = std::min(alpha, config.safety_fraction * max_alpha);
                } else if (du < -1e-4) {
                    double max_alpha = (config.u_min[i] - U_guess[k](i)) / du;
                    if (max_alpha > 0.0)
                        alpha = std::min(alpha, config.safety_fraction * max_alpha);
                }
            }
        }

        alpha = std::max(alpha, 0.05);

        double max_kkt = 0.0;
        for (size_t k = 0; k < H; ++k) {
            for (size_t i = 0; i < Nu; ++i) {
                U_guess[k](i) += alpha * riccati.du[k](i);
                U_guess[k](i) = std::clamp(U_guess[k](i), config.u_min[i] + 1e-3, config.u_max[i] - 1e-3);
                max_kkt = std::max(max_kkt, std::abs(riccati.du[k](i)));
            }
        }

        result.max_kkt_error = max_kkt;
        if (std::isnan(max_kkt) || max_kkt > 20.0)
            return execute_fallback(result, "Divergence Detected");

        u_last = U_guess[0];
        result.success = true;
        return result;
    }
};
}  // namespace controller
}  // namespace Optimization
#endif