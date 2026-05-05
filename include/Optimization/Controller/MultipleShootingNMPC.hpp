#ifndef OPTIMIZATION_MULTIPLE_SHOOTING_NMPC_HPP_
#define OPTIMIZATION_MULTIPLE_SHOOTING_NMPC_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "Optimization/Dual.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Solver/RiccatiSolver.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"
#include "Optimization/Controller/SparseNMPC.hpp"

namespace Optimization {
namespace controller {

template <size_t H, size_t Nx = 6, size_t Nu = 2>
class MultipleShootingNMPC {
   public:
    double dt;
    
    std::array<StaticVector<double, Nu>, H> U_guess;
    std::array<StaticVector<double, Nx>, H + 1> X_guess; 
    StaticVector<double, Nu> u_last;

    std::array<ObstacleFrenet, MAX_OBS> obstacles;
    solver::RiccatiSolver<H, Nx, Nu> riccati;

    MultipleShootingNMPC() {
        u_last.set_zero();
        dt = 0.1;
        for (size_t k = 0; k < H; ++k) U_guess[k].set_zero();
        for (size_t k = 0; k <= H; ++k) X_guess[k].set_zero();
        
        for (size_t i = 0; i < MAX_OBS; ++i) {
            obstacles[i].s = 10000.0;
            obstacles[i].d = 10000.0;
            obstacles[i].r = 0.1;
        }
    }

    inline void shift_sequence() {
        for (size_t k = 0; k < H - 1; ++k) {
            U_guess[k] = U_guess[k + 1];
            X_guess[k] = X_guess[k + 1];
        }
        X_guess[H - 1] = X_guess[H];
        U_guess[H - 1](0) *= 0.5;
        U_guess[H - 1](1) *= 0.5;

        vehicle::DynamicBicycleModel model;
        X_guess[H] = integrator::step_rk4<Nx, Nu>(model, X_guess[H - 1], U_guess[H - 1], dt);
    }

    inline NMPCResult execute_fallback(NMPCResult& res, const std::string& reason) {
        for (size_t k = 0; k < H; ++k) {
            U_guess[k](0) = 0.0;
            U_guess[k](1) = -4.0;
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

        // 1. 초기 상태 구속
        X_guess[0] = x_curr_frenet;
        X_guess[0](3) = MathTraits<double>::max(0.1, x_curr_frenet(3)); 

        // =========================================================================
        // [Architect's Shield: Cold Start Ignition]
        // 예측 궤적의 끝단(X_guess[H]) 속도가 0이라는 것은 메모리가 방금 할당되었음을 의미합니다.
        // 거대한 결함(Defect) 오차로 인한 폭발을 막기 위해 1회 명목 궤적을 쏴줍니다.
        // =========================================================================
        if (MathTraits<double>::abs(X_guess[H](3)) < 1e-3) {
            for (size_t k = 0; k < H; ++k) {
                X_guess[k + 1] = integrator::step_rk4<Nx, Nu>(model, X_guess[k], U_guess[k], dt);
            }
        }

        SparseNMPC<H, Nx, Nu> cost_evaluator;
        cost_evaluator.dt = this->dt;
        cost_evaluator.obstacles = this->obstacles;

        using ADVar = DualVec<double, Nx + Nu>;

        for (size_t k = 0; k < H; ++k) {
            StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k - 1];
            
            StaticVector<ADVar, Nx> x_dual;
            StaticVector<ADVar, Nu> u_dual;
            StaticVector<ADVar, Nu> u_prev_dual;

            for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(X_guess[k](i), i);
            for (size_t i = 0; i < Nu; ++i) {
                u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);
                u_prev_dual(i) = ADVar(u_prev(i)); 
            }

            StaticVector<ADVar, Nx> x_next_dual = integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);
            
            StaticVector<double, Nx> x_pred_val;
            for (size_t i = 0; i < Nx; ++i) {
                x_pred_val(i) = x_next_dual(i).v;
                for (size_t j = 0; j < Nx; ++j) riccati.A[k](i, j) = x_next_dual(i).g[j];
                for (size_t j = 0; j < Nu; ++j) riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
            }

            // Defect Constraint (마법의 열쇠)
            riccati.d[k] = x_pred_val - X_guess[k + 1];

            StaticVector<ADVar, NUM_RESIDUALS> res_dual = 
                cost_evaluator.eval_node_residuals(x_dual, u_dual, u_prev_dual, config, k);

            riccati.Q[k].set_zero();
            riccati.R[k].set_zero();
            riccati.q[k].set_zero();
            riccati.r[k].set_zero();

            for (size_t res_idx = 0; res_idx < NUM_RESIDUALS; ++res_idx) {
                double r_val = res_dual(res_idx).v;
                for (size_t i = 0; i < Nx; ++i) {
                    double J_xi = res_dual(res_idx).g[i];
                    riccati.q[k](i) += J_xi * r_val;
                    for (size_t j = 0; j < Nx; ++j) riccati.Q[k](i, j) += J_xi * res_dual(res_idx).g[j];
                }
                for (size_t i = 0; i < Nu; ++i) {
                    double J_ui = res_dual(res_idx).g[Nx + i];
                    riccati.r[k](i) += J_ui * r_val;
                    for (size_t j = 0; j < Nu; ++j) riccati.R[k](i, j) += J_ui * res_dual(res_idx).g[Nx + j];
                }
            }

            for (size_t i = 0; i < Nu; ++i) {
                double u_val = U_guess[k](i);
                double s_upper = MathTraits<double>::max(config.u_max[i] - u_val, 1e-3);
                double s_lower = MathTraits<double>::max(u_val - config.u_min[i], 1e-3);

                double grad_barrier = config.barrier_mu * (1.0 / s_upper - 1.0 / s_lower);
                double hess_barrier = config.barrier_mu * (1.0 / (s_upper * s_upper) + 1.0 / (s_lower * s_lower));

                riccati.r[k](i) += std::clamp(grad_barrier, -100.0, 100.0);
                riccati.R[k](i, i) += MathTraits<double>::min(hess_barrier, 500.0);
            }

            for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += config.damping_Q;
            for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += config.damping_R;
        }

        riccati.Q[H].set_zero();
        riccati.q[H].set_zero();

        for (size_t i = 0; i < Nx; ++i) {
            if (i == 1 || i == 2 || i == 3) {
                double err = (i == 3) ? (X_guess[H](i) - config.target_vx) : X_guess[H](i);
                double qf_i = (i == 1) ? config.Q_D * 5.0 : 
                              (i == 2) ? config.Q_mu * 5.0 : 
                                         config.Q_Vx * 5.0;
                riccati.Q[H](i, i) += qf_i;
                riccati.q[H](i) += qf_i * err;
            }
        }

        if (riccati.solve() != SolverStatus::SUCCESS) {
            return execute_fallback(result, "Factorization Failed (MS Riccati)");
        }

        double alpha = config.step_alpha;
        for (size_t k = 0; k < H; ++k) {
            for (size_t i = 0; i < Nu; ++i) {
                double du = riccati.du[k](i);
                if (du > 1e-4) {
                    double max_alpha = (config.u_max[i] - U_guess[k](i)) / du;
                    if (max_alpha > 0.0) alpha = MathTraits<double>::min(alpha, config.safety_fraction * max_alpha);
                } else if (du < -1e-4) {
                    double max_alpha = (config.u_min[i] - U_guess[k](i)) / du;
                    if (max_alpha > 0.0) alpha = MathTraits<double>::min(alpha, config.safety_fraction * max_alpha);
                }
            }
        }
        alpha = MathTraits<double>::max(alpha, 0.05);

        double max_kkt = 0.0;
        for (size_t k = 0; k < H; ++k) {
            for (size_t i = 0; i < Nu; ++i) {
                U_guess[k](i) += alpha * riccati.du[k](i);
                U_guess[k](i) = std::clamp(U_guess[k](i), config.u_min[i] + 1e-3, config.u_max[i] - 1e-3);
                max_kkt = MathTraits<double>::max(max_kkt, MathTraits<double>::abs(riccati.du[k](i)));
            }
            for (size_t i = 0; i < Nx; ++i) {
                X_guess[k + 1](i) += alpha * riccati.dx[k + 1](i);
            }
        }

        result.max_kkt_error = max_kkt;
        if (std::isnan(max_kkt) || max_kkt > 20.0) {
            return execute_fallback(result, "Divergence Detected (MS KKT Exploded)");
        }

        u_last = U_guess[0];
        result.success = true;
        return result;
    }
};

}  // namespace controller
}  // namespace Optimization

#endif  // OPTIMIZATION_MULTIPLE_SHOOTING_NMPC_HPP_