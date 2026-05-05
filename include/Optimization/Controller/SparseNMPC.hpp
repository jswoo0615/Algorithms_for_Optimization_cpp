#ifndef OPTIMIZATION_SPARSE_NMPC_HPP_
#define OPTIMIZATION_SPARSE_NMPC_HPP_

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

namespace Optimization {

// =========================================================================
// [1] 환경 및 튜닝 파라미터 구조체
// =========================================================================

struct ObstacleFrenet {
    double s = 0.0;
    double d = 0.0;
    double r = 0.5;
    double vs = 0.0;  // 도로 중심선을 따라가는 종방향 속도
    double vd = 0.0;  // 차선을 넘나드는 횡방향 속도
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

    double R_Steer = 5000.0;        
    double R_Accel = 10.0;          

    double R_Steer_Rate = 50000.0;  // Jitter 방지 (고주파 진동 억제)
    double R_Accel_Rate = 100.0;

    double Obstacle_Penalty = 20000.0;
    double Obstacle_Margin = 1.5;

    double damping_Q = 5.0;         // Hessian Regularization
    double damping_R = 500.0;

    double barrier_mu = 0.05;       // Primal Barrier Parameter
    double u_min[2] = {-0.26, -5.0};  
    double u_max[2] = {0.26, 3.0};    
    double safety_fraction = 0.95;  // Line Search Margin
    double step_alpha = 0.5;        // Base Step Size

    double kappa = 0.0;             // 도로 곡률
    double target_vx = 10.0;        // 목표 속도
};

constexpr size_t NUM_RESIDUALS = 30;
constexpr size_t MAX_OBS = 10;

// =========================================================================
// [2] Sparse NMPC 코어 엔진 (Real-Time Iteration 기반)
// =========================================================================

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

    /**
     * @brief KKT 잔차 노드 평가 (Zero-Allocation AD Engine)
     */
    template <typename T>
    inline StaticVector<T, NUM_RESIDUALS> eval_node_residuals(const StaticVector<T, Nx>& x,
                                                              const StaticVector<T, Nu>& u,
                                                              const StaticVector<T, Nu>& u_prev,
                                                              const NMPCTuningConfig& config, int k) {
        StaticVector<T, NUM_RESIDUALS> res;
        res.set_zero();
        int idx = 0;

        T s = x(0);
        T d = x(1);
        T mu = x(2);
        T vx = x(3);

        // 1. State Tracking Penalty
        res(idx++) = T(0.0);
        res(idx++) = T(std::sqrt(config.Q_D)) * d;
        res(idx++) = T(std::sqrt(config.Q_mu)) * mu;
        res(idx++) = T(std::sqrt(config.Q_Vx)) * (vx - T(config.target_vx));

        // 2. Control Input Penalty
        res(idx++) = T(std::sqrt(config.R_Steer)) * u(0);
        res(idx++) = T(std::sqrt(config.R_Accel)) * u(1);
        res(idx++) = T(std::sqrt(config.R_Steer_Rate)) * (u(0) - u_prev(0));
        res(idx++) = T(std::sqrt(config.R_Accel_Rate)) * (u(1) - u_prev(1));

        // 3. Dynamic Obstacle Avoidance (Frenet)
        double time_future = k * dt;
        for (size_t i = 0; i < MAX_OBS; ++i) {
            T obs_pred_s = T(obstacles[i].s + obstacles[i].vs * time_future);
            T obs_pred_d = T(obstacles[i].d + obstacles[i].vd * time_future);

            T ds = s - obs_pred_s;
            T dd = d - obs_pred_d;
            T dist_sq = ds * ds + dd * dd;

            T safety_margin = T(obstacles[i].r + config.Obstacle_Margin);
            T violation = safety_margin * safety_margin - dist_sq;

            // Soft Constraint Activation
            if (Optimization::get_value(violation) > 0.0) {
                res(idx++) = T(std::sqrt(config.Obstacle_Penalty)) * violation;
            } else {
                res(idx++) = T(0.0);
            }
        }

        // 4. Minimum Velocity Shield (특이점 붕괴 방어)
        T v_min_viol = T(1.0) - vx; 
        res(idx++) = (Optimization::get_value(v_min_viol) > 0.0) ? T(1000.0) * v_min_viol : T(0.0);

        return res;
    }

    /**
     * @brief Receding Horizon: 이전 해를 당겨와 웜스타트(Warm-start) 구성
     */
    inline void shift_sequence() {
        for (size_t k = 0; k < H - 1; ++k) U_guess[k] = U_guess[k + 1];
        // 꼬리 부분은 감쇠(Decay)를 주어 안정성 확보
        U_guess[H - 1](0) *= 0.5;
        U_guess[H - 1](1) *= 0.5;
    }

    /**
     * @brief 긴급 제동 로직 (Fallback Strategy)
     */
    inline NMPCResult execute_fallback(NMPCResult& res, const std::string& reason) {
        for (size_t k = 0; k < H; ++k) {
            U_guess[k](0) = 0.0;   // 조향 중립
            U_guess[k](1) = -4.0;  // 강력한 긴급 제동 (AEB)
        }
        u_last = U_guess[0];
        res.success = false;
        res.fallback_triggered = true;
        res.status_msg = reason;
        return res;
    }

    /**
     * @brief Real-Time QP Formulation & Solve (10ms Deadline Engine)
     */
    NMPCResult solve_rt_qp(const StaticVector<double, Nx>& x_curr_frenet,
                           const NMPCTuningConfig& config) {
        NMPCResult result;
        result.sqp_iterations = 1; // RTI는 매 사이클 단 1번의 QP만 풉니다.

        vehicle::DynamicBicycleModel model;
        model.kappa = config.kappa;

        // 속도 0 특이점 방어 (Singularity Shield)
        X_pred[0] = x_curr_frenet;
        X_pred[0](3) = MathTraits<double>::max(0.1, x_curr_frenet(3));

        // 1. Forward Pass (명목 궤적 생성)
        for (size_t k = 0; k < H; ++k) {
            X_pred[k + 1] = integrator::step_rk4<Nx, Nu>(model, X_pred[k], U_guess[k], dt);
        }

        // 2. Backward Assembly (KKT 시스템 조립)
        using ADVar = DualVec<double, Nx + Nu>;
        
        for (size_t k = 0; k < H; ++k) {
            StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k - 1];
            
            StaticVector<ADVar, Nx> x_dual;
            StaticVector<ADVar, Nu> u_dual;
            StaticVector<ADVar, Nu> u_prev_dual;

            // [Architect's Layout] Dual 인덱스 매핑 (x: 0~Nx-1, u: Nx~Nx+Nu-1)
            for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
            for (size_t i = 0; i < Nu; ++i) {
                u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);
                u_prev_dual(i) = ADVar(u_prev(i)); // Gradient 0 (Constant Parameter)
            }

            // A_k, B_k 추출
            StaticVector<ADVar, Nx> x_next_dual = integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);
            for (size_t i = 0; i < Nx; ++i) {
                for (size_t j = 0; j < Nx; ++j) riccati.A[k](i, j) = x_next_dual(i).g[j];
                for (size_t j = 0; j < Nu; ++j) riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                riccati.d[k](i) = 0.0; // Multiple Shooting의 Defect는 여기선 0 (단일 명목 궤적 기반)
            }

            // Gauss-Newton Hessian (Q, R) 및 Gradient (q, r) 조립
            StaticVector<ADVar, NUM_RESIDUALS> res_dual = eval_node_residuals(x_dual, u_dual, u_prev_dual, config, k);

            riccati.Q[k].set_zero();
            riccati.R[k].set_zero();
            riccati.q[k].set_zero();
            riccati.r[k].set_zero();

            for (size_t res_idx = 0; res_idx < NUM_RESIDUALS; ++res_idx) {
                double r_val = res_dual(res_idx).v;
                
                // J_x 추출 및 Q, q 누적
                for (size_t i = 0; i < Nx; ++i) {
                    double J_xi = res_dual(res_idx).g[i];
                    riccati.q[k](i) += J_xi * r_val;
                    for (size_t j = 0; j < Nx; ++j) {
                        riccati.Q[k](i, j) += J_xi * res_dual(res_idx).g[j];
                    }
                }
                
                // J_u 추출 및 R, r 누적
                for (size_t i = 0; i < Nu; ++i) {
                    double J_ui = res_dual(res_idx).g[Nx + i];
                    riccati.r[k](i) += J_ui * r_val;
                    for (size_t j = 0; j < Nu; ++j) {
                        riccati.R[k](i, j) += J_ui * res_dual(res_idx).g[Nx + j];
                    }
                }
            }

            // Primal Log-Barrier (제어 한계 방어)
            for (size_t i = 0; i < Nu; ++i) {
                double u_val = U_guess[k](i);
                double s_upper = MathTraits<double>::max(config.u_max[i] - u_val, 1e-3);
                double s_lower = MathTraits<double>::max(u_val - config.u_min[i], 1e-3);

                double grad_barrier = config.barrier_mu * (1.0 / s_upper - 1.0 / s_lower);
                double hess_barrier = config.barrier_mu * (1.0 / (s_upper * s_upper) + 1.0 / (s_lower * s_lower));

                // 수치 붕괴 방지용 클램핑
                grad_barrier = std::clamp(grad_barrier, -100.0, 100.0);
                hess_barrier = MathTraits<double>::min(hess_barrier, 500.0);

                riccati.r[k](i) += grad_barrier;
                riccati.R[k](i, i) += hess_barrier;
            }

            // Hessian Regularization (SPD 강제)
            for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += config.damping_Q;
            for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += config.damping_R;
        }

        // 3. Terminal Node Assembly
        riccati.Q[H].set_zero();
        riccati.q[H].set_zero();

        for (size_t i = 0; i < Nx; ++i) {
            if (i == 1 || i == 2 || i == 3) {
                double err = (i == 3) ? (X_pred[H](i) - config.target_vx) : X_pred[H](i);
                double qf_i = (i == 1) ? config.Q_D * 5.0 : 
                              (i == 2) ? config.Q_mu * 5.0 : 
                                         config.Q_Vx * 5.0;
                riccati.Q[H](i, i) += qf_i;
                riccati.q[H](i) += qf_i * err;
            }
        }

        // 4. Riccati Solver 타격
        if (riccati.solve() != SolverStatus::SUCCESS) {
            return execute_fallback(result, "Factorization Failed (Riccati)");
        }

        // 5. Line Search (Fraction-to-Boundary)
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

        // 6. Primal Update 및 발산 감시
        double max_kkt = 0.0;
        for (size_t k = 0; k < H; ++k) {
            for (size_t i = 0; i < Nu; ++i) {
                U_guess[k](i) += alpha * riccati.du[k](i);
                U_guess[k](i) = std::clamp(U_guess[k](i), config.u_min[i] + 1e-3, config.u_max[i] - 1e-3);
                max_kkt = MathTraits<double>::max(max_kkt, MathTraits<double>::abs(riccati.du[k](i)));
            }
        }

        result.max_kkt_error = max_kkt;
        if (std::isnan(max_kkt) || max_kkt > 20.0) {
            return execute_fallback(result, "Divergence Detected (KKT Exploded)");
        }

        u_last = U_guess[0]; // 다음 스텝의 Rate Penalty 연산을 위해 저장
        result.success = true;
        return result;
    }
};

}  // namespace controller
}  // namespace Optimization

#endif  // OPTIMIZATION_SPARSE_NMPC_HPP_