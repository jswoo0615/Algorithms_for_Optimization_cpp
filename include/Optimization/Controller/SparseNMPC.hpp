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
// 이제 장애물은 X, Y가 아닌 S(종방향 거리), D(횡방향 편차)로 정의됩니다.
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
    // [Architect's Update: Q_Y, Q_Yaw -> Q_D, Q_mu 로 전환]
    double Q_D = 200.0;   // 횡방향 오차 페널티
    double Q_mu = 500.0;  // 헤딩 오차 페널티
    double Q_Vx = 50.0;

    double R_Accel = 10.0;
    double R_Steer = 5000.0;

    double R_Accel_Rate = 100.0;
    double R_Steer_Rate = 20000.0;

    double Obstacle_Penalty = 20000.0;
    double Obstacle_Margin = 1.5;

    double damping_Q = 5.0;
    double damping_R = 500.0;

    double barrier_mu = 0.05;
    double u_min[2] = {-0.26, -5.0};
    double u_max[2] = {0.26, 3.0};
    double safety_fraction = 0.95;
    double step_alpha = 0.5;

    // [New] 주행 환경 파라미터
    double kappa = 0.0;       // 현재 도로의 곡률 (1/R)
    double target_vx = 10.0;  // 목표 종방향 속도
};

template <size_t H, size_t Nx = 6, size_t Nu = 2>
class SparseNMPC {
   public:
    double dt;
    std::array<StaticVector<double, Nu>, H> U_guess;
    std::array<StaticVector<double, Nx>, H + 1> X_pred;
    StaticVector<double, Nu> u_last;

    // 로컬/글로벌의 구분이 사라졌습니다. 오직 S-D 프레임 상의 장애물만 존재합니다.
    std::array<ObstacleFrenet, 10> obstacles;
    solver::RiccatiSolver<H, Nx, Nu> riccati;

    SparseNMPC() {
        u_last.set_zero();
        dt = 0.1;
        for (size_t k = 0; k < H; ++k) U_guess[k].set_zero();
        for (size_t i = 0; i < 10; ++i) {
            obstacles[i].s = 10000.0;
            obstacles[i].d = 10000.0;
            obstacles[i].r = 0.1;
            obstacles[i].vs = 0.0;
            obstacles[i].vd = 0.0;
        }
    }

    template <typename T>
    StaticVector<T, 25> eval_node_residuals(const StaticVector<T, Nx>& x,
                                            const StaticVector<T, Nu>& u,
                                            const StaticVector<T, Nu>& u_prev,
                                            const NMPCTuningConfig& config, int k) {
        using std::abs;
        using std::sqrt;

        StaticVector<T, 25> res;
        res.set_zero();
        int idx = 0;

        // Frenet State 추출
        T s = x(0);
        T d = x(1);
        T mu = x(2);
        T vx = x(3);

        // [Architect's Masterpiece: 궁극의 단순함]
        // 복잡한 좌표 변환(sin, cos)이 모두 증발했습니다. D와 mu를 0으로 누르기만 하면 됩니다.
        res(idx++) = T(0.0);                     // S는 전진하도록 놔둠 (페널티 없음)
        res(idx++) = T(sqrt(config.Q_D)) * d;    // 차선 중앙 유지
        res(idx++) = T(sqrt(config.Q_mu)) * mu;  // 차선 방향 정렬
        res(idx++) = T(sqrt(config.Q_Vx)) * (vx - T(config.target_vx));

        // [Architect's Fix: u(0)=Steer, u(1)=Accel 인덱스 완벽 동기화]
        res(idx++) = T(sqrt(config.R_Steer)) * u(0);
        res(idx++) = T(sqrt(config.R_Accel)) * u(1);
        res(idx++) = T(sqrt(config.R_Steer_Rate)) * (u(0) - u_prev(0));
        res(idx++) = T(sqrt(config.R_Accel_Rate)) * (u(1) - u_prev(1));

        // 동적 장애물 (Frenet 공간 내 이동)
        double time_future = k * dt;
        for (size_t i = 0; i < 10; ++i) {
            T obs_pred_s = T(obstacles[i].s + obstacles[i].vs * time_future);
            T obs_pred_d = T(obstacles[i].d + obstacles[i].vd * time_future);

            T ds = s - obs_pred_s;
            T dd = d - obs_pred_d;
            T dist_sq = ds * ds + dd * dd;
            T r_safe = T(obstacles[i].r + config.Obstacle_Margin);
            T viol = r_safe * r_safe - dist_sq;
            res(idx++) = (Optimization::get_value(viol) > 0.0)
                             ? T(sqrt(config.Obstacle_Penalty)) * viol
                             : T(0.0);
        }

        T v_min_viol = T(5.0) - vx;
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
            U_guess[k](0) = -4.0;
            U_guess[k](1) = 0.0;
        }
        u_last = U_guess[0];
        res.success = false;
        res.fallback_triggered = true;
        res.status_msg = reason;
        return res;
    }

    // [Architect's Update] 글로벌 좌표계 투영 과정을 통째로 삭제
    // NMPC는 오직 Frenet 상태(S, D, mu, Vx, Vy, r) 하나만 입력받습니다.
    NMPCResult solve_rt_qp(const StaticVector<double, Nx>& x_curr_frenet,
                           const NMPCTuningConfig& config) {
        NMPCResult result;
        result.sqp_iterations = 1;

        vehicle::DynamicBicycleModel model;
        model.kappa = config.kappa;  // 동역학 모델에 현재 곡률 주입

        // Phase 1: 선형화 및 RK4 예측
        X_pred[0] = x_curr_frenet;
        X_pred[0](3) = std::max(0.1, x_curr_frenet(3));  // 특이점 방어

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

            StaticVector<ADVar, Nx> x_next_dual =
                integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);

            for (size_t i = 0; i < Nx; ++i) {
                for (size_t j = 0; j < Nx; ++j) riccati.A[k](i, j) = x_next_dual(i).g[j];
                for (size_t j = 0; j < Nu; ++j) riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                riccati.d[k](i) = 0.0;
            }

            StaticVector<ADVar, 25> res_dual =
                eval_node_residuals(x_dual, u_dual, u_prev_dual, config, k);

            riccati.Q[k].set_zero();
            riccati.R[k].set_zero();
            riccati.q[k].set_zero();
            riccati.r[k].set_zero();

            for (size_t res_idx = 0; res_idx < 25; ++res_idx) {
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
                double hess_barrier =
                    config.barrier_mu * (1.0 / (s_upper * s_upper) + 1.0 / (s_lower * s_lower));

                grad_barrier = std::clamp(grad_barrier, -100.0, 100.0);
                hess_barrier = std::min(hess_barrier, 500.0);

                riccati.r[k](i) += grad_barrier;
                riccati.R[k](i, i) += hess_barrier;
            }

            for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += config.damping_Q;
            for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += config.damping_R;
        }

        // Phase 3: Terminal Cost (단순화의 극치)
        riccati.Q[H].set_zero();
        riccati.q[H].set_zero();

        for (size_t i = 0; i < Nx; ++i) {
            // 오직 D(1), mu(2), Vx(3) 만 타겟팅
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
                U_guess[k](i) =
                    std::clamp(U_guess[k](i), config.u_min[i] + 1e-3, config.u_max[i] - 1e-3);
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