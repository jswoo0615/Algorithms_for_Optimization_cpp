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
#include "Optimization/Utils/RigidTransform.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

namespace Optimization {

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double r = 0.5;
    double vx = 0.0;  // [Architect's Add] 장애물의 X축 속도
    double vy = 0.0;  // [Architect's Add] 장애물의 Y축 속도
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
    double Q_Y = 200.0;
    double Q_Yaw = 500.0;
    double Q_Vx = 50.0;

    template <typename T>
    StaticVector<T, 25> eval_node_residuals(const StaticVector<T, Nx>& x,
                                            const StaticVector<T, Nu>& u,
                                            const StaticVector<T, Nu>& u_prev,
                                            const StaticVector<double, Nx>& x_ref_local,
                                            const StaticVector<double, Nx>& x_curr_global,
                                            const StaticVector<double, Nx>& x_ref_global,
                                            const NMPCTuningConfig& config,
                                            int k)  // [Architect's Add] 현재 호라이즌 인덱스 주입
    {
        using std::abs;
        using std::sqrt;

        StaticVector<T, 25> res;
        res.set_zero();
        int idx = 0;

        T theta_c = T(x_curr_global(2));
        T sin_t = T(std::sin(x_curr_global(2)));
        T cos_t = T(std::cos(x_curr_global(2)));
        T y_c = T(x_curr_global(1));
        T y_target = T(x_ref_global(1));

        T y_err = x(0) * sin_t + x(1) * cos_t + y_c - y_target;

        res(idx++) = T(0.0);
        res(idx++) = T(sqrt(config.Q_Y)) * y_err;
        res(idx++) = T(sqrt(config.Q_Yaw)) * (x(2) - T(x_ref_local(2)));
        res(idx++) = T(sqrt(config.Q_Vx)) * (x(3) - T(10.0));

        res(idx++) = T(sqrt(config.R_Accel)) * u(0);
        res(idx++) = T(sqrt(config.R_Steer)) * u(1);
        res(idx++) = T(sqrt(config.R_Accel_Rate)) * (u(0) - u_prev(0));
        res(idx++) = T(sqrt(config.R_Steer_Rate)) * (u(1) - u_prev(1));

        // [Architect's Masterpiece: Dynamic Obstacle Prediction]
        double time_future = k * dt;  // k번째 스텝의 미래 시간
        for (size_t i = 0; i < 10; ++i) {
            // 미래 시간의 장애물 로컬 예상 위치 계산
            T obs_pred_x = T(local_obstacles[i].x + local_obstacles[i].vx * time_future);
            T obs_pred_y = T(local_obstacles[i].y + local_obstacles[i].vy * time_future);

            T dx = x(0) - obs_pred_x;
            T dy = x(1) - obs_pred_y;
            T dist_sq = dx * dx + dy * dy;
            T r_safe = T(local_obstacles[i].r + config.Obstacle_Margin);
            T viol = r_safe * r_safe - dist_sq;
            res(idx++) = (Optimization::get_value(viol) > 0.0)
                             ? T(sqrt(config.Obstacle_Penalty)) * viol
                             : T(0.0);
        }

        T v_min_viol = T(5.0) - x(3);
        res(idx++) = (Optimization::get_value(v_min_viol) > 0.0) ? T(1000.0) * v_min_viol : T(0.0);

        double Obstacle_Penalty = 20000.0;
        double Obstacle_Margin = 1.5;

        double damping_Q = 5.0;
        double damping_R = 500.0;

        // [IPM] Interior-Point Method 파라미터
        double barrier_mu = 0.05;         // Log-Barrier 강도 (초기엔 약간 크게)
        double u_min[2] = {-5.0, -0.52};  // Accel Min, Steer Min (-30 deg)
        double u_max[2] = {3.0, 0.52};    // Accel Max, Steer Max (+30 deg)
        double safety_fraction = 0.95;  // 경계 접근 허용치 (1.0이면 경계 터치 허용)

        double step_alpha = 0.5;  // 기본 보폭
    };

    inline double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    template <size_t H, size_t Nx = 6, size_t Nu = 2>
    class SparseNMPC {
       public:
        double dt;
        std::array<StaticVector<double, Nu>, H> U_guess;
        std::array<StaticVector<double, Nx>, H + 1> X_pred;
        StaticVector<double, Nu> u_last;
        std::array<Obstacle, 10> global_obstacles;
        std::array<Obstacle, 10> local_obstacles;
        solver::RiccatiSolver<H, Nx, Nu> riccati;

        SparseNMPC() {
            u_last.set_zero();
            dt = 0.1;
            for (size_t k = 0; k < H; ++k) U_guess[k].set_zero();
            for (size_t i = 0; i < 10; ++i) {
                global_obstacles[i].x = 10000.0;
                global_obstacles[i].y = 10000.0;
                global_obstacles[i].r = 0.1;
            }
        }

        template <typename T>
        StaticVector<T, 25> eval_node_residuals(const StaticVector<T, Nx>& x,
                                                const StaticVector<T, Nu>& u,
                                                const StaticVector<T, Nu>& u_prev,
                                                const StaticVector<double, Nx>& x_ref_local,
                                                const StaticVector<double, Nx>& x_curr_global,
                                                const StaticVector<double, Nx>& x_ref_global,
                                                const NMPCTuningConfig& config) {
            using std::abs;
            using std::sqrt;

            StaticVector<T, 25> res;
            res.set_zero();
            int idx = 0;

            T theta_c = T(x_curr_global(2));
            T sin_t = T(std::sin(x_curr_global(2)));
            T cos_t = T(std::cos(x_curr_global(2)));
            T y_c = T(x_curr_global(1));
            T y_target = T(x_ref_global(1));

            T y_err = x(0) * sin_t + x(1) * cos_t + y_c - y_target;

            // 상태 페널티
            res(idx++) = T(0.0);
            res(idx++) = T(sqrt(config.Q_Y)) * y_err;
            res(idx++) = T(sqrt(config.Q_Yaw)) * (x(2) - T(x_ref_local(2)));
            res(idx++) = T(sqrt(config.Q_Vx)) * (x(3) - T(10.0));

            // [Architect's Fix: 절대 제어 페널티 복원]
            // IPM은 경계를 막을 뿐, 중앙으로 당겨주지 않습니다. 반드시 필요합니다.
            res(idx++) = T(sqrt(config.R_Accel)) * u(0);
            res(idx++) = T(sqrt(config.R_Steer)) * u(1);

            // 제어 변화량 페널티
            res(idx++) = T(sqrt(config.R_Accel_Rate)) * (u(0) - u_prev(0));
            res(idx++) = T(sqrt(config.R_Steer_Rate)) * (u(1) - u_prev(1));

            // 장애물 회피
            for (size_t i = 0; i < 10; ++i) {
                T dx = x(0) - T(local_obstacles[i].x);
                T dy = x(1) - T(local_obstacles[i].y);
                T dist_sq = dx * dx + dy * dy;
                T r_safe = T(local_obstacles[i].r + config.Obstacle_Margin);
                T viol = r_safe * r_safe - dist_sq;
                res(idx++) = (Optimization::get_value(viol) > 0.0)
                                 ? T(sqrt(config.Obstacle_Penalty)) * viol
                                 : T(0.0);
            }

            NMPCResult solve_rt_qp(const StaticVector<double, Nx>& x_curr_global,
                                   const StaticVector<double, Nx>& x_ref_global,
                                   const NMPCTuningConfig& config) {
                NMPCResult result;
                result.sqp_iterations = 1;

                auto T_g2l = utils::SE2Transform::get_global_to_local(
                    x_curr_global(0), x_curr_global(1), x_curr_global(2));

                // 속도 변환을 위한 회전 행렬 성분 추출
                double cos_yaw = std::cos(-x_curr_global(2));
                double sin_yaw = std::sin(-x_curr_global(2));
                for (size_t i = 0; i < 10; ++i) {
                    local_obstacles[i].r = global_obstacles[i].r;

                    // 1. 위치 변환 (병진 + 회전)
                    utils::SE2Transform::transform_point(
                        T_g2l, global_obstacles[i].x, global_obstacles[i].y, local_obstacles[i].x,
                        local_obstacles[i].y);

                    // 2. 속도 변환 (순수 회전만 적용)
                    local_obstacles[i].vx =
                        global_obstacles[i].vx * cos_yaw - global_obstacles[i].vy * sin_yaw;
                    local_obstacles[i].vy =
                        global_obstacles[i].vx * sin_yaw + global_obstacles[i].vy * cos_yaw;
                }

                return res;
            }

            void shift_sequence() {
                for (size_t k = 0; k < H - 1; ++k) U_guess[k] = U_guess[k + 1];
                U_guess[H - 1](0) *= 0.5;
                U_guess[H - 1](1) *= 0.5;
            }

            NMPCResult execute_fallback(NMPCResult & res, const std::string& reason) {
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

            NMPCResult solve_rt_qp(const StaticVector<double, Nx>& x_curr_global,
                                   const StaticVector<double, Nx>& x_ref_global,
                                   const NMPCTuningConfig& config) {
                NMPCResult result;
                result.sqp_iterations = 1;

                auto T_g2l = utils::SE2Transform::get_global_to_local(
                    x_curr_global(0), x_curr_global(1), x_curr_global(2));
                for (size_t i = 0; i < 10; ++i) {
                    local_obstacles[i].r = global_obstacles[i].r;
                    utils::SE2Transform::transform_point(
                        T_g2l, global_obstacles[i].x, global_obstacles[i].y, local_obstacles[i].x,
                        local_obstacles[i].y);
                }

                StaticVector<double, Nx> x_curr_local;
                x_curr_local.set_zero();
                x_curr_local(3) = std::max(0.1, x_curr_global(3));
                x_curr_local(4) = x_curr_global(4);
                x_curr_local(5) = x_curr_global(5);

                StaticVector<double, Nx> x_ref_local;
                utils::SE2Transform::transform_point(T_g2l, x_ref_global(0), x_ref_global(1),
                                                     x_ref_local(0), x_ref_local(1));
                x_ref_local(2) = normalize_angle(x_ref_global(2) - x_curr_global(2));
                x_ref_local(3) = x_ref_global(3);
                x_ref_local(4) = x_ref_global(4);
                x_ref_local(5) = x_ref_global(5);

                vehicle::DynamicBicycleModel model;

                X_pred[0] = x_curr_local;
                for (size_t k = 0; k < H; ++k)
                    X_pred[k + 1] = integrator::step_rk4<Nx, Nu>(model, X_pred[k], U_guess[k], dt);

                for (size_t k = 0; k < H; ++k) {
                    StaticVector<double, Nu> u_prev = (k == 0) ? u_last : U_guess[k - 1];
                    using ADVar = DualVec<double, Nx + Nu>;
                    StaticVector<ADVar, Nx> x_dual;
                    StaticVector<ADVar, Nu> u_dual;
                    StaticVector<ADVar, Nu> u_prev_dual;
                    for (size_t i = 0; i < Nx; ++i)
                        x_dual(i) = ADVar::make_variable(X_pred[k](i), i);
                    for (size_t i = 0; i < Nu; ++i) {
                        u_dual(i) = ADVar::make_variable(U_guess[k](i), Nx + i);
                        u_prev_dual(i) = ADVar(u_prev(i));
                    }

                    StaticVector<ADVar, Nx> x_next_dual =
                        integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);
                    for (size_t i = 0; i < Nx; ++i) {
                        for (size_t j = 0; j < Nx; ++j) riccati.A[k](i, j) = x_next_dual(i).g[j];
                        for (size_t j = 0; j < Nu; ++j)
                            riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                        riccati.d[k](i) = 0.0;
                    }
                    StaticVector<ADVar, 25> res_dual =
                        eval_node_residuals(x_dual, u_dual, u_prev_dual, x_ref_local, x_curr_global,
                                            x_ref_global, config);
                    riccati.Q[k].set_zero();
                    riccati.R[k].set_zero();
                    riccati.q[k].set_zero();
                    riccati.r[k].set_zero();

                    for (size_t res_idx = 0; res_idx < 25; ++res_idx) {
                        double r_val = res_dual(res_idx).v;
                        for (size_t i = 0; i < Nx; ++i) {
                            for (size_t j = 0; j < Nx; ++j)
                                riccati.A[k](i, j) = x_next_dual(i).g[j];
                            for (size_t j = 0; j < Nu; ++j)
                                riccati.B[k](i, j) = x_next_dual(i).g[Nx + j];
                            riccati.d[k](i) = 0.0;
                        }
                        StaticVector<ADVar, 25> res_dual =
                            eval_node_residuals(x_dual, u_dual, u_prev_dual, x_ref_local,
                                                x_curr_global, x_ref_global, config, k);  // k 전달
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
                        for (size_t i = 0; i < Nu; ++i) {
                            double J_ui = res_dual(res_idx).g[Nx + i];
                            riccati.r[k](i) += J_ui * r_val;
                            for (size_t j = 0; j < Nu; ++j)
                                riccati.R[k](i, j) += J_ui * res_dual(res_idx).g[Nx + j];
                        }
                    }

                    // [Architect's Masterpiece: Safe IPM Hessian & Gradient Update]
                    for (size_t i = 0; i < Nu; ++i) {
                        double u_val = U_guess[k](i);
                        // 물리적 한계치 (config.u_max, u_min)
                        double s_upper = config.u_max[i] - u_val;
                        double s_lower = u_val - config.u_min[i];

                        // 수치적 폭발 방지 (Zero division 방어)
                        s_upper = std::max(s_upper, 1e-3);
                        s_lower = std::max(s_lower, 1e-3);

                        // Gradient: 경계에 가까워질수록 밀어냄
                        double grad_barrier = config.barrier_mu * (1.0 / s_upper - 1.0 / s_lower);

                        // Hessian: 곡률 정보 추가
                        double hess_barrier = config.barrier_mu * (1.0 / (s_upper * s_upper) +
                                                                   1.0 / (s_lower * s_lower));

                        // 강력한 클램핑으로 Riccati 붕괴 방지
                        grad_barrier = std::clamp(grad_barrier, -100.0, 100.0);
                        hess_barrier = std::min(hess_barrier, 500.0);

                        riccati.r[k](i) += grad_barrier;
                        riccati.R[k](i, i) += hess_barrier;
                    }

                    for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += config.damping_Q;
                    for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += config.damping_R;
                }

                riccati.Q[H].set_zero();
                riccati.q[H].set_zero();
                double sin_t = std::sin(x_curr_global(2));
                double cos_t = std::cos(x_curr_global(2));
                for (size_t i = 0; i < Nx; ++i) {
                    if (i == 1) {
                        double err_y = X_pred[H](0) * sin_t + X_pred[H](1) * cos_t +
                                       x_curr_global(1) - x_ref_global(1);
                        double qf_y = config.Q_Y * 5.0;
                        riccati.Q[H](1, 1) += qf_y * cos_t * cos_t;
                        riccati.Q[H](0, 0) += qf_y * sin_t * sin_t;
                        riccati.Q[H](0, 1) += qf_y * sin_t * cos_t;
                        riccati.Q[H](1, 0) += qf_y * sin_t * cos_t;
                        riccati.q[H](1) += qf_y * err_y * cos_t;
                        riccati.q[H](0) += qf_y * err_y * sin_t;
                    } else if (i != 0) {
                        double err = X_pred[H](i) - x_ref_local(i);
                        double qf_i =
                            (i == 2) ? config.Q_Yaw * 5.0 : ((i == 3) ? config.Q_Vx * 5.0 : 0.0);
                        riccati.Q[H](i, i) += qf_i;
                        riccati.q[H](i) += qf_i * err;
                    }
                }

                if (riccati.solve() != SolverStatus::SUCCESS)
                    return execute_fallback(result, "Factorization Failed");

                // [IPM: Fraction-to-the-boundary Rule 보완]
                double alpha = config.step_alpha;
                for (size_t k = 0; k < H; ++k) {
                    for (size_t i = 0; i < Nu; ++i) {
                        double du = riccati.du[k](i);
                        // 분모가 너무 작으면(1e-4) 무시하여 수치 불안정 방지
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

                // [Stall Prevention] alpha가 0이 되어 솔버가 굳어버리는 현상 방지
                alpha = std::max(alpha, 0.05);  // 최소 5%의 보폭은 허용하여 루프 탈출 유도

                double max_kkt = 0.0;
                for (size_t k = 0; k < H; ++k) {
                    for (size_t i = 0; i < Nu; ++i) {
                        U_guess[k](i) += alpha * riccati.du[k](i);

                        // [Strict Interior Projection]
                        U_guess[k](i) = std::clamp(U_guess[k](i), config.u_min[i] + 1e-3,
                                                   config.u_max[i] - 1e-3);

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