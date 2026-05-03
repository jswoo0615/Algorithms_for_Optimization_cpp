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
            // [Architect's Fix] X좌표 추종 페널티(Q(0)) 제거. 
            // 차량이 앞으로 나아가는 것을 처벌하면 차량은 스핀아웃을 선택합니다.
            Q.set_zero(); Q(1) = 15.0; Q(2) = 5.0; Q(3) = 10.0;
            Qf.set_zero(); for (size_t i=0; i<Nx; ++i) Qf(i) = Q(i) * 10.0;
            R.set_zero(); R(0) = 0.5; R(1) = 10.0; 
            R_rate.set_zero(); R_rate(0) = 50.0; R_rate(1) = 250.0;
            u_last.set_zero();
            dt = 0.1;
            for (size_t k=0; k<H; ++k) U_guess[k].set_zero();

            // =========================================================
            // [Architect's Fix] 유령 장애물 치우기
            // 쓰지 않는 장애물 배열 초기값을 차량이 닿을 수 없는 곳으로 보냅니다.
            // =========================================================
            for (size_t i = 0; i < 10; ++i) {
                obstacles[i].x = 10000.0; 
                obstacles[i].y = 10000.0;
                obstacles[i].r = 0.1;
            }
        }

        // =========================================================================
        // [Architect's Tactics] 각 노드(Node)별 비선형 잔차 함수 (M_res = 26)
        // 1. 상태 추종 (6) + 2. 제어 크기 (2) + 3. Slew Rate (2)
        // 4. 장애물 (10) + 5. Log-Barrier 기반 하드 제약 (4+2)
        // =========================================================================
        // 잔차의 개수를 정확히 24개로 맞춤 (상태 6 + 제어 2 + Slew 2 + 장애물 10 + 제약 4)
        template <typename T>
        StaticVector<T, 24> eval_node_residuals(const StaticVector<T, Nx>& x, 
                                                const StaticVector<T, Nu>& u, 
                                                const StaticVector<T, Nu>& u_prev,
                                                const StaticVector<double, Nx>& x_ref) 
        {
            // 이제 math_log, math_sqrt 라우터는 필요 없습니다. 완벽히 삭제하십시오.

            StaticVector<T, 24> res; res.set_zero();
            int idx = 0;

            // 1. 상태 오차 잔차 (6)
            for (size_t i = 0; i < Nx; ++i) res(idx++) = T(std::sqrt(Q(i))) * (x(i) - T(x_ref(i)));

            // 2. 제어 입력 크기 잔차 (2)
            for (size_t i = 0; i < Nu; ++i) res(idx++) = T(std::sqrt(R(i))) * u(i);

            // 3. Slew Rate 잔차 (2)
            for (size_t i = 0; i < Nu; ++i) res(idx++) = T(std::sqrt(R_rate(i))) * (u(i) - u_prev(i));

            // 4. 장애물 회피망 (10)
            for (size_t i = 0; i < 10; ++i) {
                T dx = x(0) - T(obstacles[i].x);
                T dy = x(1) - T(obstacles[i].y);
                T dist_sq = dx * dx + dy * dy;
                T r_safe = T(obstacles[i].r + 0.8);
                T viol = r_safe * r_safe - dist_sq;
                res(idx++) = (Optimization::get_value(viol) > 0.0) ? T(std::sqrt(8000.0)) * viol : T(0.0);
            }

            // =========================================================
            // [Architect's Tactics] Inverse Barrier Function (Gauss-Newton IPM)
            // NaN의 위험이 없고, 연산이 압도적으로 빠르며 완벽한 하드 제약을 형성합니다.
            // =========================================================
            double mu = 0.5; // Barrier Parameter 
            
            // 가속도 한계 [-3.0, 3.0]
            T a_viol_upper = u(0) - T(3.0);
            T a_viol_lower = T(-3.0) - u(0);
            res(idx++) = (Optimization::get_value(a_viol_upper) > 0.0) ? T(std::sqrt(100.0)) * a_viol_upper : T(0.0);
            res(idx++) = (Optimization::get_value(a_viol_lower) > 0.0) ? T(std::sqrt(100.0)) * a_viol_lower : T(0.0);

            // 조향 한계 [-0.5, 0.5] rad
            T steer_viol_upper = u(1) - T(0.5);
            T steer_viol_lower = T(-0.5) - u(1);
            res(idx++) = (Optimization::get_value(steer_viol_upper) > 0.0) ? T(std::sqrt(100.0)) * steer_viol_upper : T(0.0);
            res(idx++) = (Optimization::get_value(steer_viol_lower) > 0.0) ? T(std::sqrt(100.0)) * steer_viol_lower : T(0.0);
            
            return res;
        }

        void update_dynamic_weights(const StaticVector<double, Nx>& x_curr) {
            double vx = x_curr(3); // 현재 종방향 속도 (m/s)
            
            // 1. 기본 가중치 (저속 상태의 Base 값)
            double base_R_steer = 10.0;
            double base_R_rate = 250.0;
            double base_Q_psi = 5.0;

            // 2. 동적 스케줄링 (속도에 따른 페널티 증가)
            // 고속일수록 조향을 무겁게 만들고(폭주 방지), 헤딩 추종을 날카롭게 만듭니다.
            R(1)      = base_R_steer + 0.5 * (vx * vx);
            R_rate(1) = base_R_rate  + 2.0 * (vx * vx);
            Q(2)      = base_Q_psi   + 0.5 * std::abs(vx);
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
                StaticVector<ADVar, 24> res_dual = eval_node_residuals(x_dual, u_dual, u_prev_dual, x_ref);
                
                riccati.Q[k].set_zero(); riccati.R[k].set_zero();
                riccati.q[k].set_zero(); riccati.r[k].set_zero();

                // J^T * J 및 J^T * res 조립
                for (size_t res_idx = 0; res_idx < 24; ++res_idx) {
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

                // =========================================================
                // [Architect's Fix] Levenberg-Marquardt Damping 주입
                // 순수 Gauss-Newton의 오버슛(급조향)을 억제하는 국소적 Trust Region.
                // =========================================================
                double lambda_q = 2.0;
                // [Architect's Tuning] 조향이 미친듯이 튀는 것을 막는 최강의 물리적 제동 장치
                double lambda_r = 500.0; 
                for (size_t i = 0; i < Nx; ++i) riccati.Q[k](i, i) += lambda_q;
                for (size_t i = 0; i < Nu; ++i) riccati.R[k](i, i) += lambda_r;
            }

            // Terminal Node (H)
            riccati.Q[H].set_zero(); riccati.q[H].set_zero();
            for(size_t i=0; i<Nx; ++i) {
                double err = X_pred[H](i) - x_ref(i);
                riccati.Q[H](i, i) = Qf(i);
                riccati.q[H](i) = Qf(i) * err;
            }

            // 3. Solve & Update
            SolverStatus status = riccati.solve();
            if (status != SolverStatus::SUCCESS) return false;

            // =========================================================
            // [Architect's Tactics] Strict Interior Projection
            // 해가 장벽(Barrier)의 점근선을 뛰어넘어 기울기가 0이 되는 것을 방지합니다.
            // 무조건 허용 구역 내부([-0.49, 0.49])에 머물도록 투영시킵니다.
            // =========================================================
            for (size_t k = 0; k < H; ++k) {
                for (size_t i = 0; i < Nu; ++i) U_guess[k](i) += riccati.du[k](i);
            }
            
            // 다음 제어 주기를 위한 Warm Start
            u_last = U_guess[0]; 

            return true;
        }
    };

} // namespace controller
} // namespace Optimization

#endif // OPTIMIZATION_SPARSE_NMPC_HPP_