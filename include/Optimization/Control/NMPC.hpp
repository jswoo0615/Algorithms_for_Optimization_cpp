#ifndef OPTIMIZATION_NMPC_HPP_
#define OPTIMIZATION_NMPC_HPP_

#include "Optimization/SQPSolver.hpp"
#include "Optimization/Physics/VehicleModel.hpp"
#include "Optimization/Simulation/Integrator.hpp"
#include <array>

namespace Optimization {

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double r = 0.5; 
};

template <size_t Np>
class NMPCController {
public:
    static constexpr size_t Nx = 6;
    static constexpr size_t Nu = 2;
    static constexpr size_t N_vars = Np * Nu;
    static constexpr size_t N_eq = 0;
    
    // [Architect's Scale-up] 가속도(상/하한) + 조향(상/하한) = 스텝당 4개 제약
    // 15스텝 * 4 = 총 60개의 하드 제약조건을 IPM 엔진에 직접 주입합니다.
    static constexpr size_t N_ineq = Np * 4; 

    SQPSolver<N_vars, N_eq, N_ineq> sqp;
    DynamicBicycleModel model;
    double dt = 0.1; 

    StaticVector<double, Nx> Q;
    StaticVector<double, Nx> Qf;
    StaticVector<double, Nu> R;
    StaticVector<double, Nu> R_rate; 
    StaticVector<double, Nx> x_ref;
    
    StaticVector<double, Nu> u_last; 
    std::array<Obstacle, 10> obstacles;

    NMPCController() {
        Q.set_zero();
        Q(0) = 15.0; Q(1) = 15.0; Q(2) = 5.0; Q(3) = 10.0; 
        
        Qf.set_zero();
        for(size_t i = 0; i < Nx; ++i) Qf(static_cast<int>(i)) = Q(static_cast<int>(i)) * 10.0;

        R.set_zero();
        R(0) = 0.5; R(1) = 10.0; 

        R_rate.set_zero();
        R_rate(0) = 50.0;   
        R_rate(1) = 250.0; // 사용자가 도출한 Sweet Spot 유지
        
        u_last.set_zero();
    }

    struct CostFunc {
        StaticVector<double, Nx> x0;
        StaticVector<double, Nx> x_ref;
        StaticVector<double, Nx> Q, Qf;
        StaticVector<double, Nu> R, R_rate;
        StaticVector<double, Nu> u_last;
        DynamicBicycleModel model;
        double dt;
        std::array<Obstacle, 10> obs;

        template <typename T>
        T operator()(const StaticVector<T, N_vars>& U) const {
            T cost(0.0);
            StaticVector<T, Nx> x;
            for(size_t i = 0; i < Nx; ++i) x(static_cast<int>(i)) = T(x0(static_cast<int>(i)));

            StaticVector<T, Nu> u_prev;
            u_prev(0) = T(u_last(0));
            u_prev(1) = T(u_last(1));

            for (size_t k = 0; k < Np; ++k) {
                StaticVector<T, Nu> u;
                // [Architect's Fix] 임시방편이었던 Tanh 스쿼싱 완전 삭제
                u(0) = U(static_cast<int>(k * Nu + 0));
                u(1) = U(static_cast<int>(k * Nu + 1));

                // 1. 상태 오차 페널티
                for(size_t i = 0; i < Nx; ++i) {
                    T err = x(static_cast<int>(i)) - T(x_ref(static_cast<int>(i)));
                    cost += T(Q(static_cast<int>(i))) * err * err;
                }
                
                // 2. 크기 및 변화율(Slew Rate) 페널티
                cost += T(R(0)) * u(0) * u(0) + T(R(1)) * u(1) * u(1);
                T delta_a = u(0) - u_prev(0);
                T delta_steer = u(1) - u_prev(1);
                cost += T(R_rate(0)) * delta_a * delta_a + T(R_rate(1)) * delta_steer * delta_steer;
                u_prev = u; 

                // 3. 장애물 회피망 (장애물은 볼록성이 보장되지 않아 Soft Penalty 유지)
                for (size_t obs_i = 0; obs_i < 10; ++obs_i) {
                    T dx = x(0) - T(obs[obs_i].x);
                    T dy = x(1) - T(obs[obs_i].y);
                    T dist_sq = dx * dx + dy * dy;
                    T r_safe = T(obs[obs_i].r + 0.8); 
                    T violation = r_safe * r_safe - dist_sq;
                    if (Optimization::get_value(violation) > 0.0) {
                        cost += T(8000.0) * violation * violation; 
                    }
                }

                // 4. 플랜트 롤아웃
                x = Integrator::rk4<Nx, Nu, DynamicBicycleModel, T>(model, x, u, dt);
            }

            // 5. Terminal Cost
            for(size_t i = 0; i < Nx; ++i) {
                T err = x(static_cast<int>(i)) - T(x_ref(static_cast<int>(i)));
                cost += T(Qf(static_cast<int>(i))) * err * err;
            }
            return cost;
        }
    };

    struct DummyEq {
        template <typename T>
        StaticVector<T, 0> operator()(const StaticVector<T, N_vars>& U) const { (void)U; return StaticVector<T, 0>(); }
    };

    // [Architect's Add] 하드 제약 조건 정의 (A_ineq * x - b_ineq <= 0)
    struct BoundIneq {
        template <typename T>
        StaticVector<T, N_ineq> operator()(const StaticVector<T, N_vars>& U) const {
            StaticVector<T, N_ineq> ineq;
            for (size_t k = 0; k < Np; ++k) {
                T a = U(static_cast<int>(k * Nu + 0));
                T delta = U(static_cast<int>(k * Nu + 1));
                
                // 가속도 한계 [-3.0, 3.0]
                ineq(static_cast<int>(k * 4 + 0)) = a - T(3.0);
                ineq(static_cast<int>(k * 4 + 1)) = -a - T(3.0);
                
                // 조향각 한계 [-0.5, 0.5] rad
                ineq(static_cast<int>(k * 4 + 2)) = delta - T(0.5);
                ineq(static_cast<int>(k * 4 + 3)) = -delta - T(0.5);
            }
            return ineq;
        }
    };

    bool compute_control(const StaticVector<double, Nx>& current_x, StaticVector<double, N_vars>& U_guess) {
        CostFunc cost_f{current_x, x_ref, Q, Qf, R, R_rate, u_last, model, dt, obstacles};
        DummyEq eq_f;
        BoundIneq ineq_f;

        bool success = sqp.solve(U_guess, cost_f, eq_f, ineq_f, 15); 
        
        // IPM이 하드 제약을 완벽히 보장하므로, 변형 없이 실제 물리값 그대로 사용합니다.
        u_last(0) = U_guess(0);
        u_last(1) = U_guess(1);
        
        return success;
    }
};

} // namespace Optimization

#endif // OPTIMIZATION_NMPC_HPP_