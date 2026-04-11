#ifndef OPTIMIZATION_RTI_NMPC_HPP_
#define OPTIMIZATION_RTI_NMPC_HPP_

#include <array>
#include <cmath>

#include "Optimization/Physics/VehicleModel.hpp"
#include "Optimization/RTISolver.hpp"
#include "Optimization/Simulation/Integrator.hpp"

namespace Optimization {

template <size_t Np>
class RTINMPCController {
   public:
    static constexpr size_t Nx = 6;
    static constexpr size_t Nu = 2;
    static constexpr size_t N_vars = Np * Nu;
    static constexpr size_t N_eq = 0;
    static constexpr size_t N_ineq = Np * 4;

    // [Architect's Dimension Formulation]
    // State(15*6) + Ctrl(15*2) + Rate(15*2) + Obs(15*10) + Terminal(6) = 306
    static constexpr size_t N_res = Np * Nx + Np * Nu + Np * Nu + Np * 10 + Nx;

    // 강력한 306차원 잔차 기반의 RTI 솔버
    RTISolver<N_vars, N_eq, N_ineq, N_res> rti;

    DynamicBicycleModel model;
    double dt = 0.2;

    StaticVector<double, Nx> Q;
    StaticVector<double, Nx> Qf;
    StaticVector<double, Nu> R;
    StaticVector<double, Nu> R_rate;
    StaticVector<double, Nx> x_ref;

    StaticVector<double, Nu> u_last;

    struct RTIObstacle {
        double x = 0.0;
        double y = 0.0;
        double r = 0.5;
    };
    std::array<RTIObstacle, 10> obstacles;

    RTINMPCController() {
        Q.set_zero();
        Q(0) = 10.0;
        Q(1) = 10.0;
        Q(2) = 0.1;
        Q(3) = 5.0;

        Qf.set_zero();
        Qf(0) = 50.0;
        Qf(1) = 50.0;
        Qf(2) = 0.1;
        Qf(3) = 200.0;

        R.set_zero();
        R(0) = 5.0;
        R(1) = 10.0;

        R_rate.set_zero();
        R_rate(0) = 50.0;
        R_rate(1) = 250.0;

        u_last.set_zero();
    }

    // [Architect's Core] Residual Function
    // 스칼라 Cost를 버리고, 각 항목별 오차를 쪼개어 N_res 벡터에 담아 반환
    struct ResidualFunc {
        StaticVector<double, Nx> x0;
        StaticVector<double, Nx> x_ref;
        StaticVector<double, Nx> Q, Qf;
        StaticVector<double, Nu> R, R_rate;
        StaticVector<double, Nu> u_last;
        DynamicBicycleModel model;
        double dt;
        std::array<RTIObstacle, 10> obs;

        template <typename T>
        StaticVector<T, N_res> operator()(const StaticVector<T, N_vars>& U) const {
            StaticVector<T, N_res> r;
            r.set_zero();
            int idx = 0;

            StaticVector<T, Nx> x;
            for (size_t i = 0; i < Nx; ++i) x(static_cast<int>(i)) = T(x0(static_cast<int>(i)));

            StaticVector<T, Nu> u_prev;
            u_prev(0) = T(u_last(0));
            u_prev(1) = T(u_last(1));

            for (size_t k = 0; k < Np; ++k) {
                StaticVector<T, Nu> u;
                u(0) = U(static_cast<int>(k * Nu + 0));
                u(1) = U(static_cast<int>(k * Nu + 1));

                // 1. 상태 잔차: J^T J 과정에서 제곱되므로 sqrt(Q)를 곱함
                for (size_t i = 0; i < Nx; ++i) {
                    r(idx++) = T(std::sqrt(Q(static_cast<int>(i)))) *
                               (x(static_cast<int>(i)) - T(x_ref(static_cast<int>(i))));
                }

                // 2. 제어 잔차
                r(idx++) = T(std::sqrt(R(0))) * u(0);
                r(idx++) = T(std::sqrt(R(1))) * u(1);

                // 3. 변화율 잔차
                r(idx++) = T(std::sqrt(R_rate(0))) * (u(0) - u_prev(0));
                r(idx++) = T(std::sqrt(R_rate(1))) * (u(1) - u_prev(1));
                u_prev = u;

                // 4. 장애물 회피 잔차
                for (size_t obs_i = 0; obs_i < 10; ++obs_i) {
                    T dx = x(0) - T(obs[obs_i].x);
                    T dy = x(1) - T(obs[obs_i].y);
                    T dist_sq = dx * dx + dy * dy;
                    T r_safe = T(obs[obs_i].r + 0.8);
                    T violation = r_safe * r_safe - dist_sq;

                    if (Optimization::get_value(violation) > 0.0) {
                        r(idx++) = T(std::sqrt(8000.0)) * violation;
                    } else {
                        r(idx++) = T(0.0);
                    }
                }

                x = Integrator::rk4<Nx, Nu, DynamicBicycleModel, T>(model, x, u, dt);
            }

            // 5. 종단 잔차
            for (size_t i = 0; i < Nx; ++i) {
                r(idx++) = T(std::sqrt(Qf(static_cast<int>(i)))) *
                           (x(static_cast<int>(i)) - T(x_ref(static_cast<int>(i))));
            }

            return r;
        }
    };

    struct DummyEq {
        template <typename T>
        StaticVector<T, 0> operator()(const StaticVector<T, N_vars>& U) const {
            (void)U;
            return StaticVector<T, 0>();
        }
    };

    struct BoundIneq {
        template <typename T>
        StaticVector<T, N_ineq> operator()(const StaticVector<T, N_vars>& U) const {
            StaticVector<T, N_ineq> ineq;
            for (size_t k = 0; k < Np; ++k) {
                T a = U(static_cast<int>(k * Nu + 0));
                T delta = U(static_cast<int>(k * Nu + 1));

                ineq(static_cast<int>(k * 4 + 0)) = a - T(3.0);
                ineq(static_cast<int>(k * 4 + 1)) = -a - T(3.0);
                ineq(static_cast<int>(k * 4 + 2)) = delta - T(0.5);
                ineq(static_cast<int>(k * 4 + 3)) = -delta - T(0.5);
            }
            return ineq;
        }
    };

    bool compute_control(const StaticVector<double, Nx>& current_x,
                         StaticVector<double, N_vars>& U_guess) {
        ResidualFunc res_f{current_x, x_ref, Q, Qf, R, R_rate, u_last, model, dt, obstacles};
        DummyEq eq_f;
        BoundIneq ineq_f;

        bool success = rti.solve(U_guess, res_f, eq_f, ineq_f);

        u_last(0) = U_guess(0);
        u_last(1) = U_guess(1);

        return success;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_RTI_NMPC_HPP_