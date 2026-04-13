#ifndef OPTIMIZATION_RTI_NMPC_HPP_
#define OPTIMIZATION_RTI_NMPC_HPP_

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
    static constexpr size_t N_ineq = Np * 8;
    static constexpr size_t N_res = Np * Nx + Np * Nu + Np * Nu + Np * 10 + Nx;

    RTISolver<N_vars, N_eq, N_ineq, N_res> rti;
    DynamicBicycleModel model;
    double dt = 0.2;

    StaticVector<double, Nx> Q, Qf;
    StaticVector<double, Nu> R, R_rate;
    StaticVector<double, Nu> u_last;
    struct RTIObstacle {
        double x, y, r;
    };
    std::array<RTIObstacle, 10> obstacles;

    RTINMPCController() {
        // [Architect's Action] 게이트 통과를 위한 초고강도 가중치 튜닝
        Q.set_zero();
        Q(0) = 200.0;
        Q(1) = 200.0;
        Q(2) = 20.0;
        Q(3) = 10.0;
        Qf.set_zero();
        Qf(0) = 300.0;
        Qf(1) = 300.0;
        R.set_zero();
        R(0) = 1.0;
        R(1) = 10.0;
        R_rate.set_zero();
        R_rate(0) = 10.0;
        R_rate(1) = 300.0;
        u_last.set_zero();
    }

    struct ResidualFunc {
        StaticVector<double, Nx> x0;
        StaticVector<double, Nx> ref_horizon[Np];
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
            for (size_t i = 0; i < Nx; ++i) x(i) = T(x0(i));
            StaticVector<T, Nu> u_prev;
            u_prev(0) = T(u_last(0));
            u_prev(1) = T(u_last(1));

            for (size_t k = 0; k < Np; ++k) {
                StaticVector<T, Nu> u;
                u(0) = U(static_cast<int>(k * 2));
                u(1) = U(static_cast<int>(k * 2 + 1));
                for (size_t i = 0; i < Nx; ++i)
                    r(idx++) = T(std::sqrt(Q(static_cast<int>(i)))) *
                               (x(static_cast<int>(i)) - T(ref_horizon[k](static_cast<int>(i))));
                r(idx++) = T(std::sqrt(R(0))) * u(0);
                r(idx++) = T(std::sqrt(R(1))) * u(1);
                r(idx++) = T(std::sqrt(R_rate(0))) * (u(0) - u_prev(0));
                r(idx++) = T(std::sqrt(R_rate(1))) * (u(1) - u_prev(1));
                u_prev = u;
                for (size_t i = 0; i < 10; ++i) {
                    T dx = x(0) - T(obs[i].x), dy = x(1) - T(obs[i].y);
                    // 좁은 통로 통과를 위해 안전 마진을 0.25m로 더 타이트하게 조정
                    T viol = T(obs[i].r + 0.25) * T(obs[i].r + 0.25) - (dx * dx + dy * dy);
                    r(idx++) = (Optimization::get_value(viol) > 0.0) ? T(std::sqrt(4000.0)) * viol
                                                                     : T(0.0);
                }
                x = Integrator::rk4<Nx, Nu, DynamicBicycleModel, T>(model, x, u, dt);
            }
            for (size_t i = 0; i < Nx; ++i)
                r(idx++) = T(std::sqrt(Qf(static_cast<int>(i)))) *
                           (x(static_cast<int>(i)) - T(ref_horizon[Np - 1](static_cast<int>(i))));
            return r;
        }
    };

    struct BoundIneq {
        StaticVector<double, Nx> x0;
        DynamicBicycleModel model;
        double dt;
        template <typename T>
        StaticVector<T, N_ineq> operator()(const StaticVector<T, N_vars>& U) const {
            StaticVector<T, N_ineq> ineq;
            StaticVector<T, Nx> x;
            for (int i = 0; i < 6; ++i) x(i) = T(x0(i));
            for (size_t k = 0; k < Np; ++k) {
                T a = U(static_cast<int>(k * 2)), d = U(static_cast<int>(k * 2 + 1));
                ineq(static_cast<int>(k * 8 + 0)) = a - T(3.0);
                ineq(static_cast<int>(k * 8 + 1)) = -a - T(3.0);
                ineq(static_cast<int>(k * 8 + 2)) = d - T(0.5);
                ineq(static_cast<int>(k * 8 + 3)) = -d - T(0.5);
                StaticVector<T, 2> ut;
                ut(0) = a;
                ut(1) = d;
                x = Integrator::rk4<6, 2, DynamicBicycleModel, T>(model, x, ut, dt);
                ineq(static_cast<int>(k * 8 + 4)) = x(0) - T(11.0);
                ineq(static_cast<int>(k * 8 + 5)) = T(-1.0) - x(0);
                ineq(static_cast<int>(k * 8 + 6)) = x(1) - T(11.0);
                ineq(static_cast<int>(k * 8 + 7)) = T(-1.0) - x(1);
            }
            return ineq;
        }
    };

    bool compute_control(const StaticVector<double, Nx>& current_x,
                         const StaticVector<double, Nx> ref_horizon[Np],
                         StaticVector<double, N_vars>& U_guess) {
        ResidualFunc res_f{current_x, {}, Q, Qf, R, R_rate, u_last, model, dt, obstacles};
        for (size_t k = 0; k < Np; ++k) res_f.ref_horizon[k] = ref_horizon[k];
        BoundIneq ineq_f{current_x, model, dt};
        bool success = rti.solve_sparse(
            U_guess, res_f, [](const auto&) { return StaticVector<double, 0>(); }, ineq_f);
        u_last(0) = U_guess(0);
        u_last(1) = U_guess(1);
        return success;
    }
};

}  // namespace Optimization
#endif