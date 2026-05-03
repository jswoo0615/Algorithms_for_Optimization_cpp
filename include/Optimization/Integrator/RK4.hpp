#ifndef OPTIMIZATION_RK4_HPP_
#define OPTIMIZATION_RK4_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"

namespace Optimization {
namespace integrator {

/**
 * @brief Runge-Kutta 4th Order (RK4) 적분기
 * @details 연속 시간 모델 x_dot = f(x, u)를 이산 시간 x_{k+1}로 변환합니다.
 */
template <size_t Nx, size_t Nu, typename Functor, typename T>
StaticVector<T, Nx> step_rk4(const Functor& f, const StaticVector<T, Nx>& x,
                             const StaticVector<T, Nu>& u, double dt) {
    T dt_T = T(dt);
    T half_dt = T(dt / 2.0);
    T one_sixth = T(1.0 / 6.0);

    // k1 = f(x, u)
    StaticVector<T, Nx> k1 = f(x, u);

    // k2 = f(x + k1 * dt/2, u)
    StaticVector<T, Nx> x_k2;
    for (size_t i = 0; i < Nx; ++i)
        x_k2(static_cast<int>(i)) = x(static_cast<int>(i)) + k1(static_cast<int>(i)) * half_dt;
    StaticVector<T, Nx> k2 = f(x_k2, u);

    // k3 = f(x + k2 * dt/2, u)
    StaticVector<T, Nx> x_k3;
    for (size_t i = 0; i < Nx; ++i)
        x_k3(static_cast<int>(i)) = x(static_cast<int>(i)) + k2(static_cast<int>(i)) * half_dt;
    StaticVector<T, Nx> k3 = f(x_k3, u);

    // k4 = f(x + k3 * dt, u)
    StaticVector<T, Nx> x_k4;
    for (size_t i = 0; i < Nx; ++i)
        x_k4(static_cast<int>(i)) = x(static_cast<int>(i)) + k3(static_cast<int>(i)) * dt_T;
    StaticVector<T, Nx> k4 = f(x_k4, u);

    // x_{next} = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    StaticVector<T, Nx> x_next;
    for (size_t i = 0; i < Nx; ++i) {
        x_next(static_cast<int>(i)) =
            x(static_cast<int>(i)) +
            one_sixth * dt_T *
                (k1(static_cast<int>(i)) + T(2.0) * k2(static_cast<int>(i)) +
                 T(2.0) * k3(static_cast<int>(i)) + k4(static_cast<int>(i)));
    }

    return x_next;
}

}  // namespace integrator
}  // namespace Optimization

#endif  // OPTIMIZATION_RK4_HPP_