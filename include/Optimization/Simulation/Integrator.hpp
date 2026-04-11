#ifndef OPTIMIZATION_INTEGRATOR_HPP_
#define OPTIMIZATION_INTEGRATOR_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @brief Layer 3 - RK4 (Runge-Kutta 4th Order) Integrator
 * @details 연속 시간 비선형 모델을 이산 시간으로 변환. AD 엔진(Dual)과 완벽 호환.
 */
class Integrator {
public:
    template <size_t Nx, size_t Nu, typename Model, typename T>
    static StaticVector<T, Nx> rk4(const Model& model, 
                                   const StaticVector<T, Nx>& x, 
                                   const StaticVector<T, Nu>& u, 
                                   double dt) {
        // 스칼라 상수들의 타입 승격 (double -> T)
        T dt_T = T(dt);
        T half_dt = T(dt / 2.0);
        T sixth_dt = T(dt / 6.0);
        T two = T(2.0);

        // k1 = f(x, u)
        StaticVector<T, Nx> k1 = model(x, u);
        
        // k2 = f(x + k1 * dt/2, u)
        StaticVector<T, Nx> x2;
        for(size_t i = 0; i < Nx; ++i) {
            x2(static_cast<int>(i)) = x(static_cast<int>(i)) + k1(static_cast<int>(i)) * half_dt;
        }
        StaticVector<T, Nx> k2 = model(x2, u);
        
        // k3 = f(x + k2 * dt/2, u)
        StaticVector<T, Nx> x3;
        for(size_t i = 0; i < Nx; ++i) {
            x3(static_cast<int>(i)) = x(static_cast<int>(i)) + k2(static_cast<int>(i)) * half_dt;
        }
        StaticVector<T, Nx> k3 = model(x3, u);
        
        // k4 = f(x + k3 * dt, u)
        StaticVector<T, Nx> x4;
        for(size_t i = 0; i < Nx; ++i) {
            x4(static_cast<int>(i)) = x(static_cast<int>(i)) + k3(static_cast<int>(i)) * dt_T;
        }
        StaticVector<T, Nx> k4 = model(x4, u);
        
        // x_next = x + dt/6 * (k1 + 2k2 + 2k3 + k4)
        StaticVector<T, Nx> x_next;
        for(size_t i = 0; i < Nx; ++i) {
            x_next(static_cast<int>(i)) = x(static_cast<int>(i)) + sixth_dt * (k1(static_cast<int>(i)) + two * k2(static_cast<int>(i)) + two * k3(static_cast<int>(i)) + k4(static_cast<int>(i)));
        }
        
        return x_next;
    }
};

} // namespace Optimization

#endif // OPTIMIZATION_INTEGRATOR_HPP_