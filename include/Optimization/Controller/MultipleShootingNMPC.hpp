#ifndef OPTIMIZATION_MULTIPLE_SHOOTING_NMPC_HPP_
#define OPTIMIZATION_MULTIPLE_SHOOTING_NMPC_HPP_

#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

namespace Optimization {
namespace controller {

/**
 * @brief Multiple Shooting 기반 NMPC 잔차(Residuals) 생성기
 * @tparam H 예측 구간 (Prediction Horizon)
 * @tparam Nx 상태 변수 차원 (Bicycle = 6)
 * @tparam Nu 입력 변수 차원 (Bicycle = 2)
 */
template <size_t H, size_t Nx = 6, size_t Nu = 2>
struct NMPCResiduals {
    // Z 벡터 차원: x가 H+1개, u가 H개
    static constexpr size_t Nz = Nx * (H + 1) + Nu * H;

    // 잔차 벡터 차원 M:
    // 1. 초기 상태 구속 (Nx)
    // 2. 동역학 제약 (H * Nx)
    // 3. 상태 추종 오차 ((H+1) * Nx)
    // 4. 입력 최소화/추종 (H * Nu)
    static constexpr size_t M = Nx + (H * Nx) + ((H + 1) * Nx) + (H * Nu);

    StaticVector<double, Nx> x_current;    // 현재 차량의 실제 상태 (Initial Condition)
    StaticVector<double, Nx> x_reference;  // 목표 상태 (일단 고정 궤적으로 단순화)
    double dt;

    // 가중치 (Penalty Weights)
    double W_init = 10000.0;  // 초기 상태는 무조건 지켜야 함
    double W_dyn = 5000.0;    // 물리 법칙을 어기면 심각한 페널티
    double W_state = 10.0;    // 목표 궤적 추종 가중치 (Q)
    double W_control = 1.0;   // 입력 최소화 가중치 (R)

    template <typename T>
    StaticVector<T, M> operator()(const StaticVector<T, Nz>& Z) const {
        StaticVector<T, M> residuals;
        size_t res_idx = 0;

        vehicle::DynamicBicycleModel model;

        // --- 헬퍼 람다: Z 벡터에서 x_k 와 u_k 를 0-allocation으로 추출 ---
        auto get_x = [&](size_t k) {
            StaticVector<T, Nx> x;
            for (size_t i = 0; i < Nx; ++i)
                x(static_cast<int>(i)) = Z(static_cast<int>(k * (Nx + Nu) + i));
            return x;
        };
        auto get_u = [&](size_t k) {
            StaticVector<T, Nu> u;
            for (size_t i = 0; i < Nu; ++i)
                u(static_cast<int>(i)) = Z(static_cast<int>(k * (Nx + Nu) + Nx + i));
            return u;
        };

        // 1. 초기 상태 구속 (Initial Condition: x_0 == x_current)
        StaticVector<T, Nx> x_0 = get_x(0);
        for (size_t i = 0; i < Nx; ++i) {
            residuals(static_cast<int>(res_idx++)) =
                T(W_init) * (x_0(static_cast<int>(i)) - T(x_current(static_cast<int>(i))));
        }

        // 2. 동역학 제약 (Dynamic Constraints: x_{k+1} == RK4(x_k, u_k))
        for (size_t k = 0; k < H; ++k) {
            StaticVector<T, Nx> x_k = get_x(k);
            StaticVector<T, Nu> u_k = get_u(k);
            StaticVector<T, Nx> x_next = get_x(k + 1);

            // RK4 적분기를 통해 예측된 물리적 다음 상태
            StaticVector<T, Nx> x_pred = integrator::step_rk4<Nx, Nu>(model, x_k, u_k, dt);

            for (size_t i = 0; i < Nx; ++i) {
                // 예측된 상태와 최적화 변수 x_{k+1}의 차이가 0이 되어야 함
                residuals(static_cast<int>(res_idx++)) =
                    T(W_dyn) * (x_next(static_cast<int>(i)) - x_pred(static_cast<int>(i)));
            }
        }

        // 3. 상태 추종 오차 (State Tracking Cost)
        for (size_t k = 0; k <= H; ++k) {
            StaticVector<T, Nx> x_k = get_x(k);
            for (size_t i = 0; i < Nx; ++i) {
                residuals(static_cast<int>(res_idx++)) =
                    T(W_state) * (x_k(static_cast<int>(i)) - T(x_reference(static_cast<int>(i))));
            }
        }

        // 4. 제어 입력 최소화 (Control Minimization)
        for (size_t k = 0; k < H; ++k) {
            StaticVector<T, Nu> u_k = get_u(k);
            for (size_t i = 0; i < Nu; ++i) {
                residuals(static_cast<int>(res_idx++)) = T(W_control) * u_k(static_cast<int>(i));
            }
        }

        return residuals;
    }
};

}  // namespace controller
}  // namespace Optimization

#endif  // OPTIMIZATION_MULTIPLE_SHOOTING_NMPC_HPP_