#ifndef OPTIMIZATION_NMPC_HPP_
#define OPTIMIZATION_NMPC_HPP_

#include <array>
#include <type_traits>

#include "Optimization/Physics/VehicleModel.hpp"
#include "Optimization/SQPSolver.hpp"
#include "Optimization/Simulation/Integrator.hpp"

namespace Optimization {

struct Obstacle {
    double x = 0.0;
    double y = 0.0;
    double r = 0.5;  // 장애물 물리적 반경
};

template <size_t Np>
class NMPCController {
   public:
    static constexpr size_t Nx = 6;
    static constexpr size_t Nu = 2;
    static constexpr size_t N_vars = Np * Nu;
    static constexpr size_t N_eq = 0;
    static constexpr size_t N_ineq = 0;  // 하드 제약 완전 배제 (메모리 폭발 방지)

    SQPSolver<N_vars, N_eq, N_ineq> sqp;
    DynamicBicycleModel model;
    double dt = 0.1;  // 100ms

    StaticVector<double, Nx> Q;
    StaticVector<double, Nx> Qf;
    StaticVector<double, Nu> R;
    StaticVector<double, Nu> R_rate;  // [Architect's Add] 제어 입력 변화율(Slew Rate) 가중치
    StaticVector<double, Nx> x_ref;

    StaticVector<double, Nu> u_last;  // 현재 차량에 인가된 마지막 물리적 제어 입력 (연속성 확보)

    std::array<Obstacle, 10> obstacles;

    NMPCController() {
        Q.set_zero();
        // 타겟 도달 시 완벽한 감속을 위해 속도(Q(3)) 페널티 대폭 상향
        Q(0) = 15.0;
        Q(1) = 15.0;
        Q(2) = 5.0;
        Q(3) = 10.0;

        Qf.set_zero();
        for (size_t i = 0; i < Nx; ++i) Qf(static_cast<int>(i)) = Q(static_cast<int>(i)) * 10.0;

        // 크기 페널티
        R.set_zero();
        R(0) = 0.5;
        R(1) = 10.0;

        // [Architect's Add] 승차감 확보 및 Bang-Bang 제어 억제를 위한 변화율(Jerk/Slew Rate) 페널티
        R_rate.set_zero();
        R_rate(0) = 50.0;  // 가속도 변화율(Jerk) 억제
        R_rate(1) =
            250.0;  // 조향각 변화율(Steering Slew Rate) 강력 억제 (물리적 회피와 승차감의 타협점)

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
            for (size_t i = 0; i < Nx; ++i) x(static_cast<int>(i)) = T(x0(static_cast<int>(i)));

            auto squash = [](T val, double limit) -> T {
                if constexpr (std::is_same_v<T, double>) {
                    return limit * std::tanh(val / limit);
                } else {
                    return T(limit) * ad::tanh(val / T(limit));
                }
            };

            // [Architect's Add] 이전 스텝의 입력을 추적하기 위한 변수 (초기값은 u_last)
            StaticVector<T, Nu> u_prev;
            u_prev(0) = T(u_last(0));
            u_prev(1) = T(u_last(1));

            for (size_t k = 0; k < Np; ++k) {
                StaticVector<T, Nu> u;
                u(0) = squash(U(static_cast<int>(k * Nu + 0)), 3.0);  // 가속도
                u(1) = squash(U(static_cast<int>(k * Nu + 1)), 0.5);  // 조향

                // 1. 상태 오차 페널티 (정차 로직 포함)
                for (size_t i = 0; i < Nx; ++i) {
                    T err = x(static_cast<int>(i)) - T(x_ref(static_cast<int>(i)));
                    cost += T(Q(static_cast<int>(i))) * err * err;
                }

                // 2. 제어 입력 크기(Magnitude) 페널티
                cost +=
                    T(R(0)) * U(static_cast<int>(k * Nu + 0)) * U(static_cast<int>(k * Nu + 0)) +
                    T(R(1)) * U(static_cast<int>(k * Nu + 1)) * U(static_cast<int>(k * Nu + 1));

                // 3. [핵심] 제어 입력 변화율(Slew Rate/Jerk) 페널티 주입
                T delta_a = u(0) - u_prev(0);
                T delta_steer = u(1) - u_prev(1);
                cost += T(R_rate(0)) * delta_a * delta_a + T(R_rate(1)) * delta_steer * delta_steer;
                u_prev = u;  // 다음 스텝(k+1)의 변화율 계산을 위해 갱신

                // 4. Soft Obstacle Penalty (장애물 회피망 구축)
                for (size_t obs_i = 0; obs_i < 10; ++obs_i) {
                    T dx = x(0) - T(obs[obs_i].x);
                    T dy = x(1) - T(obs[obs_i].y);
                    T dist_sq = dx * dx + dy * dy;

                    // 장애물 반경 + 차량 여유 반경(0.8m)
                    T r_safe = T(obs[obs_i].r + 0.8);
                    T violation = r_safe * r_safe - dist_sq;

                    // 안전 거리 안으로 침범할 경우 막대한 2차 페널티 부과
                    if (Optimization::get_value(violation) > 0.0) {
                        cost += T(8000.0) * violation * violation;
                    }
                }

                // 5. 플랜트 롤아웃
                x = Integrator::rk4<Nx, Nu, DynamicBicycleModel, T>(model, x, u, dt);
            }

            // 6. Terminal Cost
            for (size_t i = 0; i < Nx; ++i) {
                T err = x(static_cast<int>(i)) - T(x_ref(static_cast<int>(i)));
                cost += T(Qf(static_cast<int>(i))) * err * err;
            }

            return cost;
        }
    };

    struct DummyEq {
        template <typename T>
        StaticVector<T, 0> operator()(const StaticVector<T, N_vars>& U) const {
            (void)U;
            return StaticVector<T, 0>();
        }
    };
    struct DummyIneq {
        template <typename T>
        StaticVector<T, 0> operator()(const StaticVector<T, N_vars>& U) const {
            (void)U;
            return StaticVector<T, 0>();
        }
    };

    bool compute_control(const StaticVector<double, Nx>& current_x,
                         StaticVector<double, N_vars>& U_guess) {
        CostFunc cost_f{current_x, x_ref, Q, Qf, R, R_rate, u_last, model, dt, obstacles};
        DummyEq eq_f;
        DummyIneq ineq_f;

        bool success = sqp.solve(U_guess, cost_f, eq_f, ineq_f, 15);

        // 다음 compute_control 호출 시 시작 변화율(u - u_last)을 계산하기 위해
        // 최적화가 완료된 후 현재 차량에 인가될 '물리적으로 스쿼싱된 제어값'을 저장합니다.
        u_last(0) = 3.0 * std::tanh(U_guess(0) / 3.0);
        u_last(1) = 0.5 * std::tanh(U_guess(1) / 0.5);

        return success;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_NMPC_HPP_