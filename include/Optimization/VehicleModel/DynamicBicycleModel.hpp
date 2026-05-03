#ifndef OPTIMIZATION_DYNAMIC_BICYCLE_MODEL_HPP_
#define OPTIMIZATION_DYNAMIC_BICYCLE_MODEL_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"

namespace Optimization {
namespace vehicle {

/**
 * @brief Dynamic Bicycle Model (6-State, 2-Control)
 * @details
 * State x: [X, Y, psi(요각), v_x(종방향 속도), v_y(횡방향 속도), r(요레이트)]^T
 * Control u: [delta(조향각), a(가속도)]^T
 */
struct DynamicBicycleModel {
    // 차량 파라미터 (임시값, 실차 데이터로 교체 가능)
    double m = 1500.0;     // 질량 (kg)
    double Iz = 3000.0;    // 요 관성 모멘트 (kg*m^2)
    double Lf = 1.2;       // 무게중심 - 전륜 거리 (m)
    double Lr = 1.6;       // 무게중심 - 후륜 거리 (m)
    double Cf = 100000.0;  // 전륜 코너링 강성 (N/rad)
    double Cr = 100000.0;  // 후륜 코너링 강성 (N/rad)

    // Functor 오버로딩 (T는 double 또는 DualVec이 들어옴)
    template <typename T>
    StaticVector<T, 6> operator()(const StaticVector<T, 6>& x, const StaticVector<T, 2>& u) const {
        using namespace ad;  // ad::sin, ad::cos, ad::atan2 등을 사용하기 위함

        // State 추출
        T psi = x(2);
        T vx = x(3);
        T vy = x(4);
        T r = x(5);

        // Control 추출
        T delta = u(0);
        T a = u(1);

        // 특이점 방어 (속도가 0에 가까울 때 Slip angle 발산 방지)
        double vx_val = get_value(vx);  // T가 double 이든 DualVec이든 순수 값만 추출
        T vx_safe = (vx_val >= 0.1) ? vx : ((vx_val <= -0.1) ? vx : T(0.1));

        // 타이어 슬립각 (Slip Angle) 계산
        T alpha_f = delta - atan2(vy + T(Lf) * r, vx_safe);
        T alpha_r = -atan2(vy - T(Lr) * r, vx_safe);

        // 선형 타이어 모델 기반 횡력 (Lateral Force)
        T Fyf = T(Cf) * alpha_f;
        T Fyr = T(Cr) * alpha_r;

        // 상태 변화율 (State Derivatives) dx/dt = f(x, u)
        StaticVector<T, 6> x_dot;

        x_dot(0) = vx * cos(psi) - vy * sin(psi);                     // X_dot
        x_dot(1) = vx * sin(psi) + vy * cos(psi);                     // Y_dot
        x_dot(2) = r;                                                 // psi_dot
        x_dot(3) = a + r * vy;                                        // vx_dot
        x_dot(4) = (Fyf * cos(delta) + Fyr) / T(m) - r * vx;          // vy_dot
        x_dot(5) = (T(Lf) * Fyf * cos(delta) - T(Lr) * Fyr) / T(Iz);  // r_dot

        return x_dot;
    }
};

}  // namespace vehicle
}  // namespace Optimization

#endif  // OPTIMIZATION_DYNAMIC_BICYCLE_MODEL_HPP_