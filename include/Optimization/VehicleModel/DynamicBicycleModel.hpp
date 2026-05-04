#ifndef OPTIMIZATION_DYNAMIC_BICYCLE_MODEL_HPP_
#define OPTIMIZATION_DYNAMIC_BICYCLE_MODEL_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"

namespace Optimization {
namespace vehicle {

/**
 * @brief Frenet Dynamic Bicycle Model (6-State, 2-Control)
 * @details
 * 곡률(kappa)이 반영된 S-D 좌표계 기반의 차량 동역학 모델.
 * State x: [S(종방향 거리), D(횡방향 오차), mu(헤딩 오차), v_x(종방향 속도), v_y(횡방향 속도), r(요레이트)]^T
 * Control u: [delta(조향각), a(가속도)]^T
 */
struct DynamicBicycleModel {
    // 차량 파라미터 
    double m = 1500.0;     
    double Iz = 3000.0;    
    double Lf = 1.2;       
    double Lr = 1.6;       
    double Cf = 100000.0;  
    double Cr = 100000.0;  

    // [Architect's Add] 현재 노드의 도로 곡률 (Curvature)
    // NMPC 솔버가 RK4 적분기를 호출하기 전에 해당 지점의 곡률을 업데이트합니다.
    double kappa = 0.0;    

    template <typename T>
    StaticVector<T, 6> operator()(const StaticVector<T, 6>& x, const StaticVector<T, 2>& u) const {
        // ADL 충돌 차단 수학 라우터
        auto math_sin = [](const T& v) -> T {
            if constexpr (std::is_same_v<T, double>)
                return std::sin(v);
            else {
                using namespace ad;
                return sin(v);
            }
        };
        auto math_cos = [](const T& v) -> T {
            if constexpr (std::is_same_v<T, double>)
                return std::cos(v);
            else {
                using namespace ad;
                return cos(v);
            }
        };
        auto math_atan2 = [](const T& y, const T& x_val) -> T {
            if constexpr (std::is_same_v<T, double>)
                return std::atan2(y, x_val);
            else {
                using namespace ad;
                return atan2(y, x_val);
            }
        };

        // State 추출 (Frenet)
        T s = x(0);      // 사실 S값 자체는 S_dot에 영향을 주지 않음
        T d = x(1);
        T mu = x(2);
        T vx = x(3);
        T vy = x(4);
        T r = x(5);

        // Control 추출
        T delta = u(0);
        T a = u(1);

        // 특이점 방어 (Slip angle 발산 방지)
        double vx_val = get_value(vx);  
        T vx_safe = (vx_val >= 0.1) ? vx : ((vx_val <= -0.1) ? vx : T(0.1));

        // 타이어 슬립각 & 횡력 연산 (차량 로컬 동역학은 기존과 완벽히 동일)
        T alpha_f = delta - math_atan2(vy + T(Lf) * r, vx_safe);
        T alpha_r = -math_atan2(vy - T(Lr) * r, vx_safe);

        T Fyf = T(Cf) * alpha_f;
        T Fyr = T(Cr) * alpha_r;

        // [Architect's Masterpiece: Frenet Kinematics]
        // 1 - d * kappa 분모가 0이 되어 발산(Singularity)하는 것을 막기 위한 안전장치
        T denom = T(1.0) - d * T(kappa);
        double denom_val = get_value(denom);
        T denom_safe = (denom_val >= 0.05) ? denom : T(0.05);

        T s_dot = (vx * math_cos(mu) - vy * math_sin(mu)) / denom_safe;

        StaticVector<T, 6> x_dot;

        // 위치 방정식 (Frenet 변환 적용)
        x_dot(0) = s_dot;                                      // S_dot
        x_dot(1) = vx * math_sin(mu) + vy * math_cos(mu);      // D_dot
        x_dot(2) = r - T(kappa) * s_dot;                       // mu_dot

        // 동역학 방정식 (기존 유지)
        x_dot(3) = a + r * vy;                                 // vx_dot
        x_dot(4) = (Fyf * math_cos(delta) + Fyr) / T(m) - r * vx; // vy_dot
        x_dot(5) = (T(Lf) * Fyf * math_cos(delta) - T(Lr) * Fyr) / T(Iz); // r_dot

        return x_dot;
    }
};

}  // namespace vehicle
}  // namespace Optimization

#endif  // OPTIMIZATION_DYNAMIC_BICYCLE_MODEL_HPP_