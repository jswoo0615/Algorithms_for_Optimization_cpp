#ifndef OPTIMIZATION_VEHICLE_MODEL_HPP_
#define OPTIMIZATION_VEHICLE_MODEL_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"
#include "Optimization/Dual.hpp"

namespace Optimization {
    /**
     * @brief 차량 물리 파라미터
     */
    struct VehicleParam {
        double m = 1500.0;              // 질량 (kg)
        double Iz = 3000.0;             // 요 관성모멘트 (kg*m^2)
        double lf = 1.2;                // 무게중심 - 앞축 거리 (m)
        double lr = 1.6;                // 무게중심 - 뒤축 거리 (m)
        double Cf = 100000.0;           // 전륜 코너링 스티프니스 (N/rad)
        double Cr = 100000.0;           // 후륜 코너링 스티프니스 (N/rad)
    };

    /**
     * @brief Layer 7 - Dynamic Bicycle Model (6-State)
     * @details 템플릿 T를 통해 double (시뮬레이션) 및 Dual (AD 자코비안 추출) 완벽 호환
     */
    class DynamicBicycleModel {
        public:
            VehicleParam param;
            DynamicBicycleModel(const VehicleParam& p = VehicleParam()) : param(p) {}

            /**
             * @brief 연속 시간 상태 방정식 : dx/dt = f(x, u)
             * @param x 상태 벡터 : [X, Y, theta, vx, vy, omega]
             * @param u 입력 벡터 : [a, delta] (종방향 가속도, 조향각)
             * @return 상태 미분 벡터 dx
             */
            template <typename T>
            StaticVector<T, 6> operator()(const StaticVector<T, 6>& x, const StaticVector<T, 2>& u) const {
                StaticVector<T, 6> dx;

                // 상태 변수 할당
                T theta = x(2);
                T vx = x(3);
                T vy = x(4);
                T omega = x(5);

                // 제어 입력 할당
                T a = u(0);
                T delta = u(1);

                // 특이점 (Singularity) 방지 : 정차 시 슬립 앵글 분모가 0이 되는 것을 방지
                T vx_safe = (ad::abs(vx) < T(0.001)) ? (vx >= T(0) ? T(0.001) : T(-0.001)) : vx;

                // 타이어 슬립 앵글 (Slip Angle)
                T alpha_f = delta - ad::atan2(vy + T(param.lf) * omega, vx_safe);
                T alpha_r = -ad::atan2(vy - T(param.lr) * omega, vx_safe);

                // 선형 타이어 횡력 (Lateral Forces)
                T Fyf = T(param.Cf) * alpha_f;
                T Fyr = T(param.Cr) * alpha_r;

                // 비선형 동역학 방정식 (Equations of Motion)
                dx(0) = vx * ad::cos(theta) - vy * ad::sin(theta);                              // X_dot
                dx(1) = vx * ad::sin(theta) + vy * ad::cos(theta);                              // Y_dot
                dx(2) = omega;                                                                  // theta_dot
                dx(3) = a + vy * omega;                                                         // vx_dot
                dy(4) = -vx * oemga + (Fyf * ad::cos(delta) + Fyr) / T(param.m);                // vy_dot
                dx(5) = (T(param.lf) * Fyf * ad::cos(delta) - T(param.lr) * Fyr) / T(param.Iz); // omega_dot

                return dx;
            }   
    };
} // namespace Optimization

#endif // OPTIMIZATION_VEHICLE_MODEL_HPP_