#ifndef OPTIMIZATION_VEHICLE_MODEL_HPP_
#define OPTIMIZATION_VEHICLE_MODEL_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"
#include "Optimization/Dual.hpp"
#include <cmath>

namespace Optimization {

// Math Bridge: T가 double일 때는 std::를, Dual일 때는 ad::를 자동 매칭
namespace math {
    using std::sin;
    using std::cos;
    using std::atan2;
    using std::abs;
    using std::tanh; // [Architect's Add] Tanh 브릿지 추가
    
    using ad::sin;
    using ad::cos;
    using ad::atan2;
    using ad::abs;
    using ad::tanh;
}

struct VehicleParam {
    double m = 1500.0;     // 질량 (kg)
    double Iz = 3000.0;    // 요 관성모멘트 (kg*m^2)
    double lf = 1.2;       // 무게중심 - 앞축 거리 (m)
    double lr = 1.6;       // 무게중심 - 뒤축 거리 (m)
    double Cf = 100000.0;  // 전륜 코너링 스티프니스 (N/rad)
    double Cr = 100000.0;  // 후륜 코너링 스티프니스 (N/rad)
};

class DynamicBicycleModel {
public:
    VehicleParam param;

    DynamicBicycleModel(const VehicleParam& p = VehicleParam()) : param(p) {}

    template <typename T>
    StaticVector<T, 6> operator()(const StaticVector<T, 6>& x, const StaticVector<T, 2>& u) const {
        StaticVector<T, 6> dx;

        T theta  = x(2);
        T vx     = x(3);
        T vy     = x(4);
        T omega  = x(5);

        T a      = u(0);
        T delta  = u(1);

        double vx_val = Optimization::get_value(vx);
        T vx_safe = (std::abs(vx_val) < 0.1) ? (vx_val >= 0.0 ? T(0.1) : T(-0.1)) : vx;

        T alpha_f = delta - math::atan2(vy + T(param.lf) * omega, vx_safe);
        T alpha_r = -math::atan2(vy - T(param.lr) * omega, vx_safe);

        // [Architect's Fix] 선형 타이어 모델 물리적 포화(Saturation) 방어 로직 추가
        // 타이어가 버틸 수 있는 최대 횡력 한계 (F_max = m * g * 0.5 수준으로 대략 가정)
        T F_max = T(param.m * 9.81 * 0.5); 
        
        // Tanh 함수를 통해 슬립 앵글이 아무리 커져도 횡력이 F_max를 초과하지 못하게 스쿼싱
        T Fyf = F_max * math::tanh((T(param.Cf) * alpha_f) / F_max);
        T Fyr = F_max * math::tanh((T(param.Cr) * alpha_r) / F_max);

        dx(0) = vx * math::cos(theta) - vy * math::sin(theta);                      
        dx(1) = vx * math::sin(theta) + vy * math::cos(theta);                      
        dx(2) = omega;                                                          
        dx(3) = a + vy * omega;                                                 
        dx(4) = -vx * omega + (Fyf * math::cos(delta) + Fyr) / T(param.m);        
        dx(5) = (T(param.lf) * Fyf * math::cos(delta) - T(param.lr) * Fyr) / T(param.Iz); 

        return dx;
    }
};

} // namespace Optimization

#endif // OPTIMIZATION_VEHICLE_MODEL_HPP_