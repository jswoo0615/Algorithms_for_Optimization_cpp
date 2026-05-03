#include <gtest/gtest.h>

#include "Optimization/Dual.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::vehicle;
using namespace Optimization::integrator;

TEST(VehicleIntegratorTest, RK4_AutoDiff_Jacobian) {
    DynamicBicycleModel model;
    double dt = 0.1;  // 100ms

    // 입력(Control)에 대한 자코비안을 구하기 위해, u를 DualVec(2차원)으로 선언
    using ADVar = DualVec<double, 2>;

    StaticVector<ADVar, 6> x;
    // 초기 상태: X=0, Y=0, psi=0, vx=10m/s, vy=0, r=0 (직진 상태)
    x(0) = ADVar(0.0);
    x(1) = ADVar(0.0);
    x(2) = ADVar(0.0);
    x(3) = ADVar(10.0);
    x(4) = ADVar(0.0);
    x(5) = ADVar(0.0);

    StaticVector<ADVar, 2> u;
    // 독립 변수(Variable) 선언: 조향각 delta=0.1 rad, 가속도 a=1.0 m/s^2
    u(0) = ADVar::make_variable(0.1, 0);  // delta에 대한 편미분 추적
    u(1) = ADVar::make_variable(1.0, 1);  // a에 대한 편미분 추적

    // 마법의 시작: RK4 적분기에 DualVec을 그대로 통과시킨다.
    StaticVector<ADVar, 6> x_next = step_rk4<6, 2>(model, x, u, dt);

    // 검증 1: 값이 제대로 적분되었는가? (종방향 속도가 가속도 1.0 * 0.1초 = 0.1 증가했는가)
    EXPECT_NEAR(x_next(3).v, 10.1, 1e-2);

    // 검증 2: 자코비안이 추출되었는가?
    // x_next(5)는 다음 스텝의 요레이트(r).
    // x_next(5).g[0] 은 "조향각(delta)이 요레이트(r)에 미치는 영향" 즉, 입력 자코비안 B행렬의 (5,
    // 0) 성분입니다.
    std::cout << "\n[Vehicle Model AD Result]" << std::endl;
    std::cout << "Next Yaw Rate (r) = " << x_next(5).v << " rad/s" << std::endl;
    std::cout << "d(r_{k+1}) / d(delta) = " << x_next(5).g[0] << std::endl;

    // 조향을 꺾었으므로 요레이트 변화율이 양수여야 함
    EXPECT_GT(x_next(5).g[0], 0.0);
}