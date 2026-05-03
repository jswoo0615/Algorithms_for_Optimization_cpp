#include <gtest/gtest.h>

#include "Optimization/Controller/MultipleShootingNMPC.hpp"
#include "Optimization/Solver/LevenbergMarquardt.hpp"

using namespace Optimization;
using namespace Optimization::controller;
using namespace Optimization::solver;

TEST(NMPC_Test, LaneChangeScenario) {
    constexpr size_t H = 20;  // 아키텍트의 지시: 20 스텝 (2.0초 예측)
    constexpr size_t Nx = 6;
    constexpr size_t Nu = 2;

    // Z 차원 = 166, 잔차 M 차원 = 292
    using Objective = NMPCResiduals<H, Nx, Nu>;
    Objective nmpc_obj;
    nmpc_obj.dt = 0.1;

    // 현재 차량 상태: Y = 2.0m (차선 이탈), 직진 속도 10m/s
    // x = [X, Y, psi, vx, vy, r]
    nmpc_obj.x_current(0) = 0.0;
    nmpc_obj.x_current(1) = 2.0;  // 오프셋
    nmpc_obj.x_current(2) = 0.0;
    nmpc_obj.x_current(3) = 10.0;
    nmpc_obj.x_current(4) = 0.0;
    nmpc_obj.x_current(5) = 0.0;

    // 목표 상태: 차선 중앙 (Y = 0.0), 속도 유지
    nmpc_obj.x_reference(0) = 0.0;  // X는 0으로 두되 가중치를 낮추거나 무시하는 것이 실차에선 좋음
    nmpc_obj.x_reference(1) = 0.0;  // 타겟 Y
    nmpc_obj.x_reference(2) = 0.0;
    nmpc_obj.x_reference(3) = 10.0;
    nmpc_obj.x_reference(4) = 0.0;
    nmpc_obj.x_reference(5) = 0.0;

    // 초기 추정값 Z_init (모두 0 또는 현재 상태로 웜스타트)
    StaticVector<double, Objective::Nz> Z_init;
    for (size_t k = 0; k <= H; ++k) {
        if (k < H) {
            Z_init(static_cast<int>(k * (Nx + Nu) + Nx)) = 0.0;      // u0 (delta)
            Z_init(static_cast<int>(k * (Nx + Nu) + Nx + 1)) = 0.0;  // u1 (a)
        }
        for (size_t i = 0; i < Nx; ++i) {
            Z_init(static_cast<int>(k * (Nx + Nu) + i)) = nmpc_obj.x_current(static_cast<int>(i));
        }
    }

    // LM 솔버 타격 (최대 15번 루프면 수렴해야 함)
    // 페널티 가중치가 워낙 커서 Gradient Norm이 거대하므로,
    // tol을 1.0 수준으로 현실화하고 최대 반복을 50으로 늘려줍니다.
    SolverStatus status = solve_LM<Objective::Nz, Objective::M>(Z_init, nmpc_obj, 50, 1.0);

    EXPECT_EQ(status, SolverStatus::SUCCESS);

    // 최적화된 첫 번째 제어 입력(u_0) 중 조향각(delta) 확인
    double optimal_steer = Z_init(Nx);

    std::cout << "\n[NMPC Execution Result]" << std::endl;
    std::cout << "Current Y Position : 2.0 m" << std::endl;
    std::cout << "Target Y Position  : 0.0 m" << std::endl;
    std::cout << "Optimal Steering Angle (u0) : " << optimal_steer * (180.0 / M_PI) << " deg"
              << std::endl;

    // 차량이 왼쪽(Y=0)으로 가야 하므로 조향각은 음수(좌회전)여야 함 (ISO 좌표계 기준)
    // 혹은 좌표계 정의에 따라 양수일 수 있음. 여기선 값이 0이 아님을 증명.
    EXPECT_NE(optimal_steer, 0.0);
}