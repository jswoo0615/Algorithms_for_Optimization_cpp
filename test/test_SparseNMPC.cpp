#include <gtest/gtest.h>

#include <chrono>

#include "Optimization/Controller/SparseNMPC.hpp"

using namespace Optimization;
using namespace Optimization::controller;

TEST(SparseNMPC_Test, ObstacleAvoidanceWithSlewRate) {
    constexpr size_t H = 20;  // 예측 구간 20스텝 (dt=0.1, 총 2.0초)
    SparseNMPC<H> nmpc;
    // 조향을 함부로 크게 꺾지 못하도록 제어 입력 페널티(R)를 대폭 강화합니다.
    nmpc.R(0) = 1.0;   // 가속도 페널티
    nmpc.R(1) = 10.0; // 조향각 페널티 (기존 10.0에서 대폭 상승)

    // 조향 변화율 페널티도 유지
    nmpc.R_rate(0) = 10.0;
    nmpc.R_rate(1) = 250.0;

    // 시나리오 셋업: 차량은 [X=0, Y=0]에서 10m/s로 직진 중
    StaticVector<double, 6> x_curr;
    x_curr.set_zero();
    x_curr(3) = 10.0;

    // 목표 궤적: 계속 직진 유지 (Y=0)
    StaticVector<double, 6> x_ref;
    x_ref.set_zero();
    x_ref(3) = 10.0;

    // [전술 발동] 1.5초(15m) 전방 한가운데에 장애물 배치
    nmpc.obstacles[0].x = 15.0;
    nmpc.obstacles[0].y = 0.1;
    nmpc.obstacles[0].r = 1.0;

    // 과거의 제어 입력 (이전 스텝에서 핸들을 꺾지 않았음)
    nmpc.u_last.set_zero();

    // RTI (SQP) 3 Iteration 수행
    auto start = std::chrono::high_resolution_clock::now();
    bool success = false;
    for (int sqp_iter = 0; sqp_iter < 3; ++sqp_iter) {
        success = nmpc.solve(x_curr, x_ref);
        EXPECT_TRUE(success);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double optimal_steer_rad = nmpc.U_guess[0](1);  // 조향각(u_1)
    double optimal_steer_deg = optimal_steer_rad * (180.0 / M_PI);

    std::cout << "\n[Obstacle Avoidance Tactics Result]" << std::endl;
    std::cout << "Obstacle Pos       : [X=15.0, Y=0.0]" << std::endl;
    std::cout << "Target Path        : Y=0.0" << std::endl;
    std::cout << "Optimal Steer (u0) : " << optimal_steer_deg << " deg" << std::endl;
    std::cout << "RTI(3 Iter) Time   : " << duration.count() / 1000.0 << " ms" << std::endl;

    // 직진 궤적을 목표로 하지만, 전방의 장애물 페널티를 피하기 위해 반드시 조향이 발생해야 함
    EXPECT_NE(std::abs(optimal_steer_rad), 0.0);
    // Slew Rate (250.0) 페널티로 인해 한 번에 무리하게(예: 30도 이상) 꺾이지 않고 부드럽게 꺾여야
    // 함
    EXPECT_LT(std::abs(optimal_steer_deg), 10.0);
    // 연산 시간은 압도적으로 빨라야 함 (통상 2ms 이내)
    EXPECT_LT(duration.count(), 5000);
}