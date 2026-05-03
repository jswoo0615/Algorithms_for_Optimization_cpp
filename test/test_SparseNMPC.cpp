#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::controller;

TEST(SparseNMPC_Test, ObstacleAvoidanceSimulation) {
    constexpr size_t H = 30;  // 3초 시야
    SparseNMPC<H> nmpc;

    // [Architect's Tuning] 가중치 재설정
    nmpc.Q.set_zero();
    nmpc.Q(1) = 20.0;   // Y (차선 유지 강제)
    nmpc.Q(2) = 500.0;  // Yaw (직진 방향 정렬)
    nmpc.Q(3) = 10.0;   // Vx (정속 주행)

    for (size_t i = 0; i < 6; ++i) nmpc.Qf(i) = nmpc.Q(i) * 10.0;

    nmpc.R(1) = 30.0;        // 조향을 더 무겁게
    nmpc.R_rate(1) = 800.0;  // 급조향 절대 금지

    StaticVector<double, 6> x_curr;
    x_curr.set_zero();
    x_curr(3) = 10.0;  // 10m/s

    // 장애물: 15m 앞, Y=0.2 (중앙에서 살짝 왼쪽)
    nmpc.global_obstacles[0].x = 25.0;
    nmpc.global_obstacles[0].y = 0.2;
    nmpc.global_obstacles[0].r = 1.0;

    vehicle::DynamicBicycleModel plant;
    double dt = 0.1;

    std::string csv_filename = "nmpc_straight_line_log.csv";
    std::ofstream log_file(csv_filename);
    if (log_file.is_open()) {
        log_file << "Step,X(m),Y(m),Yaw(deg),Vx(m/s),Steer(deg),Time(ms),MaxKKT,SQPIter\n";
    }

    std::cout << "\n================ [ NMPC Straight Corridor Tracking ] ================\n";
    std::cout << "Step |   X(m)  |   Y(m)  | Yaw(deg) | Vx(m/s) | Steer(deg) | KKT Err | Iter\n";
    std::cout << "--------------------------------------------------------------------------\n";

    for (int step = 0; step < 80; ++step) {
        auto start = std::chrono::high_resolution_clock::now();

        // [Architect's Fix]
        // 목표는 무조건 '글로벌 X축 500m 앞 지점'입니다.
        // 차가 옆으로 가든 돌든, 목표는 변하지 않는 깃발이어야 합니다.
        // Fix: To prevent artificial cross-track error from longitudinal distance in local frame,
        // we set target_x to the car's current X so that the local Y reference correctly reflects
        // cross-track error.
        double target_x = x_curr(0);
        double target_y = 0.0;
        double target_theta = 0.0;
        double target_vx = 10.0;

        StaticVector<double, 6> x_ref;
        x_ref.set_zero();
        x_ref(0) = target_x;      // X: 시간에 비례하여 전진
        x_ref(1) = target_y;      // Y: 중앙선 유지
        x_ref(2) = target_theta;  // Yaw: 직진 유지
        x_ref(3) = target_vx;     // Vx: 목표 속도 유지

        NMPCResult res = nmpc.solve(x_curr, x_ref, 1);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        StaticVector<double, 2> u_opt = nmpc.U_guess[0];
        double steer_deg = u_opt(1) * (180.0 / M_PI);
        double time_ms = duration.count() / 1000.0;

        std::cout << std::setw(4) << step << " | " << std::fixed << std::setprecision(2)
                  << std::setw(7) << x_curr(0) << " | " << std::setw(7) << x_curr(1) << " | "
                  << std::setw(8) << x_curr(2) * (180 / M_PI) << " | " << std::setw(7) << x_curr(3)
                  << " | " << std::setw(10) << steer_deg << " | " << std::setprecision(4)
                  << std::setw(7) << res.max_kkt_error << " | " << std::setw(4)
                  << res.sqp_iterations << "\n";

        if (log_file.is_open()) {
            log_file << step << "," << x_curr(0) << "," << x_curr(1) << ","
                     << x_curr(2) * (180 / M_PI) << "," << x_curr(3) << "," << steer_deg << ","
                     << time_ms << "," << res.max_kkt_error << "," << res.sqp_iterations << "\n";
        }

        x_curr = integrator::step_rk4<6, 2>(plant, x_curr, u_opt, dt);
        nmpc.shift_sequence();

        // 장애물 통과 시 업데이트 로직 (유지)
        if (nmpc.global_obstacles[0].x < x_curr(0) - 2.0) {
            nmpc.global_obstacles[0].x = 10000.0;
        }
    }
    if (log_file.is_open()) log_file.close();
}