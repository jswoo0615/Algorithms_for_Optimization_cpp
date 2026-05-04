#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::controller;

TEST(SparseNMPC_Test, RTQP_DynamicTuningSimulation) {
    constexpr size_t H = 30;
    SparseNMPC<H> nmpc;

    StaticVector<double, 6> x_curr;
    x_curr.set_zero();
    x_curr(3) = 10.0;

    nmpc.global_obstacles[0].x = 25.0;
    nmpc.global_obstacles[0].y = 0.2;
    nmpc.global_obstacles[0].r = 1.0;

    vehicle::DynamicBicycleModel plant;
    double dt = 0.1;

    std::string csv_filename = "nmpc_rt_qp_log.csv";
    std::ofstream log_file(csv_filename);
    if (log_file.is_open()) {
        log_file << "Step,X(m),Y(m),Yaw(deg),Vx(m/s),Steer(deg),Time(ms),MaxKKT\n";
    }

    std::cout << "\n================ [ RT-QP & Dynamic Tuning Engine ] ================\n";
    std::cout << "Step |   X(m)  |   Y(m)  | Yaw(deg) | Vx(m/s) | Steer(deg) | KKT Err\n";
    std::cout << "----------------------------------------------------------------------\n";

    for (int step = 0; step < 80; ++step) {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. [실시간 타겟 생성]
        StaticVector<double, 6> x_ref;
        x_ref.set_zero();
        x_ref(0) = x_curr(0);  // 종방향 오차 투영 방지
        x_ref(1) = 0.0;
        x_ref(2) = 0.0;
        x_ref(3) = 10.0;

        // 2. [Gain Scheduling] 상황에 따른 계수 실시간 튜닝
        NMPCTuningConfig config;
        double vx = std::max(0.1, x_curr(3));
        double y_err_abs = std::abs(x_curr(1));

        // [Architect's Fix: 미친 가중치 롤백]
        // R_Steer를 5000에서 10~50 수준으로 대폭 낮춰야 동역학이 숨을 쉽니다.
        config.R_Steer = 10.0 + 0.5 * (vx * vx);
        config.R_Steer_Rate = 50.0 + 1.0 * (vx * vx);  // 급조향 방지 정도만 유지

        // 차선 유지력도 너무 크면 진동을 유발하므로 현실적으로 조절
        config.Q_Y = 50.0;
        if (y_err_abs > 0.5) {
            config.Q_Y = 100.0;
        }

        // 3. [RT-QP 단일 풀이]
        NMPCResult res = nmpc.solve_rt_qp(x_curr, x_ref, config);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        StaticVector<double, 2> u_opt = nmpc.U_guess[0];
        double steer_deg = u_opt(1) * (180.0 / M_PI);
        double time_ms = duration.count() / 1000.0;

        std::cout << std::setw(4) << step << " | " << std::fixed << std::setprecision(2)
                  << std::setw(7) << x_curr(0) << " | " << std::setw(7) << x_curr(1) << " | "
                  << std::setw(8) << x_curr(2) * (180 / M_PI) << " | " << std::setw(7) << x_curr(3)
                  << " | " << std::setw(10) << steer_deg << " | " << std::setprecision(4)
                  << std::setw(7) << res.max_kkt_error << "\n";

        if (log_file.is_open()) {
            log_file << step << "," << x_curr(0) << "," << x_curr(1) << ","
                     << x_curr(2) * (180 / M_PI) << "," << x_curr(3) << "," << steer_deg << ","
                     << time_ms << "," << res.max_kkt_error << "\n";
        }

        x_curr = integrator::step_rk4<6, 2>(plant, x_curr, u_opt, dt);
        nmpc.shift_sequence();

        if (nmpc.global_obstacles[0].x < x_curr(0) - 2.0) {
            nmpc.global_obstacles[0].x = 10000.0;
        }
    }
    if (log_file.is_open()) log_file.close();
}