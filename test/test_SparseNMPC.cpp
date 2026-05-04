#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>  // 노이즈 생성용
#include <string>

#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/Estimator/EKF.hpp"  // EKF 헤더 포함
#include "Optimization/Integrator/RK4.hpp"

using namespace Optimization;
using namespace Optimization::controller;
using namespace Optimization::estimator;

TEST(SparseNMPC_Test, RTQP_with_EKF_Simulation) {
    constexpr size_t H = 30;
    SparseNMPC<H> nmpc;
    EKF<6, 2> ekf;  // 추정기 인스턴스화

    // 물리 엔진의 진짜 상태 (True State)
    StaticVector<double, 6> x_true;
    x_true.set_zero();
    x_true(3) = 10.0;

    // EKF 초기화 (완벽한 상태에서 출발한다고 가정)
    ekf.x_est = x_true;

    nmpc.global_obstacles[0].x = 30.0;  // 30m 앞
    nmpc.global_obstacles[0].y = -3.0;  // 우측 차선 밖에서 시작
    nmpc.global_obstacles[0].r = 1.0;
    nmpc.global_obstacles[0].vx = 5.0;  // X 방향으로 5m/s 이동 (차량보다 느림)
    nmpc.global_obstacles[0].vy = 0.5;  // Y 방향으로 0.5m/s 이동 (차선 중앙을 향해 끼어들기)

    vehicle::DynamicBicycleModel plant;
    double dt = 0.1;

    // 노이즈 생성기 세팅 (가혹한 센서 환경 모사)
    std::mt19937 gen(42);
    std::normal_distribution<double> noise_pos(0.0, 0.15);  // X, Y: 15cm 표준편차
    std::normal_distribution<double> noise_yaw(0.0, 0.05);  // Yaw: 약 3도
    std::normal_distribution<double> noise_vel(0.0, 0.2);   // 속도: 0.2m/s

    std::string csv_filename = "nmpc_ekf_closed_loop_log.csv";
    std::ofstream log_file(csv_filename);
    if (log_file.is_open()) {
        log_file << "Step,True_X,True_Y,Meas_Y,Est_Y,Est_Yaw,Est_Vx,Steer(deg),MaxKKT\n";
    }

    std::cout << "\n================ [ Full Stack: Plant -> EKF -> RT-QP NMPC ] ================\n";
    std::cout << "Step | True Y | Meas Y |  Est Y | Est Yaw | Steer(deg) | KKT Err\n";
    std::cout << "----------------------------------------------------------------------\n";

    StaticVector<double, 2> u_applied;
    u_applied.set_zero();

    for (int step = 0; step < 80; ++step) {
        auto start = std::chrono::high_resolution_clock::now();

        nmpc.global_obstacles[0].x += nmpc.global_obstacles[0].vx * dt;
        nmpc.global_obstacles[0].y += nmpc.global_obstacles[0].vy * dt;

        // 1. [Sensor Model] 진짜 상태에 노이즈를 섞어 측정값 생성
        StaticVector<double, 6> x_meas = x_true;
        x_meas(0) += noise_pos(gen);
        x_meas(1) += noise_pos(gen);
        x_meas(2) += noise_yaw(gen);
        x_meas(3) += noise_vel(gen);
        // (편의상 측면 미끄러짐과 요레이트는 노이즈가 적다고 가정)

        // 2. [Estimator Layer] EKF 예측 및 업데이트
        ekf.predict(plant, u_applied, dt);  // 이전 스텝의 제어 입력으로 예측
        ekf.update(x_meas);                 // 오염된 센서 데이터로 보정

        // 3. [Planning Layer] 목표점은 현재 EKF 추정 X축을 기준으로 생성
        StaticVector<double, 6> x_ref;
        x_ref.set_zero();
        x_ref(0) = ekf.x_est(0);
        x_ref(1) = 0.0;
        x_ref(2) = 0.0;
        x_ref(3) = 10.0;

        // 4. [Gain Scheduling] 추정된 상태를 기반으로 튜닝
        NMPCTuningConfig config;
        double vx = std::max(0.1, ekf.x_est(3));
        double y_err_abs = std::abs(ekf.x_est(1));

        config.R_Steer = 10.0 + 0.5 * (vx * vx);
        config.R_Steer_Rate = 50.0 + 1.0 * (vx * vx);
        config.Q_Y = (y_err_abs > 0.5) ? 100.0 : 50.0;

        // 5. [Controller Layer] NMPC는 '추정된 상태(x_est)'만을 믿고 제어 입력을 산출
        NMPCResult res = nmpc.solve_rt_qp(ekf.x_est, x_ref, config);

        auto end = std::chrono::high_resolution_clock::now();

        u_applied = nmpc.U_guess[0];  // 산출된 제어 입력
        double steer_deg = u_applied(1) * (180.0 / M_PI);

        // 출력 및 로깅 (실제 Y, 노이즈 낀 Y, EKF가 걸러낸 Y를 비교)
        std::cout << std::setw(4) << step << " | " << std::fixed << std::setprecision(3)
                  << std::setw(6) << x_true(1) << " | " << std::setw(6) << x_meas(1) << " | "
                  << std::setw(6) << ekf.x_est(1) << " | " << std::setw(7)
                  << ekf.x_est(2) * (180 / M_PI) << " | " << std::setw(10) << steer_deg << " | "
                  << std::setprecision(4) << std::setw(7) << res.max_kkt_error << "\n";

        if (log_file.is_open()) {
            log_file << step << "," << x_true(0) << "," << x_true(1) << "," << x_meas(1) << ","
                     << ekf.x_est(1) << "," << ekf.x_est(2) * (180 / M_PI) << "," << ekf.x_est(3)
                     << "," << steer_deg << "," << res.max_kkt_error << "\n";
        }

        // 6. [Plant Simulation] 산출된 제어 입력을 진짜 물리 세계에 적용
        x_true = integrator::step_rk4<6, 2>(plant, x_true, u_applied, dt);
        nmpc.shift_sequence();

        if (nmpc.global_obstacles[0].x < x_true(0) - 2.0) {
            nmpc.global_obstacles[0].x = 10000.0;
        }
    }
    if (log_file.is_open()) log_file.close();
}