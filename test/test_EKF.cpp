#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "Optimization/Estimator/EKF.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::estimator;

TEST(EKF_Filter_Test, SensorNoiseFiltering) {
    constexpr size_t Nx = 6;
    constexpr size_t Nu = 2;
    double dt = 0.1;

    // 1. 시스템 초기화
    EKF<Nx, Nu> ekf;
    vehicle::DynamicBicycleModel plant;

    // 실제 차량의 진짜 상태 (True State)
    StaticVector<double, Nx> x_true;
    x_true.set_zero();
    x_true(3) = 10.0;  // 초기 속도 10m/s

    // EKF의 초기 추정치 (일부러 오차를 주어 수렴성 테스트)
    ekf.x_est = x_true;
    ekf.x_est(0) -= 2.0;  // S 위치 2m 오차
    ekf.x_est(1) += 1.0;  // D 위치 1m 오차

    // 2. 가우시안 노이즈 생성기 (센서 시뮬레이션)
    std::random_device rd;
    std::mt19937 gen(rd());

    // EKF의 R 행렬과 유사한 수준의 실제 센서 노이즈 (표준편차)
    std::normal_distribution<double> noise_s(0.0, 0.4);   // GPS S오차
    std::normal_distribution<double> noise_d(0.0, 0.4);   // GPS D오차
    std::normal_distribution<double> noise_mu(0.0, 0.1);  // IMU Yaw 오차
    std::normal_distribution<double> noise_v(0.0, 0.2);   // 휠스피드 오차
    std::normal_distribution<double> noise_r(0.0, 0.05);  // Yaw rate 오차

    // 3. 로깅 준비
    std::ofstream csv_file("ekf_filtering_test.csv");
    if (csv_file.is_open()) {
        csv_file << "Step,True_S,Meas_S,Est_S,True_D,Meas_D,Est_D,True_Vx,Meas_Vx,Est_Vx\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "[ EKF Standalone Test ]: Gaussian Noise Filtering\n";
    std::cout << "Step | True D | Meas D | Est D  || True Vx | Meas Vx | Est Vx \n";
    std::cout << "------------------------------------------------------------\n";

    for (int step = 0; step < 100; ++step) {
        // [시뮬레이션] 가상의 제어 입력 (S자 곡선 유도)
        StaticVector<double, Nu> u_cmd;
        u_cmd(0) = 0.1 * std::sin(0.2 * step);  // 조향각 변동
        u_cmd(1) = 0.5 * std::cos(0.1 * step);  // 가속도 변동

        // 1. 진짜 차량 이동 (Plant)
        x_true = integrator::step_rk4<Nx, Nu>(plant, x_true, u_cmd, dt);

        // 2. 센서 측정 (노이즈 합성)
        StaticVector<double, Nx> z_meas;
        z_meas(0) = x_true(0) + noise_s(gen);
        z_meas(1) = x_true(1) + noise_d(gen);
        z_meas(2) = x_true(2) + noise_mu(gen);
        z_meas(3) = x_true(3) + noise_v(gen);
        z_meas(4) = x_true(4) + noise_v(gen);  // Vy
        z_meas(5) = x_true(5) + noise_r(gen);

        // 3. EKF 타격 (Predict & Update)
        // 우리가 만든 SIMD 가속 StaticMatrix 연산이 여기서 불을 뿜습니다.
        ekf.predict(plant, u_cmd, dt);
        ekf.update(z_meas);

        // 4. 결과 출력 및 로깅
        if (step % 5 == 0) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(4) << step << " | "
                      << std::setw(6) << x_true(1) << " | " << std::setw(6) << z_meas(1) << " | "
                      << std::setw(6) << ekf.x_est(1) << " || " << std::setw(7) << x_true(3)
                      << " | " << std::setw(7) << z_meas(3) << " | " << std::setw(7) << ekf.x_est(3)
                      << "\n";
        }

        if (csv_file.is_open()) {
            csv_file << step << "," << x_true(0) << "," << z_meas(0) << "," << ekf.x_est(0) << ","
                     << x_true(1) << "," << z_meas(1) << "," << ekf.x_est(1) << "," << x_true(3)
                     << "," << z_meas(3) << "," << ekf.x_est(3) << "\n";
        }
    }

    if (csv_file.is_open()) csv_file.close();

    // [Architect's Assertion] 최종 추정 오차가 일정 수준 이하인지 검증
    double error_d = std::abs(x_true(1) - ekf.x_est(1));
    EXPECT_LT(error_d, 0.5) << "EKF failed to converge on lateral position (D).";

    std::cout << "============================================================\n";
}