#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// [Architect's Full Stack]
#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/Estimator/EKF.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::controller;
using namespace Optimization::estimator;

TEST(Integrated_Pipeline, EKF_NMPC_ClosedLoop) {
    constexpr size_t H = 30;
    constexpr size_t Nx = 6;
    constexpr size_t Nu = 2;
    double dt = 0.1;

    // 1. 코어 엔진 초기화
    SparseNMPC<H> nmpc;
    EKF<Nx, Nu> ekf;
    vehicle::DynamicBicycleModel plant;

    NMPCTuningConfig config;
    config.target_vx = 10.0;
    config.kappa = 0.0;

    // 2. 물리 세계 (True State) 및 추정 세계 (Est State)
    StaticVector<double, Nx> x_true;
    x_true.set_zero();
    x_true(3) = 10.0;

    ekf.x_est = x_true; // 초기화

    StaticVector<double, Nu> u_cmd;
    u_cmd.set_zero();

    // 3. 센서 노이즈 생성기 (Carla의 GPS/IMU 모사)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_s(0.0, 0.3);  
    std::normal_distribution<double> noise_d(0.0, 0.3);  
    std::normal_distribution<double> noise_mu(0.0, 0.05); 
    std::normal_distribution<double> noise_v(0.0, 0.1);  
    std::normal_distribution<double> noise_r(0.0, 0.02); 

    // 4. 로깅 준비
    std::ofstream csv_file("integrated_pipeline_test.csv");
    if (csv_file.is_open()) {
        csv_file << "Step,Ref_D,True_D,Meas_D,Est_D,Steer_deg,Accel,KKT_Err\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "[ Grand Integration ]: EKF (Sensor) -> NMPC (Brain) -> Plant\n";
    std::cout << "Step | Ref_D | True_D | Meas_D | Est_D  || Steer(deg) | Accel \n";
    std::cout << "------------------------------------------------------------\n";

    for (int step = 0; step < 150; ++step) {
        
        // ---------------------------------------------------------
        // [Phase 1: 예언 (Trajectory Generation)]
        // ---------------------------------------------------------
        double current_s_est = ekf.x_est(0); // NMPC는 오직 EKF의 추정치만 믿어야 함
        for (size_t k = 0; k <= H; ++k) {
            double future_s = current_s_est + (config.target_vx * k * dt);
            config.target_d[k] = 3.0 * std::sin(0.1 * future_s); // S자 궤적
        }
        double ref_d = config.target_d[0];

        // ---------------------------------------------------------
        // [Phase 2: 센서 계측 (Sensor Synthesis)]
        // ---------------------------------------------------------
        StaticVector<double, Nx> z_meas;
        z_meas(0) = x_true(0) + noise_s(gen);
        z_meas(1) = x_true(1) + noise_d(gen);
        z_meas(2) = x_true(2) + noise_mu(gen);
        z_meas(3) = x_true(3) + noise_v(gen);
        z_meas(4) = x_true(4) + noise_v(gen);
        z_meas(5) = x_true(5) + noise_r(gen);

        // ---------------------------------------------------------
        // [Phase 3: EKF 노이즈 정제 (State Estimation)]
        // ---------------------------------------------------------
        // 이전 제어 입력(u_cmd)을 바탕으로 예측 후 센서값(z_meas)으로 보정
        ekf.predict(plant, u_cmd, dt);
        ekf.update(z_meas);

        // ---------------------------------------------------------
        // [Phase 4: NMPC 제어 연산 (Optimal Control)]
        // ---------------------------------------------------------
        // 진짜 상태(x_true)는 은닉하고, EKF가 정제한 상태(ekf.x_est)만 NMPC에 투입!
        NMPCResult res = nmpc.solve_rt_qp(ekf.x_est, config);
        EXPECT_TRUE(res.success || res.fallback_triggered);
        
        u_cmd = nmpc.U_guess[0]; // 다음 스텝 및 물리 엔진을 위한 제어 입력 확정

        double steer_deg = u_cmd(0) * 180.0 / M_PI;
        double accel = u_cmd(1);

        // ---------------------------------------------------------
        // [Phase 5: 물리 세계 업데이트 (Plant Simulation)]
        // ---------------------------------------------------------
        x_true = integrator::step_rk4<Nx, Nu>(plant, x_true, u_cmd, dt);
        nmpc.shift_sequence();

        // ---------------------------------------------------------
        // [Phase 6: 입법자의 블랙박스 (Logging)]
        // ---------------------------------------------------------
        if (step % 5 == 0) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(4) << step << " | "
                      << std::setw(5) << ref_d << " | " << std::setw(6) << x_true(1) << " | " 
                      << std::setw(6) << z_meas(1) << " | " << std::setw(6) << ekf.x_est(1) << " || "
                      << std::setw(10) << steer_deg << " | " << std::setw(5) << accel << "\n";
        }

        if (csv_file.is_open()) {
            csv_file << step << "," << ref_d << "," << x_true(1) << "," << z_meas(1) << "," 
                     << ekf.x_est(1) << "," << steer_deg << "," << accel << "," << res.max_kkt_error << "\n";
        }
    }

    if (csv_file.is_open()) csv_file.close();
    std::cout << "============================================================\n";
}