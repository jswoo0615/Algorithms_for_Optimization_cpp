#include <gtest/gtest.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <cmath>
#include <random>

#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Estimator/EKF.hpp"

using namespace Optimization;
using namespace Optimization::controller;
using namespace Optimization::estimator;

TEST(SparseNMPC_Test, Frenet_Dynamic_Obstacle_Avoidance) {
    constexpr size_t H = 30; 
    SparseNMPC<H> nmpc;
    EKF<6, 2> ekf; 

    // [Architect's Scenario: 굽은 도로 (Curve)]
    // 곡률 kappa = 0.01 (반지름 100m의 좌코너)
    double current_kappa = 0.01; 

    // [True State in Frenet] 
    // [S(거리), D(횡방향 편차), mu(헤딩 오차), Vx, Vy, r]
    StaticVector<double, 6> x_true;
    x_true.set_zero();
    x_true(3) = 10.0; // 초기 속도 10m/s

    ekf.x_est = x_true;

    // [Dynamic Obstacle in Frenet]
    // 30m 앞, 우측 차선(D = -3.0)에서 좌측 차선 중앙(D = 0)으로 초당 0.5m씩 밀고 들어옴
    nmpc.obstacles[0].s = 30.0; 
    nmpc.obstacles[0].d = -3.0; 
    nmpc.obstacles[0].r = 1.0;
    nmpc.obstacles[0].vs = 5.0;  // 장애물 종방향 속도 (내 차보다 느림)
    nmpc.obstacles[0].vd = 0.5;  // 장애물 횡방향 침투 속도

    vehicle::DynamicBicycleModel plant;
    plant.kappa = current_kappa; // 식물(Plant) 물리 엔진에 곡률 주입
    double dt = 0.1;

    // 센서 노이즈 (Frenet 공간에서도 노이즈는 존재함)
    std::mt19937 gen(42);
    std::normal_distribution<double> noise_pos(0.0, 0.1);  // D 오차 10cm
    std::normal_distribution<double> noise_yaw(0.0, 0.03); // mu 오차 약 1.7도
    std::normal_distribution<double> noise_vel(0.0, 0.2);  // Vx 오차 0.2m/s

    std::string csv_filename = "nmpc_frenet_log.csv";
    std::ofstream log_file(csv_filename);
    if (log_file.is_open()) {
        log_file << "Step,S,D,Meas_D,Est_D,Est_mu,Vx,Steer(deg),MaxKKT\n";
    }

    std::cout << "\n================ [ Frenet NMPC: Curve & Dynamic Obstacle ] ================\n";
    std::cout << "Step | True S | True D | Meas D | Est D | Est mu | Steer(deg) | KKT Err\n";
    std::cout << "---------------------------------------------------------------------------\n";

    StaticVector<double, 2> u_applied; u_applied.set_zero();

    for (int step = 0; step < 80; ++step) { 
        // 1. 동적 장애물 물리 업데이트 (Frenet 공간 이동)
        nmpc.obstacles[0].s += nmpc.obstacles[0].vs * dt;
        nmpc.obstacles[0].d += nmpc.obstacles[0].vd * dt;
        
        // 2. 센서 노이즈 주입 (Frenet 상태 측정)
        StaticVector<double, 6> x_meas = x_true;
        x_meas(1) += noise_pos(gen); // D 노이즈
        x_meas(2) += noise_yaw(gen); // mu 노이즈
        x_meas(3) += noise_vel(gen); // Vx 노이즈

        // 3. EKF 추정 (Frenet 모델 사용)
        ekf.predict(plant, u_applied, dt); 
        ekf.update(x_meas);                

        // 4. 제어기 튜닝 및 타겟 설정
        NMPCTuningConfig config;
        config.kappa = current_kappa; 
        config.target_vx = 10.0;
        
        double vx = std::max(0.1, ekf.x_est(3));
        
        // [Architect's Fix: Control Jitter 억제]
        // 1. 절대 조향각 페널티 증가 (직진하려는 성향 강화)
        config.R_Steer = 200.0 + 5.0 * (vx * vx); // 기존 10.0 에서 대폭 상향
        
        // 2. 조향 변화율 페널티 '극대화' (핸들을 빠르게 떠는 것을 원천 봉쇄)
        // 이 값이 커지면 NMPC는 Low-pass 필터처럼 부드러운 궤적만 생성하게 됩니다.
        config.R_Steer_Rate = 50000.0 + 100.0 * (vx * vx); // 기존 50.0 에서 1000배 이상 상향
        
        // 5. Frenet NMPC 타격 (로컬 타겟 x_ref 불필요)
        NMPCResult res = nmpc.solve_rt_qp(ekf.x_est, config);
        
        u_applied = nmpc.U_guess[0]; 
        double steer_deg = u_applied(0) * (180.0 / M_PI); // 이제 U(0)가 조향입니다 (기존 코드와 인덱스 일치 여부 확인)

        // 로깅
        std::cout << std::setw(4) << step << " | "
                  << std::fixed << std::setprecision(2)
                  << std::setw(6) << x_true(0) << " | " // S
                  << std::setw(6) << x_true(1) << " | " // D
                  << std::setw(6) << x_meas(1) << " | "
                  << std::setw(6) << ekf.x_est(1) << " | "
                  << std::setw(6) << ekf.x_est(2)*(180/M_PI) << " | "
                  << std::setw(10) << steer_deg << " | "
                  << std::setprecision(4) << std::setw(7) << res.max_kkt_error << "\n";

        if (log_file.is_open()) {
            log_file << step << "," << x_true(0) << "," << x_true(1) << "," 
                     << x_meas(1) << "," << ekf.x_est(1) << "," 
                     << ekf.x_est(2)*(180/M_PI) << "," << ekf.x_est(3) << "," 
                     << steer_deg << "," << res.max_kkt_error << "\n";
        }

        // 6. 물리 엔진 업데이트 (진짜 차량 이동)
        x_true = integrator::step_rk4<6, 2>(plant, x_true, u_applied, dt);
        nmpc.shift_sequence();

        // 장애물 패스 시 제거
        if (nmpc.obstacles[0].s < x_true(0) - 2.0) {
            nmpc.obstacles[0].s = 10000.0; 
        }
    }
    if (log_file.is_open()) log_file.close();
}