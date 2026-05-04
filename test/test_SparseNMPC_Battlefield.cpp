#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"
#include "Optimization/Integrator/RK4.hpp"

using namespace Optimization;
using namespace Optimization::controller;

enum class Scenario { PARALLEL, SUDDEN_STOP, ZIGZAG };

// 시나리오 이름을 문자열로 변환하는 헬퍼 함수
std::string get_scenario_name(Scenario type) {
    switch (type) {
        case Scenario::PARALLEL:    return "parallel";
        case Scenario::SUDDEN_STOP: return "sudden_stop";
        case Scenario::ZIGZAG:      return "zigzag";
        default:                    return "unknown";
    }
}

void setup_scenario(Scenario type, SparseNMPC<30>& nmpc) {
    // 장애물 초기화 (안전한 곳으로 배치)
    for (int i = 0; i < 10; ++i) {
        nmpc.obstacles[i].s = 1000.0;
        nmpc.obstacles[i].d = 0.0;
    }

    switch (type) {
        case Scenario::PARALLEL:
            // 1. 병렬 저속 주행: 추월 불가 구간
            nmpc.obstacles[0] = {35.0,  1.75, 1.0, 5.0, 0.0}; 
            nmpc.obstacles[1] = {35.0, -1.75, 1.0, 5.0, 0.0}; 
            break;

        case Scenario::SUDDEN_STOP:
            // 2. 전방 급정거
            for (int i = 0; i < 3; ++i) {
                nmpc.obstacles[i] = {30.0 + i * 15.0, 0.0, 1.0, 8.0, 0.0};
            }
            break;

        case Scenario::ZIGZAG:
            // 3. 지그재그 회피 (슬라럼)
            for (int i = 0; i < 4; ++i) {
                double side = (i % 2 == 0) ? 2.0 : -2.0;
                nmpc.obstacles[i] = {25.0 + i * 20.0, side, 1.0, 3.0, 0.0};
            }
            break;
    }
}

TEST(SparseNMPC_Battlefield, MultiObstacleLoggingTest) {
    constexpr size_t H = 30;
    SparseNMPC<H> nmpc;
    vehicle::DynamicBicycleModel plant;
    double dt = 0.1;

    // [Architect's Choice] 시나리오 선택
    Scenario current_sim = Scenario::PARALLEL; 
    setup_scenario(current_sim, nmpc);

    // CSV 파일 준비
    std::string filename = "battlefield_" + get_scenario_name(current_sim) + ".csv";
    std::ofstream csv_file(filename);
    
    // CSV 헤더 작성
    if (csv_file.is_open()) {
        csv_file << "Step,S,D,mu,Vx,Vy,r,Steer_deg,Accel,KKT_Err\n";
    }

    StaticVector<double, 6> x_true; 
    x_true.set_zero();
    x_true(3) = 10.0; // 초기 속도 10m/s

    NMPCTuningConfig config;
    config.target_vx = 10.0;
    config.kappa = 0.0; 

    std::cout << "\n--- Scenario: " << get_scenario_name(current_sim) << " ---\n";
    std::cout << "Step | S | D | Vx | Steer(deg) | Accel | KKT\n";

    for (int step = 0; step < 120; ++step) {
        // 급정거 시나리오 특수 이벤트: Step 20에서 장애물 전면 정지
        if (current_sim == Scenario::SUDDEN_STOP && step == 25) {
            for (int i = 0; i < 3; ++i) nmpc.obstacles[i].vs = 0.0;
            std::cout << ">>> Obstacles Sudden Braking! <<<\n";
        }

        // 장애물 미래 위치 업데이트
        for (int i = 0; i < 10; ++i) {
            nmpc.obstacles[i].s += nmpc.obstacles[i].vs * dt;
            nmpc.obstacles[i].d += nmpc.obstacles[i].vd * dt;
        }

        // NMPC 솔버 타격
        NMPCResult res = nmpc.solve_rt_qp(x_true, config);
        
        double steer_rad = nmpc.U_guess[0](0);
        double accel = nmpc.U_guess[0](1);
        double steer_deg = steer_rad * 180.0 / M_PI;

        // 터미널 출력
        if (step % 5 == 0) { // 출력이 너무 많지 않게 5스텝마다
            std::cout << std::fixed << std::setprecision(2)
                      << step << " | " << x_true(0) << " | " << x_true(1) << " | " 
                      << x_true(3) << " | " << steer_deg << " | " 
                      << accel << " | " << std::scientific << res.max_kkt_error << "\n";
        }

        // CSV 데이터 기록
        if (csv_file.is_open()) {
            csv_file << step << "," 
                     << x_true(0) << "," << x_true(1) << "," << x_true(2) << ","
                     << x_true(3) << "," << x_true(4) << "," << x_true(5) << ","
                     << steer_deg << "," << accel << "," << res.max_kkt_error << "\n";
        }

        // 실제 물리 엔진 반영
        x_true = integrator::step_rk4<6, 2>(plant, x_true, nmpc.U_guess[0], dt);
        nmpc.shift_sequence();
        
        // 120m 이상 주행 시 종료
        if (x_true(0) > 120.0) break;
    }

    if (csv_file.is_open()) {
        csv_file.close();
        std::cout << "\n[Success] Simulation log saved to: " << filename << "\n";
    }
}