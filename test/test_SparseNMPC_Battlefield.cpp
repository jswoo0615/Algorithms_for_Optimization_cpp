#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// [Architect's Note] SparseNMPC 또는 MultipleShootingNMPC 자유롭게 교체 가능
#include "Optimization/Controller/SparseNMPC.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::controller;

// 1. [Architect] 시나리오에 Trajectory Tracking 옵션 추가
enum class Scenario { PARALLEL, SUDDEN_STOP, ZIGZAG, TRACK_SINE, TRACK_LANE_CHANGE };

std::string get_scenario_name(Scenario type) {
    switch (type) {
        case Scenario::PARALLEL: return "parallel";
        case Scenario::SUDDEN_STOP: return "sudden_stop";
        case Scenario::ZIGZAG: return "zigzag";
        case Scenario::TRACK_SINE: return "track_sine";
        case Scenario::TRACK_LANE_CHANGE: return "track_lane_change";
        default: return "unknown";
    }
}

template <typename NMPC_Type>
void setup_scenario(Scenario type, NMPC_Type& nmpc) {
    // 장애물 초기화 (관측 범위 밖으로 유배)
    for (int i = 0; i < 10; ++i) {
        nmpc.obstacles[i].s = 1000.0;
        nmpc.obstacles[i].d = 0.0;
    }

    switch (type) {
        // ... (기존 시나리오 생략 없이 동일하게 유지) ...
        case Scenario::PARALLEL:
            nmpc.obstacles[0] = {35.0, 1.75, 1.0, 5.0, 0.0};
            nmpc.obstacles[1] = {35.0, -1.75, 1.0, 5.0, 0.0};
            break;
        case Scenario::SUDDEN_STOP:
            for (int i = 0; i < 3; ++i) {
                nmpc.obstacles[i] = {30.0 + i * 15.0, 0.0, 1.0, 8.0, 0.0};
            }
            break;
        case Scenario::ZIGZAG:
            for (int i = 0; i < 4; ++i) {
                double side = (i % 2 == 0) ? 2.0 : -2.0;
                nmpc.obstacles[i] = {25.0 + i * 20.0, side, 1.0, 3.0, 0.0};
            }
            break;
            
        // 2. [Architect] 장애물이 없는 순수 추종 시나리오는 장애물 배치 안 함
        case Scenario::TRACK_SINE:
        case Scenario::TRACK_LANE_CHANGE:
            // (필요 시 특정 구간에만 장애물을 추가하여 융합 테스트 가능)
            break;
    }
}

TEST(SparseNMPC_Battlefield, MultiObstacleLoggingTest) {
    constexpr size_t H = 30;
    SparseNMPC<H> nmpc;
    vehicle::DynamicBicycleModel plant;
    double dt = 0.1;

    // 3. [Architect] 시나리오를 TRACK_SINE으로 변경
    Scenario current_sim = Scenario::TRACK_SINE;
    setup_scenario(current_sim, nmpc);

    std::string filename = "battlefield_" + get_scenario_name(current_sim) + ".csv";
    std::ofstream csv_file(filename);

    // 4. CSV 헤더에 Ref_D (목표 횡방향 위치) 추가
    if (csv_file.is_open()) {
        csv_file << "Step,S,D,Ref_D,mu,Vx,Vy,r,Steer_deg,Accel,KKT_Err,MinDistObs\n";
    }

    StaticVector<double, 6> x_true;
    x_true.set_zero();
    x_true(3) = 10.0;  // 초기 속도 10m/s

    NMPCTuningConfig config;
    config.target_vx = 10.0;
    config.kappa = 0.0;

    std::cout << "\n============================================================\n";
    std::cout << "[ Battlefield Scenario ]: " << get_scenario_name(current_sim) << "\n";
    std::cout << "Step |   S   |   D   | Ref_D | Steer(deg) | Accel |  KKT  \n";
    std::cout << "------------------------------------------------------------\n";

    double absolute_min_dist = 1000.0;

    for (int step = 0; step < 120; ++step) {
        
        // 5. [Architect's Dynamic Reference] 미래 H 스텝의 목표 궤적 배열 업데이트
        double current_s = x_true(0);
        
        for (size_t k = 0; k <= H; ++k) {
            // 미래 시간 k * dt 동안 target_vx로 이동했을 때의 예상 종방향 위치(S)
            double future_s = current_s + (config.target_vx * k * dt);
            
            if (current_sim == Scenario::TRACK_SINE) {
                // 진폭 3.0m, 파장 약 60m의 연속 S자 곡선
                config.target_d[k] = 3.0 * std::sin(0.1 * future_s); 
            } 
            else if (current_sim == Scenario::TRACK_LANE_CHANGE) {
                // s=30m 지점에서 3.5m(옆 차선)로 급격한 목표 차선 변경
                config.target_d[k] = (future_s > 30.0) ? 3.5 : 0.0;
            } 
            else {
                // 기본 시나리오(직진)
                config.target_d[k] = 0.0;
            }
        }

        // 현재 스텝(k=0)의 목표 d값 (콘솔 및 CSV 로깅용)
        double ref_d = config.target_d[0];

        // [이하 기존 로직 동일] 장애물 위치 업데이트
        for (int i = 0; i < 10; ++i) {
            if (nmpc.obstacles[i].s < 500.0) {  
                nmpc.obstacles[i].s += nmpc.obstacles[i].vs * dt;
                nmpc.obstacles[i].d += nmpc.obstacles[i].vd * dt;
            }
        }

        // NMPC 타격
        NMPCResult res = nmpc.solve_rt_qp(x_true, config);
        EXPECT_TRUE(res.success || res.fallback_triggered) << "Solver crashed at step " << step;

        double steer_rad = nmpc.U_guess[0](0);
        double accel = nmpc.U_guess[0](1);
        double steer_deg = steer_rad * 180.0 / M_PI;

        double current_min_dist = 1000.0;
        for (int i = 0; i < 10; ++i) {
            if (nmpc.obstacles[i].s > 500.0) continue;
            double ds = x_true(0) - nmpc.obstacles[i].s;
            double dd = x_true(1) - nmpc.obstacles[i].d;
            double dist = std::sqrt(ds * ds + dd * dd) - nmpc.obstacles[i].r;
            if (dist < current_min_dist) current_min_dist = dist;
        }
        if (current_min_dist < absolute_min_dist) absolute_min_dist = current_min_dist;

        // 콘솔 및 CSV 출력에 Ref_D 포함
        if (step % 5 == 0 || res.fallback_triggered) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(4) << step << " | "
                      << std::setw(5) << x_true(0) << " | " << std::setw(5) << x_true(1) << " | "
                      << std::setw(5) << ref_d << " | " << std::setw(10) << steer_deg << " | "
                      << std::setw(5) << accel << " | " << std::setprecision(3) << res.max_kkt_error << "\n";
        }

        if (csv_file.is_open()) {
            csv_file << step << "," << x_true(0) << "," << x_true(1) << "," << ref_d << ","
                     << x_true(2) << "," << x_true(3) << "," << x_true(4) << "," << x_true(5) << "," 
                     << steer_deg << "," << accel << "," << res.max_kkt_error << "," << current_min_dist << "\n";
        }

        // 차량 물리 엔진 진행
        x_true = integrator::step_rk4<6, 2>(plant, x_true, nmpc.U_guess[0], dt);
        nmpc.shift_sequence();

        if (current_sim != Scenario::TRACK_SINE && current_sim != Scenario::TRACK_LANE_CHANGE) {
            EXPECT_GT(current_min_dist, 0.0) << "COLLISION DETECTED at step " << step;
        }

        if (x_true(0) > 130.0) break;  
    }

    if (csv_file.is_open()) csv_file.close();

    std::cout << "============================================================\n";
}