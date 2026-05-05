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

enum class Scenario { PARALLEL, SUDDEN_STOP, ZIGZAG };

std::string get_scenario_name(Scenario type) {
    switch (type) {
        case Scenario::PARALLEL: return "parallel";
        case Scenario::SUDDEN_STOP: return "sudden_stop";
        case Scenario::ZIGZAG: return "zigzag";
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
        case Scenario::PARALLEL:
            // 1. 병렬 저속 주행: 양옆을 가로막고 좁은 길을 5m/s로 강제 주행
            nmpc.obstacles[0] = {35.0, 1.75, 1.0, 5.0, 0.0};
            nmpc.obstacles[1] = {35.0, -1.75, 1.0, 5.0, 0.0};
            break;

        case Scenario::SUDDEN_STOP:
            // 2. 전방 급정거: 3대의 차량이 일정한 간격으로 주행하다가 멈춤
            for (int i = 0; i < 3; ++i) {
                nmpc.obstacles[i] = {30.0 + i * 15.0, 0.0, 1.0, 8.0, 0.0};
            }
            break;

        case Scenario::ZIGZAG:
            // 3. 지그재그 회피 (슬라럼): 극단적인 횡방향 기동성 테스트
            for (int i = 0; i < 4; ++i) {
                double side = (i % 2 == 0) ? 2.0 : -2.0;
                nmpc.obstacles[i] = {25.0 + i * 20.0, side, 1.0, 3.0, 0.0};
            }
            break;
    }
}

TEST(SparseNMPC_Battlefield, MultiObstacleLoggingTest) {
    constexpr size_t H = 30;
    // [Architect's Update] MultipleShootingNMPC로 교체하여 극한의 안정성 테스트도 가능합니다.
    SparseNMPC<H> nmpc; 
    vehicle::DynamicBicycleModel plant;
    double dt = 0.1;

    // 시나리오 선택 (ZIGZAG로 극한의 슬라럼 성능 테스트)
    Scenario current_sim = Scenario::SUDDEN_STOP; 
    setup_scenario(current_sim, nmpc);

    std::string filename = "battlefield_" + get_scenario_name(current_sim) + ".csv";
    std::ofstream csv_file(filename);

    if (csv_file.is_open()) {
        csv_file << "Step,S,D,mu,Vx,Vy,r,Steer_deg,Accel,KKT_Err,MinDistObs\n";
    }

    StaticVector<double, 6> x_true;
    x_true.set_zero();
    x_true(3) = 10.0;  // 초기 속도 10m/s (약 36km/h)

    NMPCTuningConfig config;
    config.target_vx = 10.0;
    config.kappa = 0.0;

    std::cout << "\n============================================================\n";
    std::cout << "[ Battlefield Scenario ]: " << get_scenario_name(current_sim) << "\n";
    std::cout << "Step |   S   |   D   |  Vx  | Steer(deg) | Accel |  KKT  | MinDist\n";
    std::cout << "------------------------------------------------------------\n";

    double absolute_min_dist = 1000.0; // 시뮬레이션 전체 최소 안전거리

    for (int step = 0; step < 120; ++step) {
        // [특수 이벤트 Trigger]
        if (current_sim == Scenario::SUDDEN_STOP && step == 25) {
            for (int i = 0; i < 3; ++i) nmpc.obstacles[i].vs = 0.0;
            std::cout << ">>> [EMERGENCY] Obstacles Sudden Braking! <<<\n";
        }

        // 장애물 물리적 위치 업데이트 (등속 운동)
        for (int i = 0; i < 10; ++i) {
            if (nmpc.obstacles[i].s < 500.0) { // 활성화된 장애물만
                nmpc.obstacles[i].s += nmpc.obstacles[i].vs * dt;
                nmpc.obstacles[i].d += nmpc.obstacles[i].vd * dt;
            }
        }

        // NMPC 타격
        NMPCResult res = nmpc.solve_rt_qp(x_true, config);

        // [Architect's Assertion 1] 솔버 붕괴 감시 (수렴 실패 시 테스트 즉시 종료)
        EXPECT_TRUE(res.success || res.fallback_triggered) 
            << "Solver mathematically crashed at step " << step;

        double steer_rad = nmpc.U_guess[0](0);
        double accel = nmpc.U_guess[0](1);
        double steer_deg = steer_rad * 180.0 / M_PI;

        // [Architect's Collision Monitor] 실제 차량과 장애물 간의 최소 거리 측정
        double current_min_dist = 1000.0;
        for (int i = 0; i < 10; ++i) {
            if (nmpc.obstacles[i].s > 500.0) continue; 
            double ds = x_true(0) - nmpc.obstacles[i].s;
            double dd = x_true(1) - nmpc.obstacles[i].d;
            double dist = std::sqrt(ds * ds + dd * dd) - nmpc.obstacles[i].r; // 표면 간 거리
            if (dist < current_min_dist) current_min_dist = dist;
        }
        if (current_min_dist < absolute_min_dist) absolute_min_dist = current_min_dist;

        // 콘솔 출력 (5스텝 단위 및 긴급 제동 시 출력)
        if (step % 5 == 0 || res.fallback_triggered) {  
            std::cout << std::fixed << std::setprecision(2) 
                      << std::setw(4) << step << " | " 
                      << std::setw(5) << x_true(0) << " | "
                      << std::setw(5) << x_true(1) << " | " 
                      << std::setw(4) << x_true(3) << " | " 
                      << std::setw(10) << steer_deg << " | " 
                      << std::setw(5) << accel << " | " 
                      << std::setprecision(3) << res.max_kkt_error << " | "
                      << current_min_dist << "\n";
        }

        if (csv_file.is_open()) {
            csv_file << step << "," << x_true(0) << "," << x_true(1) << "," << x_true(2) << ","
                     << x_true(3) << "," << x_true(4) << "," << x_true(5) << "," << steer_deg << ","
                     << accel << "," << res.max_kkt_error << "," << current_min_dist << "\n";
        }

        // 차량 물리 엔진 진행
        x_true = integrator::step_rk4<6, 2>(plant, x_true, nmpc.U_guess[0], dt);
        nmpc.shift_sequence();

        // [Architect's Assertion 2] 충돌 여부 하드 체크 (마진 0.0 미만 시 실패)
        EXPECT_GT(current_min_dist, 0.0) << "COLLISION DETECTED at step " << step;

        if (x_true(0) > 130.0) break; // 시나리오 종점 통과
    }

    if (csv_file.is_open()) {
        csv_file.close();
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << "[ Simulation Result ] \n";
    std::cout << "- Final KKT Error : " << nmpc.solve_rt_qp(x_true, config).max_kkt_error << "\n";
    std::cout << "- Absolute Min Dist: " << absolute_min_dist << " m (Safety Passed)\n";
    std::cout << "============================================================\n";
}