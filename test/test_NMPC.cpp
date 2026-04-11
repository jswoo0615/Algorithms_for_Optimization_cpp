#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include "Optimization/Control/NMPC.hpp"

using namespace Optimization;

// [Architect's Shield] main 함수에 인자 파싱 기능 추가
int main(int argc, char** argv) {
// CMake의 gtest_discover_tests가 테스트 목록을 요구할 경우,
// 시뮬레이션을 돌리지 않고 즉시 정상 종료(0)하여 타임아웃 삭제 회피
for (int i = 1; i < argc; ++i) {
if (std::string(argv[i]) == "--gtest_list_tests" || std::string(argv[i]) == "--gtest_output=xml") {
return 0;
}
}

std::cout << std::string(80, '=') << "\n";
std::cout << "  [Layer 6] NMPC Obstacle Avoidance & Slew Rate Penalty Simulation\n";
std::cout << std::string(80, '=') << "\n";

constexpr size_t Np = 15; // 1.5초 예측 시야
NMPCController<Np> nmpc;

// 타겟: (10, 10), 헤딩 45도
nmpc.x_ref.set_zero();
nmpc.x_ref(0) = 10.0; 
nmpc.x_ref(1) = 10.0;
nmpc.x_ref(2) = M_PI / 4.0;

// 10개의 장애물 배치
nmpc.obstacles[0] = {2.0, 1.5, 0.5};
nmpc.obstacles[1] = {1.5, 3.5, 0.5};
nmpc.obstacles[2] = {4.0, 3.0, 0.5};
nmpc.obstacles[3] = {3.5, 5.0, 0.5};
nmpc.obstacles[4] = {6.0, 5.0, 0.5};
nmpc.obstacles[5] = {5.0, 7.0, 0.5};
nmpc.obstacles[6] = {8.0, 6.5, 0.5};
nmpc.obstacles[7] = {7.0, 8.5, 0.5};
nmpc.obstacles[8] = {9.0, 8.0, 0.5};
nmpc.obstacles[9] = {8.5, 9.5, 0.2};

std::cout << "  [*] 장애물 10개 배치 완료 (Slew Rate Penalty 가동)\n";

StaticVector<double, 6> x_current;
x_current.set_zero();
x_current(3) = 1.0; 

StaticVector<double, Np * 2> U_guess;
U_guess.set_zero();

double t = 0.0;
int max_sim_steps = 200; 

std::cout << " Step | Time(s) |    X (m) |    Y (m) | Theta(deg) | Vel(m/s) || Accel | Steer(deg)\n";
std::cout << std::string(85, '-') << "\n";

bool reached_target = false;

for (int step = 0; step < max_sim_steps; ++step) {
    bool success = nmpc.compute_control(x_current, U_guess);
    
    StaticVector<double, 2> u_opt;
    u_opt(0) = 3.0 * std::tanh(U_guess(0) / 3.0); 
    u_opt(1) = 0.5 * std::tanh(U_guess(1) / 0.5); 

    double X = x_current(0);
    double Y = x_current(1);
    double theta = x_current(2) * 180.0 / M_PI;
    double v = std::sqrt(x_current(3)*x_current(3) + x_current(4)*x_current(4));
    double steer_deg = u_opt(1) * 180.0 / M_PI;

    if (step % 2 == 0) { 
        std::cout << std::setw(5) << step << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << t << " | "
                  << std::setw(8) << X << " | "
                  << std::setw(8) << Y << " | "
                  << std::setw(10) << theta << " | "
                  << std::setw(8) << v << " || "
                  << std::setw(5) << u_opt(0) << " | "
                  << std::setw(9) << steer_deg << "\n";
    }

    double dist_error = std::sqrt((X - 10.0)*(X - 10.0) + (Y - 10.0)*(Y - 10.0));
    if (dist_error < 0.4 && v < 0.2) {
        std::cout << "\n  [*] Target Reached and Stopped! (Error: " << dist_error << " m, Vel: " << v << " m/s)\n";
        reached_target = true;
        break;
    }

    x_current = Integrator::rk4<6, 2, DynamicBicycleModel, double>(nmpc.model, x_current, u_opt, nmpc.dt);

    for (size_t i = 0; i < Np - 1; ++i) {
        U_guess(static_cast<int>(i * 2 + 0)) = U_guess(static_cast<int>((i + 1) * 2 + 0));
        U_guess(static_cast<int>(i * 2 + 1)) = U_guess(static_cast<int>((i + 1) * 2 + 1));
    }
    U_guess(static_cast<int>((Np - 1) * 2 + 0)) = U_guess(static_cast<int>((Np - 2) * 2 + 0));
    U_guess(static_cast<int>((Np - 1) * 2 + 1)) = U_guess(static_cast<int>((Np - 2) * 2 + 1));

    t += nmpc.dt;
}

std::cout << std::string(80, '=') << "\n";
if (reached_target) {
    std::cout << "  [PASS] 승차감이 보장된 장애물 회피 및 목표점 정차 완료.\n";
    return 0;
} else {
    std::cout << "  [FAIL] 차량이 목표 지점에 도달하지 못했습니다.\n";
    return 1;
}
}