#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Optimization/PathPlanner.hpp"
#include "Optimization/RTINMPC.hpp"

using namespace Optimization;

int main() {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "  [Layer 6] RT-NMPC Extreme Slalom: Gate-Crossing Mode\n";
    std::cout << std::string(80, '=') << "\n";

    constexpr size_t Np = 15;
    RTINMPCController<Np> nmpc;
    StaticVector<double, 6> x_final;
    x_final.set_zero();
    x_final(0) = 10.0;
    x_final(1) = 10.0;
    x_final(2) = M_PI / 4.0;

    // 장애물 배치 및 플래너 데이터 주입
    nmpc.obstacles[0] = {2.0, 1.5, 0.5};
    nmpc.obstacles[1] = {1.5, 3.5, 0.5};
    nmpc.obstacles[2] = {4.0, 3.0, 0.5};
    nmpc.obstacles[3] = {3.5, 5.0, 0.5};
    nmpc.obstacles[4] = {6.0, 5.0, 0.5};
    nmpc.obstacles[5] = {5.0, 7.0, 0.5};
    nmpc.obstacles[6] = {8.0, 6.5, 0.5};
    nmpc.obstacles[7] = {7.0, 8.5, 0.5};
    nmpc.obstacles[8] = {9.0, 8.0, 0.5};
    nmpc.obstacles[9] = {8.0, 10.5, 0.2};

    std::array<PathPlanner<Np, 6>::ObstacleInfo, 10> obs_data;
    for (int i = 0; i < 10; ++i) {
        obs_data[i] = {nmpc.obstacles[i].x, nmpc.obstacles[i].y, nmpc.obstacles[i].r};
    }

    StaticVector<double, 6> x_curr;
    x_curr.set_zero();
    x_curr(3) = 1.0;
    StaticVector<double, Np * 2> U_guess;
    U_guess.set_zero();

    std::cout << " Step |    X    |    Y    |  Steer(deg) | Status\n"
              << std::string(55, '-') << "\n";

    for (int step = 0; step < 100; ++step) {
        StaticVector<double, 6> ref_horizon[Np];

        // [Corrected Function Call]
        PathPlanner<Np, 6>::generate_slalom_reference(x_curr, x_final, obs_data, nmpc.dt,
                                                      ref_horizon);

        bool success = nmpc.compute_control(x_curr, ref_horizon, U_guess);

        StaticVector<double, 2> u_opt;
        u_opt(0) = U_guess(0);
        u_opt(1) = U_guess(1);

        std::cout << std::setw(5) << step << " | " << std::fixed << std::setprecision(2)
                  << std::setw(7) << x_curr(0) << " | " << std::setw(7) << x_curr(1) << " | "
                  << std::setw(10) << u_opt(1) * 180 / M_PI << " | " << (success ? "OK" : "FAIL")
                  << "\n";

        if (std::sqrt(std::pow(x_curr(0) - x_final(0), 2) + std::pow(x_curr(1) - x_final(1), 2)) <
            0.3) {
            std::cout << "\n  [*] Slalom Course Completed!\n";
            break;
        }
        x_curr =
            Integrator::rk4<6, 2, DynamicBicycleModel, double>(nmpc.model, x_curr, u_opt, nmpc.dt);

        // Warm-start Shift
        for (size_t i = 0; i < Np - 1; ++i) {
            U_guess(static_cast<int>(i * 2)) = U_guess(static_cast<int>((i + 1) * 2));
            U_guess(static_cast<int>(i * 2 + 1)) = U_guess(static_cast<int>((i + 1) * 2 + 1));
        }
    }
    std::cout << std::string(80, '=') << "\n";
    return 0;
}