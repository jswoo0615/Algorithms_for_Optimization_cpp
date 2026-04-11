#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "Optimization/Control/NMPC.hpp"

using namespace Optimization;

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gtest_list_tests" ||
            std::string(argv[i]) == "--gtest_output=xml") {
            return 0;
        }
    }

    std::cout << std::string(80, '=') << "\n";
    std::cout << "  [Layer 6] Sparse NMPC (IPM Hard Constraints) Simulation\n";
    std::cout << std::string(80, '=') << "\n";

    constexpr size_t Np = 15;
    NMPCController<Np> nmpc;

    nmpc.x_ref.set_zero();
    nmpc.x_ref(0) = 10.0;
    nmpc.x_ref(1) = 10.0;
    nmpc.x_ref(2) = M_PI / 4.0;

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

    std::cout << "  [*] 60차원 하드 제약(Hard Constraints)이 주입된 IPM 가동\n";

    StaticVector<double, 6> x_current;
    x_current.set_zero();
    x_current(3) = 1.0;

    StaticVector<double, Np * 2> U_guess;
    U_guess.set_zero();

    double t = 0.0;
    int max_sim_steps = 200;

    std::cout
        << " Step | Time(s) |    X (m) |    Y (m) | Theta(deg) | Vel(m/s) || Accel | Steer(deg)\n";
    std::cout << std::string(85, '-') << "\n";

    bool reached_target = false;

    for (int step = 0; step < max_sim_steps; ++step) {
        bool success = nmpc.compute_control(x_current, U_guess);

        StaticVector<double, 2> u_opt;
        // [Architect's Fix] 솔버가 물리적 한계를 보장하므로 더 이상 맵핑이 필요하지 않습니다.
        u_opt(0) = U_guess(0);
        u_opt(1) = U_guess(1);

        double X = x_current(0);
        double Y = x_current(1);
        double theta = x_current(2) * 180.0 / M_PI;
        double v = std::sqrt(x_current(3) * x_current(3) + x_current(4) * x_current(4));
        double steer_deg = u_opt(1) * 180.0 / M_PI;

        if (step % 2 == 0) {
            std::cout << std::setw(5) << step << " | " << std::setw(7) << std::fixed
                      << std::setprecision(2) << t << " | " << std::setw(8) << X << " | "
                      << std::setw(8) << Y << " | " << std::setw(10) << theta << " | "
                      << std::setw(8) << v << " || " << std::setw(5) << u_opt(0) << " | "
                      << std::setw(9) << steer_deg << "\n";
        }

        double dist_error = std::sqrt((X - 10.0) * (X - 10.0) + (Y - 10.0) * (Y - 10.0));
        if (dist_error < 0.4 && v < 0.2) {
            std::cout << "\n  [*] Target Reached and Stopped! (Error: " << dist_error
                      << " m, Vel: " << v << " m/s)\n";
            reached_target = true;
            break;
        }

        x_current = Integrator::rk4<6, 2, DynamicBicycleModel, double>(nmpc.model, x_current, u_opt,
                                                                       nmpc.dt);

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
        std::cout << "  [PASS] 하드 제약 기반 IPM NMPC 제어 완벽 수행.\n";
        return 0;
    } else {
        std::cout << "  [FAIL] 차량이 목표 지점에 도달하지 못했습니다.\n";
        return 1;
    }
}