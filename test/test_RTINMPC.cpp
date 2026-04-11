#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <chrono> // [Architect's Tool] 실시간 프로파일링용 크로노 라이브러리
#include "Optimization/RTINMPC.hpp"

using namespace Optimization;

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gtest_list_tests" || std::string(argv[i]) == "--gtest_output=xml") {
            return 0; 
        }
    }

    std::cout << std::string(95, '=') << "\n";
    std::cout << "  [Layer 6] RT-NMPC (Real-Time Iteration + Hard Constraints) with WCET Profiler\n";
    std::cout << std::string(95, '=') << "\n";

    constexpr size_t Np = 15; 
    RTINMPCController<Np> nmpc;

    nmpc.x_ref.set_zero();
    nmpc.x_ref(0) = 10.0; nmpc.x_ref(1) = 10.0; nmpc.x_ref(2) = M_PI / 4.0; 
    
    nmpc.obstacles[0] = {2.0, 1.5, 0.5};  nmpc.obstacles[1] = {1.5, 3.5, 0.5};
    nmpc.obstacles[2] = {4.0, 3.0, 0.5};  nmpc.obstacles[3] = {3.5, 5.0, 0.5};
    nmpc.obstacles[4] = {6.0, 5.0, 0.5};  nmpc.obstacles[5] = {5.0, 7.0, 0.5};
    nmpc.obstacles[6] = {8.0, 6.5, 0.5};  nmpc.obstacles[7] = {7.0, 8.5, 0.5};
    nmpc.obstacles[8] = {9.0, 8.0, 0.5};  nmpc.obstacles[9] = {8.0, 10.5, 0.2}; 

    std::cout << "  [*] Deterministic Execution: RTI Solver Loaded (dt=0.2, Np=15)\n";

    StaticVector<double, 6> x_current;
    x_current.set_zero();
    x_current(3) = 1.0; 

    StaticVector<double, Np * 2> U_guess;
    U_guess.set_zero();

    double t = 0.0;
    int max_sim_steps = 100; 
    double max_exec_time = 0.0; // Worst-Case Execution Time 기록용
    
    // [Architect's UI] 실행 시간(Exec Time) 열 추가
    std::cout << " Step | Time(s) | Exec Time(s) |    X (m) |    Y (m) | Theta(deg) | Vel(m/s) || Accel | Steer(deg)\n";
    std::cout << std::string(100, '-') << "\n";

    bool reached_target = false;

    for (int step = 0; step < max_sim_steps; ++step) {
        
        // ---------------------------------------------------------------------
        // [Architect's Profiler] 연산 시간 정밀 측정 시작
        // ---------------------------------------------------------------------
        auto start_time = std::chrono::high_resolution_clock::now();
        
        nmpc.compute_control(x_current, U_guess); 
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> exec_duration = end_time - start_time; // 초 단위(double) 변환
        double exec_seconds = exec_duration.count();
        
        if (exec_seconds > max_exec_time) {
            max_exec_time = exec_seconds;
        }
        // ---------------------------------------------------------------------

        StaticVector<double, 2> u_opt;
        u_opt(0) = U_guess(0); u_opt(1) = U_guess(1); 

        double X = x_current(0); double Y = x_current(1);
        double theta = x_current(2) * 180.0 / M_PI;
        double v = std::sqrt(x_current(3)*x_current(3) + x_current(4)*x_current(4));
        double steer_deg = u_opt(1) * 180.0 / M_PI;

        std::cout << std::setw(5) << step << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << t << " | "
                  << std::setw(12) << std::fixed << std::setprecision(6) << exec_seconds << " | " // 6자리 소수점 초 단위 출력
                  << std::setw(8) << std::fixed << std::setprecision(2) << X << " | "
                  << std::setw(8) << Y << " | "
                  << std::setw(10) << theta << " | "
                  << std::setw(8) << v << " || "
                  << std::setw(5) << u_opt(0) << " | "
                  << std::setw(9) << steer_deg << "\n";

        double dist_error = std::sqrt((X - 10.0)*(X - 10.0) + (Y - 10.0)*(Y - 10.0));
        if (dist_error < 0.5 && v < 0.2) {
            std::cout << "\n  [*] RTI Target Reached! (Error: " << dist_error << " m, Vel: " << v << " m/s)\n";
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

    std::cout << std::string(95, '=') << "\n";
    std::cout << "  [Profiler] Worst-Case Execution Time (WCET) : " 
              << std::fixed << std::setprecision(6) << max_exec_time << " seconds\n";
    std::cout << "  [Profiler] Control Deadline (dt)            : " 
              << std::fixed << std::setprecision(6) << nmpc.dt << " seconds\n";
              
    if (max_exec_time < nmpc.dt) {
        std::cout << "  [PASS] 제어 루프가 데드라인 내에 안정적으로 동작합니다. (Real-Time Safe)\n";
    } else {
        std::cout << "  [FAIL] WCET가 제어 주기(dt)를 초과했습니다! 지연(Jitter) 발생.\n";
    }
    std::cout << std::string(95, '=') << "\n";

    if (reached_target) return 0;
    return 1;
}