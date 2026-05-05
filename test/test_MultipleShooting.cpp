#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "Optimization/Controller/MultipleShootingNMPC.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/VehicleModel/DynamicBicycleModel.hpp"

using namespace Optimization;
using namespace Optimization::controller;

// =========================================================================
// [Test Suite] O(H) Riccati-based Multiple Shooting NMPC
// =========================================================================
TEST(MultipleShootingTest, O_H_Riccati_Formulation) {
    constexpr size_t H = 20;
    constexpr size_t Nx = 6;
    constexpr size_t Nu = 2;

    MultipleShootingNMPC<H, Nx, Nu> nmpc;
    NMPCTuningConfig config;
    
    StaticVector<double, Nx> x_curr;
    x_curr.set_zero();
    x_curr(0) = 0.0;   // s
    x_curr(1) = 1.0;   // d (1m 엇나감)
    x_curr(2) = 0.2;   // mu (약 11도 틀어짐)
    x_curr(3) = 10.0;  // vx (10m/s)

    std::cout << "================ [ Multiple Shooting NMPC ] ================\n";
    std::cout << "Step | Est D  | Est mu | Steer(deg) | KKT Err\n";
    std::cout << "------------------------------------------------------------\n";

    // [Architect's Fix] 물리적 복귀 시간을 충분히 주기 위해 30스텝(3.0초)으로 증가
    for (int step = 0; step <= 30; ++step) {
        NMPCResult res = nmpc.solve_rt_qp(x_curr, config);

        EXPECT_TRUE(res.success) << "MS-NMPC Solver failed at step " << step;

        std::cout << std::setw(4) << step << " | "
                  << std::fixed << std::setprecision(2) << std::setw(6) << x_curr(1) << " | "
                  << std::setw(6) << x_curr(2) << " | "
                  << std::setw(10) << nmpc.U_guess[0](0) * 180.0 / M_PI << " | "
                  << std::setprecision(4) << res.max_kkt_error << "\n";

        vehicle::DynamicBicycleModel model;
        x_curr = integrator::step_rk4<Nx, Nu>(model, x_curr, nmpc.U_guess[0], nmpc.dt);

        nmpc.shift_sequence();
    }
    
    // 3초 후에는 완벽하게 차선 중앙을 유지하고 있어야 함
    EXPECT_NEAR(x_curr(1), 0.0, 0.05); 
    EXPECT_NEAR(x_curr(2), 0.0, 0.05);  
}