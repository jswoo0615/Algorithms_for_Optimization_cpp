#include <gtest/gtest.h>
#include "Optimization/Solver/QPSolver_IPM.hpp"

using namespace Optimization;
using namespace Optimization::solver;

// =========================================================================
// [Test Suite] Primal-Dual Interior-Point Method (PDIPM) Solver
// =========================================================================
TEST(QPSolverTest, Constrained2D) {
    // 1. Hessian (H) & Gradient (g) 설정
    StaticMatrix<double, 2, 2> H;
    H(0, 0) = 1.0; H(0, 1) = 0.0;
    H(1, 0) = 0.0; H(1, 1) = 2.0;

    StaticVector<double, 2> g;
    g(0) = -1.0; g(1) = -2.0;

    // 2. 부등식 제약 조건 설정 (C * x <= d)
    StaticMatrix<double, 3, 2> C;
    C(0, 0) =  1.0; C(0, 1) =  1.0; // x1 + x2 <= 1
    C(1, 0) = -1.0; C(1, 1) =  0.0; // -x1 <= 0
    C(2, 0) =  0.0; C(2, 1) = -1.0; // -x2 <= 0

    StaticVector<double, 3> d;
    d(0) = 1.0; 
    d(1) = 0.0; 
    d(2) = 0.0;

    // 3. 최적화 변수 초기화
    StaticVector<double, 2> x_opt;
    x_opt.set_zero();

    // 4. IPM 솔버 타격
    SolverStatus status = QPSolver_IPM<2, 3>::solve(H, g, C, d, x_opt, 50, 1e-6);

    // 5. 검증 (Validation)
    EXPECT_EQ(status, SolverStatus::SUCCESS);
    
    // 허용 오차(Tolerance) 1e-4 내에서 이론적 최적해(1/3, 2/3)에 수렴하는지 확인
    EXPECT_NEAR(x_opt(0), 1.0 / 3.0, 1e-4);
    EXPECT_NEAR(x_opt(1), 2.0 / 3.0, 1e-4);
}