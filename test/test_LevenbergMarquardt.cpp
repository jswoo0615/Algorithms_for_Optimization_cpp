#include <gtest/gtest.h>
#include "Optimization/Solver/LevenbergMarquardt.hpp"

using namespace Optimization;
using namespace Optimization::solver;

// =========================================================================
// 악마의 바나나 함수 (Rosenbrock Function) 미니마이저 테스트
// Cost = 0.5 * (r0^2 + r1^2)
// r0(x0, x1) = 10 * (x1 - x0^2)
// r1(x0, x1) = 1 - x0
// 최소점: x0 = 1.0, x1 = 1.0 (Cost = 0)
// =========================================================================
struct RosenbrockResiduals {
    template <typename T>
    StaticVector<T, 2> operator()(const StaticVector<T, 2>& x) const {
        StaticVector<T, 2> r;
        r(0) = T(10.0) * (x(1) - x(0) * x(0));
        r(1) = T(1.0) - x(0);
        return r;
    }
};

TEST(LevenbergMarquardtTest, MinimizeRosenbrock) {
    // N=2 (변수), M=2 (잔차)
    StaticVector<double, 2> x_opt;
    // 일부러 해(1,1)에서 꽤 먼 악의적인 계곡 벽면(-1.2, 1.0)에서 시작
    x_opt(0) = -1.2; 
    x_opt(1) = 1.0;  

    RosenbrockResiduals res_func;
    SolverStatus status = solve_LM<2, 2>(x_opt, res_func, 100, 1e-6);

    EXPECT_EQ(status, SolverStatus::SUCCESS);

    // 구한 해가 전역 최소점(Global Minimum)인 (1.0, 1.0)에 도달했는지 확인
    EXPECT_NEAR(x_opt(0), 1.0, 1e-4);
    EXPECT_NEAR(x_opt(1), 1.0, 1e-4);
}