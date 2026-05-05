#include <gtest/gtest.h>
#include <cmath>

#include "Optimization/Solver/SQPSolver.hpp"

using namespace Optimization;
using namespace Optimization::solver;

// =========================================================================
// [Functors] 비선형 목적 함수 및 제약 조건 정의
// =========================================================================

// 1. 비선형 목적 함수: f(x) = (x0 - 2)^2 + (x1 - 2)^2
struct CircleCost {
    template <typename T>
    T operator()(const StaticVector<T, 2>& x) const {
        // [Architect's Note] DualVec 엔진의 연산자 오버로딩이 완벽히 작동하는 구간
        T dx = x(0) - 2.0;
        T dy = x(1) - 2.0;
        return dx * dx + dy * dy;
    }
};

// 2. 비선형 부등식 제약 조건: h(x) = x0^2 + x1^2 - 1 <= 0
struct CircleIneq {
    template <typename T>
    StaticVector<T, 1> operator()(const StaticVector<T, 2>& x) const {
        StaticVector<T, 1> h;
        h(0) = x(0) * x(0) + x(1) * x(1) - 1.0;
        return h;
    }
};

// =========================================================================
// [Test Suite] Sequential Quadratic Programming (SQP) Solver
// =========================================================================
TEST(SQPSolverTest, NonlinearCircleConstraint) {
    CircleCost cost_func;
    CircleIneq ineq_func;

    // 1. 초기 위치 (원점)
    StaticVector<double, 2> x_opt;
    x_opt.set_zero();

    // 2. SQP 사령탑 타격 (최대 15번의 QP 하청, 허용 오차 1e-5)
    SolverStatus status = SQPSolver<2, 1, CircleCost, CircleIneq>::solve(
        cost_func, ineq_func, x_opt, 15, 1e-5);

    // 3. 상태 검증
    EXPECT_EQ(status, SolverStatus::SUCCESS);

    // 4. 수학적 해석해 검증 (1/sqrt(2) = 0.70710678)
    double expected_val = 1.0 / std::sqrt(2.0);
    
    // 허용 오차 1e-4 내에서 정확히 경계면에 안착했는지 확인
    EXPECT_NEAR(x_opt(0), expected_val, 1e-4);
    EXPECT_NEAR(x_opt(1), expected_val, 1e-4);

    // [로깅] 결과 확인용
    std::cout << "[SQP Output] x0: " << x_opt(0) << ", x1: " << x_opt(1) << "\n";
}