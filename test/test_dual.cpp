#include <gtest/gtest.h>

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"

using namespace Optimization;

// =========================================================================
// [Test Suite 1] Scalar Auto Differentiation (Dual)
// =========================================================================
TEST(AutoDiffTest, ScalarDerivative) {
    // f(x) = x^2 * sin(x) + sqrt(x)
    Dual<double> x(2.0, 1.0);

    Dual<double> result = (x * x) * ad::sin(x) + ad::sqrt(x);

    double expected_v = 4.0 * std::sin(2.0) + std::sqrt(2.0);
    double expected_d = 4.0 * std::sin(2.0) + 4.0 * std::cos(2.0) + (0.5 / std::sqrt(2.0));

    EXPECT_NEAR(result.v, expected_v, 1e-6);
    EXPECT_NEAR(result.d, expected_d, 1e-6);
}

// =========================================================================
// [Test Suite 2] N-Dimensional Auto Differentiation (DualVec) + Matrix Engine
// =========================================================================
TEST(AutoDiffTest, MultiVariableJacobian) {
    // 2변수 입력 시스템
    using ADVar = DualVec<double, 2>;

    StaticMatrix<ADVar, 2, 2> A;
    // A = [1.0, x0;
    //      x1,  4.0]
    // 행렬 안에 독립 변수 x0, x1을 배치하여 비선형적인 상황 연출

    ADVar x0 = ADVar::make_variable(2.0, 0);  // x0 = 2.0, 편미분 인덱스 0
    ADVar x1 = ADVar::make_variable(3.0, 1);  // x1 = 3.0, 편미분 인덱스 1

    A(0, 0) = 1.0;
    A(0, 1) = x0;
    A(1, 0) = x1;
    A(1, 1) = 4.0;

    StaticVector<ADVar, 2> vec;
    vec(0) = x0;
    vec(1) = x1;

    // y = A * vec 계산 (비선형 행렬 변환)
    // y0 = 1.0*x0 + x0*x1
    // y1 = x1*x0 + 4.0*x1
    auto y = A * vec;

    // 수동 해석적 자코비안 계산
    // dy0/dx0 = 1.0 + x1 = 4.0
    // dy0/dx1 = x0 = 2.0
    // dy1/dx0 = x1 = 3.0
    // dy1/dx1 = x0 + 4.0 = 6.0

    // Value 검증
    EXPECT_DOUBLE_EQ(y(0).v, 2.0 + 6.0);   // 8.0
    EXPECT_DOUBLE_EQ(y(1).v, 6.0 + 12.0);  // 18.0

    // Jacobian(Gradient Vector) 검증
    EXPECT_DOUBLE_EQ(y(0).g[0], 4.0);
    EXPECT_DOUBLE_EQ(y(0).g[1], 2.0);
    EXPECT_DOUBLE_EQ(y(1).g[0], 3.0);
    EXPECT_DOUBLE_EQ(y(1).g[1], 6.0);
}