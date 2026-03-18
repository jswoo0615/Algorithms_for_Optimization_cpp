#include <gtest/gtest.h>

#include <cmath>

#include "Optimization/Dual.hpp"

using namespace Optimization;

// ======================================================
// 1. Scalar Dual (1D AD) 테스트
// ======================================================
TEST(DualTest, ScalarArithmetic) {
    // f(x) = (x + 3) * (x^2)
    // f(2.0) = (2+3) * 4 = 20
    // f'(x) = 1*(x^2) + (x+3)*(2x) = 3x^2 + 6x -> f'(2.0) = 12 + 12 = 24
    Dual<double> x(2.0, 1.0);  // x = 2.0, dx/dx = 1.0
    auto f = (x + 3.0) * (x * x);

    EXPECT_DOUBLE_EQ(f.v, 20.0);
    EXPECT_DOUBLE_EQ(f.d, 24.0);
}

TEST(DualTest, MathFunctions) {
    // f(x) = sin(x) + exp(x)
    // f'(x) = cos(x) + exp(x)
    double val = 1.5;
    Dual<double> x(val, 1.0);
    auto f = sin(x) + exp(x);

    EXPECT_DOUBLE_EQ(f.v, std::sin(val) + std::exp(val));
    EXPECT_DOUBLE_EQ(f.d, std::cos(val) + std::exp(val));
}

// ======================================================
// 2. Vector Dual (N-Dimensional AD) 테스트
// ======================================================
TEST(DualVecTest, GradientCalculation) {
    // f(x, y) = x^2 * y + sin(x)
    // df/dx = 2xy + cos(x)
    // df/dy = x^2
    const double val_x = 2.0;
    const double val_y = 3.0;

    auto x = DualVec<double, 2>::make_variable(val_x, 0);  // index 0
    auto y = DualVec<double, 2>::make_variable(val_y, 1);  // index 1

    auto f = (x * x * y) + sin(x);

    // 값 검증
    EXPECT_DOUBLE_EQ(f.v, (val_x * val_x * val_y) + std::sin(val_x));

    // 편미분(Gradient) 검증
    EXPECT_DOUBLE_EQ(f.g[0], (2.0 * val_x * val_y) + std::cos(val_x));  // df/dx
    EXPECT_DOUBLE_EQ(f.g[1], (val_x * val_x));                          // df/dy
}

// ======================================================
// 3. 수치적 안정성 (Singularity Handling) 테스트
// ======================================================
TEST(StabilityTest, SqrtAtZero) {
    // f(x) = sqrt(x), x = 0.0일 때 f'(x)는 Inf가 되기 쉬움
    // 우리가 구현한 보호 로직 (u.v <= 1e-16 ? 0.0) 검증
    Dual<double> x(0.0, 1.0);
    auto f = sqrt(x);

    EXPECT_DOUBLE_EQ(f.v, 0.0);
    EXPECT_DOUBLE_EQ(f.d, 0.0);  // 발산하지 않고 0으로 처리되는지 확인
}

TEST(StabilityTest, Atan2Origin) {
    // atan2(y, x)에서 x, y가 모두 0일 때의 안전성
    Dual<double> y(0.0, 1.0);
    Dual<double> x(0.0, 0.0);
    auto f = atan2(y, x);

    EXPECT_DOUBLE_EQ(f.v, 0.0);
    EXPECT_DOUBLE_EQ(f.d, 0.0);
}

// ======================================================
// 4. Complex Step Derivative (CSD) 테스트
// ======================================================
TEST(CSDTest, PrecisionComparison) {
    // f(x) = exp(x) / sin(x)^3 같은 복잡한 함수도 CSD는 기계 정밀도급 미분 가능
    double h = 1e-100;
    double x_val = 1.0;
    std::complex<double> z(x_val, h);

    auto f_z = Optimization::exp(z);  // 헤더의 CSD 오버로드 호출

    // f'(x) = exp(x)
    EXPECT_NEAR(f_z.real(), std::exp(x_val), 1e-15);
    EXPECT_NEAR(f_z.imag() / h, std::exp(x_val), 1e-15);
}