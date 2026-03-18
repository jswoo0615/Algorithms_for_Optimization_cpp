#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "Optimization/AutoDiff.hpp"

using namespace Optimization;

class AutoDiffTest : public ::testing::Test {
protected:
    // 테스트에 사용할 스칼라 함수 f(x, y) = x^2 + 3xy + y^3
    // x = 2, y = 3 일 때:
    // Value = 4 + 18 + 27 = 49
    // Gradient = [2x + 3y, 3x + 3y^2] = [13, 33]
    // Hessian = [[2, 3], [3, 6y]] = [[2, 3], [3, 18]]
    static constexpr auto scalar_func = [](const auto& x) {
        return x[0]*x[0] + 3.0*x[0]*x[1] + x[1]*x[1]*x[1];
    };

    // 테스트에 사용할 벡터 함수 F(x, y) = [x^2 + y, 5x - y^2]^T
    // x = 2, y = 3 일 때:
    // Jacobian = [[2x, 1], [5, -2y]] = [[4, 1], [5, -6]]
    static constexpr auto vector_func = [](const auto& x) {
        // 입력 타입(double or DualVec)에 맞춰 반환 배열 타입 자동 결정
        using T = typename std::decay_t<decltype(x)>::value_type; 
        std::array<T, 2> res;
        res[0] = x[0]*x[0] + x[1];
        res[1] = T(5.0)*x[0] - x[1]*x[1];
        return res;
    };
};

// 1. Value 함수 테스트
TEST_F(AutoDiffTest, ValueTest) {
    std::array<double, 2> x = {2.0, 3.0};
    double val = AutoDiff::value<2>(scalar_func, x);
    EXPECT_DOUBLE_EQ(val, 49.0);
}

// 2. Gradient 함수 테스트
TEST_F(AutoDiffTest, GradientTest) {
    std::array<double, 2> x = {2.0, 3.0};
    auto grad = AutoDiff::gradient<2>(scalar_func, x);
    
    EXPECT_DOUBLE_EQ(grad[0], 13.0);
    EXPECT_DOUBLE_EQ(grad[1], 33.0);
}

// 3. Value and Gradient 동시 추출 테스트
TEST_F(AutoDiffTest, ValueAndGradientTest) {
    std::array<double, 2> x = {2.0, 3.0};
    double val = 0.0;
    std::array<double, 2> grad = {0.0, 0.0};
    
    AutoDiff::value_and_gradient<2>(scalar_func, x, val, grad);
    
    EXPECT_DOUBLE_EQ(val, 49.0);
    EXPECT_DOUBLE_EQ(grad[0], 13.0);
    EXPECT_DOUBLE_EQ(grad[1], 33.0);
}

// 4. Jacobian 행렬 테스트
TEST_F(AutoDiffTest, JacobianTest) {
    std::array<double, 2> x = {2.0, 3.0};
    
    // M=2(출력차원), N=2(입력차원)
    auto J = AutoDiff::jacobian<2, 2>(vector_func, x); 
    
    EXPECT_DOUBLE_EQ(J[0][0], 4.0);
    EXPECT_DOUBLE_EQ(J[0][1], 1.0);
    EXPECT_DOUBLE_EQ(J[1][0], 5.0);
    EXPECT_DOUBLE_EQ(J[1][1], -6.0);
}

// 5. Hessian 행렬 테스트 (수치 미분이 섞여 있으므로 EXPECT_NEAR 사용)
TEST_F(AutoDiffTest, HessianTest) {
    std::array<double, 2> x = {2.0, 3.0};
    auto H = AutoDiff::hessian<2>(scalar_func, x);
    
    double tolerance = 1e-5; // 수치 오차 허용 범위
    
    EXPECT_NEAR(H[0][0], 2.0, tolerance);
    EXPECT_NEAR(H[0][1], 3.0, tolerance);
    EXPECT_NEAR(H[1][0], 3.0, tolerance);
    EXPECT_NEAR(H[1][1], 18.0, tolerance);
}