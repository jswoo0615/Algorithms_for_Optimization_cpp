#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"

using namespace Optimization;

class AutoDiffTest : public ::testing::Test {
   protected:
    // 테스트에 사용할 스칼라 함수 f(x, y) = x^2 + 3xy + y^3
    // x = 2, y = 3 일 때:
    // Value = 4 + 18 + 27 = 49
    // Gradient = [2x + 3y, 3x + 3y^2] = [13, 33]
    static constexpr auto scalar_func = [](const auto& x) {
        // x(0)의 타입을 추출하여 실수(double)와 Dual 타입을 유연하게 처리
        using T = std::decay_t<decltype(x(0))>;
        return x(0) * x(0) + T(3.0) * x(0) * x(1) + x(1) * x(1) * x(1);
    };

    // 테스트에 사용할 벡터 함수 F(x, y) = [x^2 + y, 5x - y^2]^T
    // x = 2, y = 3 일 때:
    // Jacobian = [[2x, 1], [5, -2y]] = [[4, 1], [5, -6]]
    static constexpr auto vector_func = [](const auto& x) {
        using T = std::decay_t<decltype(x(0))>;
        StaticVector<T, 2> res;
        res(0) = x(0) * x(0) + x(1);
        res(1) = T(5.0) * x(0) - x(1) * x(1);
        return res;
    };
};

// 1. Value 함수 테스트
TEST_F(AutoDiffTest, ValueTest) {
    StaticVector<double, 2> x;
    x(0) = 2.0;
    x(1) = 3.0;

    double val = AutoDiff::value<2>(scalar_func, x);
    EXPECT_DOUBLE_EQ(val, 49.0);
}

// 2. Gradient 함수 테스트
TEST_F(AutoDiffTest, GradientTest) {
    StaticVector<double, 2> x;
    x(0) = 2.0;
    x(1) = 3.0;

    auto grad = AutoDiff::gradient<2>(scalar_func, x);

    EXPECT_DOUBLE_EQ(grad(0), 13.0);
    EXPECT_DOUBLE_EQ(grad(1), 33.0);
}

// 3. Value and Gradient 동시 추출 테스트
TEST_F(AutoDiffTest, ValueAndGradientTest) {
    StaticVector<double, 2> x;
    x(0) = 2.0;
    x(1) = 3.0;

    double val = 0.0;
    StaticVector<double, 2> grad;
    grad.set_zero();

    AutoDiff::value_and_gradient<2>(scalar_func, x, val, grad);

    EXPECT_DOUBLE_EQ(val, 49.0);
    EXPECT_DOUBLE_EQ(grad(0), 13.0);
    EXPECT_DOUBLE_EQ(grad(1), 33.0);
}

// 4. Jacobian 행렬 테스트
TEST_F(AutoDiffTest, JacobianTest) {
    StaticVector<double, 2> x;
    x(0) = 2.0;
    x(1) = 3.0;

    // M=2(출력차원), N=2(입력차원)
    auto J = AutoDiff::jacobian<2, 2>(vector_func, x);

    EXPECT_DOUBLE_EQ(J(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(J(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(J(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(J(1, 1), -6.0);
}

// 5. Hessian 행렬 테스트는 아키텍처 결단에 따라 폐기되었습니다.
// SQP 루프 내부에서 BFGS 업데이트를 통해 헤시안을 근사할 예정입니다.