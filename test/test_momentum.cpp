#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/Momentum.hpp"

namespace Optimization {
namespace Test {

// ==============================================================================
// 1. 단순 2차 볼록 함수 (Quadratic Convex Function)
// 수식: f(x, y) = x^2 + 2y^2
// 특징: 가장 기본적인 볼록 함수. 정답은 (0, 0)
// ==============================================================================
struct QuadraticFunc {
    template <typename T>
    T operator()(const std::array<T, 2>& x) const {
        return x[0] * x[0] + T(2.0) * x[1] * x[1];
    }
};

// ==============================================================================
// 2. 로젠브록 함수 (Rosenbrock Function)
// 수식: f(x, y) = (1 - x)^2 + 100(y - x^2)^2
// 특징: 좁고 휘어진 골짜기를 가진 비선형 최적화 표준 벤치마크. 정답은 (1, 1)
// Momentum의 관성(beta) 효과를 검증하기에 최적화된 테스트.
// ==============================================================================
struct RosenbrockFunc {
    template <typename T>
    T operator()(const std::array<T, 2>& x) const {
        T term1 = T(1.0) - x[0];
        T term2 = x[1] - x[0] * x[0];
        return term1 * term1 + T(100.0) * term2 * term2;
    }
};

class MomentumOptimizerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 테스트 전역 설정이 필요한 경우 (현재는 불필요)
    }
    void TearDown() override {
        // 메모리 해제 등 (정적 할당만 사용하므로 불필요)
    }
};

// ------------------------------------------------------------------------------
// Test Case 1: Quadratic Function 수렴 검증
// ------------------------------------------------------------------------------
TEST_F(MomentumOptimizerTest, ConvergesOnQuadraticFunction) {
    QuadraticFunc func;
    std::array<double, 2> initial_guess = {5.0, -4.0};

    // Momentum 적용 (Learning Rate: 0.05, 관성: 0.9)
    auto result =
        Optimization::Momentum::optimize<2>(func, initial_guess, 0.05, 0.9, 5000, 1e-6, false);

    // 허용 오차 1e-4 내에서 (0, 0)으로 수렴하는지 확인
    EXPECT_NEAR(result[0], 0.0, 1e-4) << "X coordinate failed to converge.";
    EXPECT_NEAR(result[1], 0.0, 1e-4) << "Y coordinate failed to converge.";
}

// ------------------------------------------------------------------------------
// Test Case 2: Rosenbrock Function (비선형 골짜기) 수렴 검증
// ------------------------------------------------------------------------------
TEST_F(MomentumOptimizerTest, ConvergesOnRosenbrockFunction) {
    RosenbrockFunc func;
    std::array<double, 2> initial_guess = {-1.2, 1.0};

    // 로젠브록은 골짜기가 좁아 보폭(alpha)을 줄이고 반복 횟수(max_iter)를 늘려야 함
    auto result =
        Optimization::Momentum::optimize<2>(func, initial_guess, 0.001, 0.9, 20000, 1e-5, false);

    // 허용 오차 1e-2 내에서 (1, 1)로 수렴하는지 확인 (비선형성이 강하므로 tolerance 조정)
    EXPECT_NEAR(result[0], 1.0, 1e-2) << "X coordinate failed to converge on Rosenbrock.";
    EXPECT_NEAR(result[1], 1.0, 1e-2) << "Y coordinate failed to converge on Rosenbrock.";
}

}  // namespace Test
}  // namespace Optimization