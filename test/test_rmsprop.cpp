#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/RMSProp.hpp"

namespace Optimization {
namespace Test {

// ==============================================================================
// 1. 10D 다차원 2차 함수 (10-Dimensional Quadratic Function)
// ==============================================================================
struct HighDimQuadraticFunc {
    template <typename T>
    T operator()(const std::array<T, 10>& x) const {
        T sum = T(0.0);
        for (size_t i = 0; i < 10; ++i) {
            sum = sum + T(i + 1) * x[i] * x[i];
        }
        return sum;
    }
};

// ==============================================================================
// 2. 2D 로젠브록 함수 (Rosenbrock Function)
// 특징: RMSProp의 지수 이동 평균(decay)이 협곡 탈출에 얼마나 효과적인지 검증합니다.
// ==============================================================================
struct RosenbrockFunc {
    template <typename T>
    T operator()(const std::array<T, 2>& x) const {
        T term1 = T(1.0) - x[0];
        T term2 = x[1] - x[0] * x[0];
        return term1 * term1 + T(100.0) * term2 * term2;
    }
};

class RMSPropTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ------------------------------------------------------------------------------
// Test Case 1: 10D 고차원 수렴 검증
// ------------------------------------------------------------------------------
TEST_F(RMSPropTest, ConvergesOn10DQuadraticFunction) {
    HighDimQuadraticFunc func;
    std::array<double, 10> initial_guess;
    initial_guess.fill(5.0);

    // RMSProp 적용 (학습률 0.1, decay 0.9)
    auto result = Optimization::RMSProp::optimize<10>(func, initial_guess, 0.1, 0.9, 1e-8, 15000,
                                                      1e-4, false);

    // 10개 차원 모두 0.0으로 수렴했는지 확인 (허용 오차 1e-3)
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(result[i], 0.0, 1e-3) << "Dimension " << i << " failed to converge.";
    }
}

// ------------------------------------------------------------------------------
// Test Case 2: 2D 로젠브록 함수 검증 (협곡 쾌속 탈출)
// ------------------------------------------------------------------------------
TEST_F(RMSPropTest, ConvergesOnRosenbrockFunction) {
    RosenbrockFunc func;
    std::array<double, 2> initial_guess = {-1.2, 1.0};

    // [입법자의 튜닝]
    // alpha를 0.002로 대폭 줄여 절벽 사이의 진동을 막고,
    // decay를 0.99로 늘려 이동 관성을 안정적으로 유지합니다.
    auto result = Optimization::RMSProp::optimize<2>(func, initial_guess, 0.002, 0.99, 1e-8, 50000,
                                                     1e-4, false);

    // 정답 (1.0, 1.0) 수렴 확인
    EXPECT_NEAR(result[0], 1.0, 1e-2) << "X coordinate failed to converge on Rosenbrock.";
    EXPECT_NEAR(result[1], 1.0, 1e-2) << "Y coordinate failed to converge on Rosenbrock.";
}

}  // namespace Test
}  // namespace Optimization