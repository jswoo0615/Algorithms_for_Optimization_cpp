#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "Optimization/Adam.hpp"
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
namespace Test {

// ==============================================================================
// 1. 10D 다차원 2차 함수 (10-Dimensional Quadratic Function)
// 특징: 차원별 곡률이 다른 환경에서 Adam의 2차 모멘트(v)가 학습률을
//       어떻게 자동 스케일링하는지 검증합니다.
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
// 특징: AdaGrad의 '학습률 소멸'과 RMSProp의 '지그재그 진동(Oscillation)'을
//       Adam의 1차 모멘트(관성, m)가 어떻게 돌파하는지 검증합니다.
// ==============================================================================
struct RosenbrockFunc {
    template <typename T>
    T operator()(const std::array<T, 2>& x) const {
        T term1 = T(1.0) - x[0];
        T term2 = x[1] - x[0] * x[0];
        return term1 * term1 + T(100.0) * term2 * term2;
    }
};

class AdamTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ------------------------------------------------------------------------------
// Test Case 1: 10D 고차원 수렴 검증
// ------------------------------------------------------------------------------
TEST_F(AdamTest, ConvergesOn10DQuadraticFunction) {
    HighDimQuadraticFunc func;
    std::array<double, 10> initial_guess;
    initial_guess.fill(5.0);

    // Adam 기본 논문 하이퍼파라미터 세팅 (alpha=0.1로 약간 상향)
    auto result = Optimization::Adam::optimize<10>(func, initial_guess, 0.1, 0.9, 0.999, 1e-8,
                                                   15000, 1e-4, false);

    // 10개 차원 모두 0.0으로 수렴했는지 확인 (허용 오차 1e-3)
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(result[i], 0.0, 1e-3) << "Dimension " << i << " failed to converge.";
    }
}

// ------------------------------------------------------------------------------
// Test Case 2: 2D 로젠브록 함수 검증 (최적 궤적 산출)
// ------------------------------------------------------------------------------
TEST_F(AdamTest, ConvergesOnRosenbrockFunction) {
    RosenbrockFunc func;
    std::array<double, 2> initial_guess = {-1.2, 1.0};

    // RMSProp에서 진동을 막기 위해 0.002까지 내렸던 alpha를
    // Adam에서는 관성(m) 덕분에 0.05까지 끌어올려도 안정적으로 수렴합니다.
    auto result = Optimization::Adam::optimize<2>(func, initial_guess, 0.05, 0.9, 0.999, 1e-8,
                                                  30000, 1e-4, false);

    // 정답 (1.0, 1.0) 수렴 확인
    EXPECT_NEAR(result[0], 1.0, 1e-2) << "X coordinate failed to converge on Rosenbrock.";
    EXPECT_NEAR(result[1], 1.0, 1e-2) << "Y coordinate failed to converge on Rosenbrock.";
}

}  // namespace Test
}  // namespace Optimization