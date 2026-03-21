#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "Optimization/AdaGrad.hpp"
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
namespace Test {

// ==============================================================================
// 1. 10D 다차원 2차 함수 (10-Dimensional Quadratic Function)
// 특징: 각 차원별로 곡률(기울기)이 다릅니다. AdaGrad가 차원별로 학습률을
//       알아서 튜닝(Adaptive)하여 정답(0)을 찾아가는지 검증합니다.
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
// 특징: AdaGrad의 치명적 단점인 '학습률 소멸(Learning Rate Decay)'을 저격합니다.
//       협곡을 따라가는 동안 기울기 제곱합(G)이 무한히 누적되어 보폭이 0에 수렴합니다.
// ==============================================================================
struct RosenbrockFunc {
    template <typename T>
    T operator()(const std::array<T, 2>& x) const {
        T term1 = T(1.0) - x[0];
        T term2 = x[1] - x[0] * x[0];
        return term1 * term1 + T(100.0) * term2 * term2;
    }
};

class AdaGradTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ------------------------------------------------------------------------------
// Test Case 1: 10D 고차원 수렴 검증 (AdaGrad의 장점 확인)
// ------------------------------------------------------------------------------
TEST_F(AdaGradTest, ConvergesOn10DQuadraticFunction) {
    HighDimQuadraticFunc func;
    std::array<double, 10> initial_guess;
    initial_guess.fill(5.0);

    // AdaGrad는 분모가 계속 커지므로 초기 학습률(alpha)을 1.5처럼 아주 크게 잡아줘야 합니다.
    auto result =
        Optimization::AdaGrad::optimize<10>(func, initial_guess, 1.5, 1e-8, 30000, 1e-4, false);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(result[i], 0.0, 1e-3) << "Dimension " << i << " failed to converge.";
    }
}

// ------------------------------------------------------------------------------
// Test Case 2: 2D 로젠브록 함수 검증 (AdaGrad의 한계 확인)
// ------------------------------------------------------------------------------
TEST_F(AdaGradTest, CrawlsOnRosenbrockFunction) {
    RosenbrockFunc func;
    std::array<double, 2> initial_guess = {-1.2, 1.0};

    // 학습률이 죽어버리는 현상을 극복하기 위해 iteration을 무려 10만 번으로 늘리고,
    // 초기 alpha를 2.0으로 강제 주입합니다.
    auto result =
        Optimization::AdaGrad::optimize<2>(func, initial_guess, 2.0, 1e-8, 100000, 1e-4, false);

    // 그럼에도 불구하고 수렴이 굉장히 더디기 때문에 허용 오차(Tolerance)를 5e-2로 넉넉하게
    // 잡습니다.
    EXPECT_NEAR(result[0], 1.0, 5e-2) << "X coordinate failed to converge on Rosenbrock.";
    EXPECT_NEAR(result[1], 1.0, 5e-2) << "Y coordinate failed to converge on Rosenbrock.";
}

}  // namespace Test
}  // namespace Optimization