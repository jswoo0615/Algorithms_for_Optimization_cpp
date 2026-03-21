#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include "Optimization/NestrovMomentum.hpp"
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
namespace Test {

// ==============================================================================
// 1. 10D 다차원 2차 함수 (10-Dimensional Quadratic Function)
// 수식: f(x) = x_0^2 + 2*x_1^2 + 3*x_2^2 + ... + 10*x_9^2
// ==============================================================================
struct HighDimQuadraticFunc {
    template <typename T>
    T operator()(const std::array<T, 10>& x) const {
        T sum = T(0.0);
        for (size_t i = 0; i < 10; ++i) {
            // AD 엔진의 in-place 연산자 부재로 인한 임시 우회 코드
            sum = sum + T(i + 1) * x[i] * x[i]; 
        }
        return sum;
    }
};

// ==============================================================================
// 2. 2D 로젠브록 함수 (Rosenbrock Function)
// 수식: f(x, y) = (1 - x)^2 + 100(y - x^2)^2
// ==============================================================================
struct RosenbrockFunc {
    template <typename T>
    T operator()(const std::array<T, 2>& x) const {
        T term1 = T(1.0) - x[0];
        T term2 = x[1] - x[0] * x[0];
        return term1 * term1 + T(100.0) * term2 * term2;
    }
};

class NestrovMomentumTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ------------------------------------------------------------------------------
// Test Case 1: 10D 고차원 수렴 검증
// ------------------------------------------------------------------------------
TEST_F(NestrovMomentumTest, ConvergesOn10DQuadraticFunction) {
    HighDimQuadraticFunc func;
    std::array<double, 10> initial_guess;
    
    // 시작점을 모두 5.0으로 초기화
    initial_guess.fill(5.0);

    // Nesterov 적용 (Learning Rate: 0.01, 관성: 0.9)
    auto result = Optimization::NestrovMomentum::optimize<10>(
        func, initial_guess, 0.01, 0.9, 10000, 1e-5, false
    );

    // 10개 차원 모두 0.0으로 수렴했는지 확인 (허용 오차 1e-4)
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(result[i], 0.0, 1e-4) << "Dimension " << i << " failed to converge.";
    }
}

// ------------------------------------------------------------------------------
// Test Case 2: 2D 로젠브록 함수 검증 (협곡 탈출)
// ------------------------------------------------------------------------------
TEST_F(NestrovMomentumTest, ConvergesOnRosenbrockFunction) {
    RosenbrockFunc func;
    std::array<double, 2> initial_guess = {-1.2, 1.0};

    // 로젠브록 협곡의 느린 수렴(Crawling) 현상을 극복하기 위해
    // 학습률(alpha)을 0.001로 안정화하고, 반복 횟수(max_iter)를 50000으로 증가시킵니다.
    auto result = Optimization::NestrovMomentum::optimize<2>(
        func, initial_guess, 0.001, 0.9, 50000, 1e-5, false
    );

    // 정답 (1.0, 1.0) 수렴 확인
    EXPECT_NEAR(result[0], 1.0, 1e-2) << "X coordinate failed to converge on Rosenbrock.";
    EXPECT_NEAR(result[1], 1.0, 1e-2) << "Y coordinate failed to converge on Rosenbrock.";
}

} // namespace Test
} // namespace Optimization