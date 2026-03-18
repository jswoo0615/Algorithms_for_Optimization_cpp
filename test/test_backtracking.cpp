#include <gtest/gtest.h>

#include <array>

#include "Optimization/BacktrackingLineSearch.hpp"

using namespace Optimization;

class BacktrackingTest : public ::testing::Test {
   protected:
    // 1차원 테스트 함수: f(x) = x^2
    static constexpr auto func_1d = [](const auto& x) { return x[0] * x[0]; };

    // 2차원 테스트 함수: f(x, y) = x^2 + y^2
    static constexpr auto func_2d = [](const auto& v) { return v[0] * v[0] + v[1] * v[1]; };
};

// 1. 1차원 정상 하강 테스트 (Backtracking 불필요)
TEST_F(BacktrackingTest, NoBacktrackNeeded) {
    std::array<double, 1> x = {2.0};   // 현재 위치: x = 2
    std::array<double, 1> d = {-1.0};  // 탐색 방향: 음수 방향 (내리막)

    // 기울기: 4.0, 방향 도함수: -4.0
    // alpha = 1.0 이면 x_new = 1.0, f_new = 1.0
    // 타겟값 = 4.0 + 1e-4 * 1.0 * (-4.0) = 3.9996
    // f_new(1.0) <= 3.9996 이므로 즉시 통과해야 함!
    double alpha = BacktrackingLineSearch::search<1>(func_1d, x, d, 1.0);
    EXPECT_DOUBLE_EQ(alpha, 1.0);
}

// 2. 1차원 백트래킹 발생 테스트 (Overshoot)
TEST_F(BacktrackingTest, BacktrackOccurs) {
    std::array<double, 1> x = {2.0};
    std::array<double, 1> d = {
        -10.0};  // 일부러 방향 스케일을 엄청 크게 줌 (반대편 언덕으로 날아감)

    // 과정 추적:
    // f(x) = 4.0, 기울기 = 4.0, 방향 도함수 = -40.0
    // [Iter 1] alpha = 1.0 -> x_new = -8.0, f_new = 64.0 (증가함! 실패)
    // [Iter 2] alpha = 0.5 -> x_new = -3.0, f_new = 9.0 (증가함! 실패)
    // [Iter 3] alpha = 0.25 -> x_new = -0.5, f_new = 0.25 (감소함! 성공)
    // 따라서 최종 alpha는 0.25가 나와야 합니다.
    double alpha = BacktrackingLineSearch::search<1>(func_1d, x, d, 1.0, 0.5);
    EXPECT_DOUBLE_EQ(alpha, 0.25);
}

// 3. 2차원 다변수 함수 백트래킹 테스트
TEST_F(BacktrackingTest, MultiDimBacktrack) {
    std::array<double, 2> x = {2.0, 2.0};    // 현재 위치: (2, 2), 함수값 = 8.0
    std::array<double, 2> d = {-4.0, -4.0};  // 과도하게 큰 하강 방향

    // 과정 추적:
    // 기울기 = (4, 4), 방향 도함수 = -16 - 16 = -32.0
    // [Iter 1] alpha = 1.0 -> x_new = (-2, -2), f_new = 8.0 (원래랑 똑같음, 충분한 감소 아님. 실패)
    // [Iter 2] alpha = 0.5 -> x_new = (0, 0), f_new = 0.0 (충분히 감소함! 성공)
    // 따라서 최종 alpha는 0.5가 나와야 합니다.
    double alpha = BacktrackingLineSearch::search<2>(func_2d, x, d, 1.0, 0.5);
    EXPECT_DOUBLE_EQ(alpha, 0.5);
}

// 4. (보너스) Failsafe 테스트 - 하강 방향이 아닌 경우 (오르막)
TEST_F(BacktrackingTest, FailsafeOnAscentDirection) {
    std::array<double, 1> x = {2.0};
    std::array<double, 1> d = {1.0};  // 오르막 방향!

    // 오르막이므로 아무리 alpha를 줄여도 함수값은 항상 타겟보다 큼.
    // 결국 alpha가 1e-10 밑으로 떨어지며 Failsafe가 발동해야 함.
    double alpha = BacktrackingLineSearch::search<1>(func_1d, x, d, 1.0);
    EXPECT_LT(alpha, 1e-10);  // alpha가 매우 작은 값이어야 함
}