#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "Optimization/LineSearch.hpp"

using namespace Optimization;

class LineSearchTest : public ::testing::Test {
   protected:
    // 테스트용 N차원 (2차원) 스칼라 함수
    // f(x1, x2) = x1^2 + x2^2 + 2*x1 + 4
    // 최솟값 위치: (-1, 0), 이때 f(-1, 0) = 3
    static constexpr auto test_func = [](const auto& v) {
        return v[0] * v[0] + v[1] * v[1] + 2.0 * v[0] + 4.0;
    };

    std::array<double, 2> x_start;
    std::array<double, 2> direction;

    void SetUp() override {
        // (0, 0)에서 시작해서 (-1, 0) 방향으로 직진!
        // 정확히 alpha = 1.0 일 때 최솟값에 도달해야 합니다.
        x_start = {0.0, 0.0};
        direction = {-1.0, 0.0};
    }
};

// 1. Bracket Minimum 테스트
TEST_F(LineSearchTest, BracketMinimumTest) {
    auto bracket = LineSearch::bracket_minimum<2>(test_func, x_start, direction);

    double lower = std::min(bracket.first, bracket.second);
    double upper = std::max(bracket.first, bracket.second);

    // 정답인 alpha = 1.0 이 반환된 구간 [lower, upper] 안에 있어야 함
    EXPECT_LE(lower, 1.0);
    EXPECT_GE(upper, 1.0);
}

// 2. Golden Section Search 테스트
TEST_F(LineSearchTest, GoldenSectionSearchTest) {
    // 0.0 ~ 2.0 구간에서 탐색
    double alpha_opt =
        LineSearch::golden_section_search<2>(test_func, x_start, direction, 0.0, 2.0);
    EXPECT_NEAR(alpha_opt, 1.0, 1e-4);
}

// 3. Quadratic Fit Search 테스트
TEST_F(LineSearchTest, QuadraticFitSearchTest) {
    // 세 점 a, b, c (0.0, 0.5, 2.0)를 주고 2차 함수로 근사해서 찾기
    // 우리가 준 테스트 함수 자체가 완벽한 2차 함수이므로, 단숨에 1.0을 찾아야 함!
    double alpha_opt =
        LineSearch::quadratic_fit_search<2>(test_func, x_start, direction, 0.0, 0.5, 2.0);
    EXPECT_NEAR(alpha_opt, 1.0, 1e-5);
}

// 4. Shubert-Piyavskii 전역 최적화 테스트
TEST_F(LineSearchTest, ShubertPiyavskiiTest) {
    double L = 5.0;
    // 톨러런스를 파라미터로 명시적으로 전달 (y 허용 오차: 1e-4)
    double alpha_opt =
        LineSearch::shubert_piyavskii<2>(test_func, x_start, direction, 0.0, 3.0, L, 1e-4);

    // 포물선의 평평한 바닥 특성을 고려하여, alpha(x축)의 허용 오차를 1e-2(0.01)로 조정!
    EXPECT_NEAR(alpha_opt, 1.0, 1e-2);
}

// 5. Bracket Sign Change 테스트 (도함수 부호 바뀌는 구간)
TEST_F(LineSearchTest, BracketSignChangeTest) {
    // 0.0과 0.5에서 시작. 정답인 1.0을 포함할 때까지 구간을 확장해야 함
    auto bracket = LineSearch::bracket_sign_change<2>(test_func, x_start, direction, 0.0, 0.5);

    double lower = std::min(bracket.first, bracket.second);
    double upper = std::max(bracket.first, bracket.second);

    EXPECT_LE(lower, 1.0);
    EXPECT_GE(upper, 1.0);
}

// 6. Bisection Method 테스트
TEST_F(LineSearchTest, BisectionTest) {
    // [0.0, 3.0] 사이에서 1차 미분이 0이 되는 지점(alpha=1.0)을 이분법으로 탐색
    double alpha_opt = LineSearch::bisection<2>(test_func, x_start, direction, 0.0, 3.0);
    EXPECT_NEAR(alpha_opt, 1.0, 1e-4);
}