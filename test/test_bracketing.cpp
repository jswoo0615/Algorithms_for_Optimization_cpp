#include <gtest/gtest.h>

#include "Optimization/Bracketing.hpp"

using namespace Optimization;

class BracketingTest : public ::testing::Test {
   protected:
    // 일반 스칼라 함수 f(x) = (x - 2)^2 + 1
    // x = 2 에서 최소값 1을 가짐
    static constexpr auto scalar_func = [](double x) { return (x - 2.0) * (x - 2.0) + 1.0; };

    // Bisection Method를 위한 Dual 넘버 전용 함수
    static constexpr auto dual_func = [](const Dual<double>& x) {
        return (x - 2.0) * (x - 2.0) + 1.0;
    };
};

// 1. Bracket Minimum 테스트 (최소값이 구간 안에 잘 들어오는지 확인)
TEST_F(BracketingTest, BracketMinimumTest) {
    // x = 0 에서 시작
    Range r = bracket_minimum(scalar_func, 0.0);

    // a와 b가 순서대로 정렬되어 있다고 보장할 수 없으므로 min/max 처리
    double lower = std::min(r.a, r.b);
    double upper = std::max(r.a, r.b);

    // 최소점 x=2 가 구간 [lower, upper] 안에 있어야 함
    EXPECT_LE(lower, 2.0);
    EXPECT_GE(upper, 2.0);
}

// 2. Golden Section Search 테스트
TEST_F(BracketingTest, GoldenSectionSearchTest) {
    Range r = {0.0, 5.0};  // 0 ~ 5 사이에서 탐색
    int iterations = 50;   // 50번 반복

    double x_opt = golden_section_search(scalar_func, r, iterations);

    // 찾은 x값이 정답 2.0과 거의 일치하는지 확인
    EXPECT_NEAR(x_opt, 2.0, 1e-5);
}

// 3. Bisection Method 테스트 (미분 정보 활용)
TEST_F(BracketingTest, BisectionMethodTest) {
    Range r = {0.0, 5.0};
    int iterations = 50;

    double x_opt = bisection_method(dual_func, r, iterations);

    EXPECT_NEAR(x_opt, 2.0, 1e-5);
}

// 4. Shubert-Piyavskii 전역 최적화 테스트
TEST_F(BracketingTest, ShubertPiyavskiiTest) {
    Range r = {0.0, 5.0};

    // 립시츠 상수 l 설정
    // f'(x) = 2(x-2) 이고, 구간 [0,5]에서 최대 기울기는 x=5일 때 6임.
    // 따라서 립시츠 상수 l은 6보다 큰 7.0 정도로 잡으면 안전함.
    double l = 7.0;
    double eps = 1e-4;
    int max_iter = 100;

    // 주의: Shubert-Piyavskii는 x값이 아니라 "함수의 최소값(y)"을 반환하도록 구현됨
    double f_min = shubert_piyavskii(scalar_func, r, l, eps, max_iter);

    // 최소값인 1.0과 일치하는지 확인
    EXPECT_NEAR(f_min, 1.0, 1e-3);
}