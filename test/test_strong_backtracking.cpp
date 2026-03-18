#include <gtest/gtest.h>

#include <array>

#include "Optimization/StrongBacktrackingLineSearch.hpp"

using namespace Optimization;

class StrongWolfeTest : public ::testing::Test {
   protected:
    // f(x) = x^2
    static constexpr auto func_1d = [](const auto& x) { return x[0] * x[0]; };

    // f(x, y) = x^2 + 2y^2
    static constexpr auto func_2d = [](const auto& v) { return v[0] * v[0] + 2.0 * v[1] * v[1]; };
};

// 1. 완벽한 초기 보폭 (한 번에 통과)
TEST_F(StrongWolfeTest, PerfectInitialStep) {
    std::array<double, 1> x = {2.0};
    std::array<double, 1> d = {-4.0};  // x=2에서 기울기는 4, 방향은 -4

    // f(x) = x^2 이고 방향이 -4일 때, alpha = 0.5 이면 x_new = 0 이 되어 정확히 최솟값 도착
    // Curvature 검사 통과 여부를 확인!
    double alpha = StrongBacktrackingLineSearch::search<1>(func_1d, x, d, 0.5);
    EXPECT_NEAR(alpha, 0.5, 1e-4);
}

// 2. 초기 보폭이 너무 작은 경우 (Curvature 실패 -> alpha_hi가 없으므로 2배씩 확장)
TEST_F(StrongWolfeTest, StepTooSmall_TriggersExpansion) {
    std::array<double, 1> x = {2.0};
    std::array<double, 1> d = {-4.0};

    // 시작 alpha를 0.01로 아주 작게 주고,
    // c2(곡률 조건)를 0.1로 아주 깐깐하게 설정합니다! (기본값 0.9는 너무 관대함)
    // 0.01, 1e-4(c1), 0.1(c2)
    double alpha = StrongBacktrackingLineSearch::search<1>(func_1d, x, d, 0.01, 1e-4, 0.1);

    EXPECT_GT(alpha, 0.01);        // 확장이 일어났으므로 무조건 커져야 함!
    EXPECT_NEAR(alpha, 0.5, 0.1);  // c2가 깐깐해졌으므로 0.45 ~ 0.55 안착
}

// 3. 초기 보폭이 너무 큰 경우 (Armijo 실패 -> 절반씩 축소)
TEST_F(StrongWolfeTest, StepTooLarge_TriggersShrinking) {
    std::array<double, 1> x = {2.0};
    std::array<double, 1> d = {-4.0};

    // 시작 alpha를 5.0으로 줌 (반대편 언덕으로 튕겨 올라감)
    // Armijo 조건(충분 감소) 실패로 보폭을 계속 깎아내려야 함
    double alpha = StrongBacktrackingLineSearch::search<1>(func_1d, x, d, 5.0);

    EXPECT_LT(alpha, 5.0);         // 초기값보다는 작아져야 함!
    EXPECT_NEAR(alpha, 0.5, 0.2);  // 결국 최적점 근처로 안착
}

// 4. 다변수 함수에서의 Strong Wolfe 테스트
TEST_F(StrongWolfeTest, MultiDimStrongWolfe) {
    std::array<double, 2> x = {2.0, 2.0};
    std::array<double, 2> d = {-4.0, -8.0};  // -gradient 방향

    // 다변수에서도 무사히 축소/확장을 거쳐 적절한 보폭을 찾아내는지 확인
    double alpha = StrongBacktrackingLineSearch::search<2>(func_2d, x, d, 1.0);

    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 1.0);  // 1.0은 너무 멀기 때문에 깎여야 정상
}