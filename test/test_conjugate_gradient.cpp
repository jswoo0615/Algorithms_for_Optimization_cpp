#include <gtest/gtest.h>
#include <array>
#include "Optimization/ConjugateGradient.hpp"

using namespace Optimization;

class ConjugateGradientTest : public ::testing::Test {
protected:
    // 1. 원형 그릇 (2차 함수)
    static constexpr auto quadratic_bowl = [](const auto& v) {
        return v[0]*v[0] + v[1]*v[1];
    };

    // 2. 타원형 그릇 (2차 함수, 한쪽이 가파름)
    static constexpr auto elliptical_bowl = [](const auto& v) {
        return v[0]*v[0] + 4.0 * v[1]*v[1];
    };

    // 3. 악명 높은 로젠브록 함수 (비선형, 바나나 계곡)
    // 이전장에서 std::pow 에러를 겪고 단련된 안전한 곱셈 형태입니다 ㅋㅋㅋㅋ
    static constexpr auto rosenbrock = [](const auto& v) {
        auto term1 = 1.0 - v[0];
        auto term2 = v[1] - v[0]*v[0];
        return term1*term1 + 100.0 * term2*term2;
    };
};

// 테스트 1: 원형 그릇 - 2차원이므로 이론상 최대 2번 만에 수렴해야 함!
TEST_F(ConjugateGradientTest, QuadraticBowl) {
    std::array<double, 2> start = {2.0, 2.0};
    auto result = ConjugateGradient::optimize<2>(quadratic_bowl, start, 100, 1e-4, false);
    
    EXPECT_NEAR(result[0], 0.0, 1e-3);
    EXPECT_NEAR(result[1], 0.0, 1e-3);
}

// 테스트 2: 타원형 그릇 - 일반 GD는 여기서 지그재그로 헤매지만, CG는 직진합니다!
TEST_F(ConjugateGradientTest, EllipticalBowl) {
    std::array<double, 2> start = {2.0, 2.0};
    auto result = ConjugateGradient::optimize<2>(elliptical_bowl, start, 100, 1e-4, false);
    
    EXPECT_NEAR(result[0], 0.0, 1e-3);
    EXPECT_NEAR(result[1], 0.0, 1e-3);
}

// 테스트 3: 로젠브록 함수 - 비선형에서도 GD보다 훨씬 적은 횟수로 바닥을 찾습니다.
TEST_F(ConjugateGradientTest, Rosenbrock) {
    std::array<double, 2> start = {-1.2, 1.0};
    
    // CG는 GD보다 압도적으로 빠르므로 2000번의 기회면 차고 넘칩니다.
    auto result = ConjugateGradient::optimize<2>(rosenbrock, start, 2000, 1e-4, false);
    
    EXPECT_NEAR(result[0], 1.0, 1e-2);
    EXPECT_NEAR(result[1], 1.0, 1e-2);
}