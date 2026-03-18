#include <gtest/gtest.h>

#include <array>

#include "Optimization/TrustRegion.hpp"

using namespace Optimization;

class TrustRegionTest : public ::testing::Test {
   protected:
    // 1. 원형 그릇 (가장 쉬운 형태)
    // f(x, y) = x^2 + y^2, 최솟값 (0, 0)
    static constexpr auto quadratic_bowl = [](const auto& v) { return v[0] * v[0] + v[1] * v[1]; };

    // 2. 타원형 그릇 (한 쪽이 더 가파름)
    // f(x, y) = x^2 + 4y^2, 최솟값 (0, 0)
    static constexpr auto elliptical_bowl = [](const auto& v) {
        return v[0] * v[0] + 4.0 * v[1] * v[1];
    };

    // 3. 로젠브록 함수 (최적화계의 바나나 밸리)
    // f(x, y) = (1-x)^2 + 100(y-x^2)^2, 최솟값 (1, 1)
    // 좁고 평평한 곡선 골짜기를 따라가야 해서 웬만한 알고리즘은 여기서 죽습니다.
    static constexpr auto rosenbrock = [](const auto& v) {
        auto term1 = 1.0 - v[0];
        auto term2 = v[1] - v[0] * v[0];
        return term1 * term1 + 100.0 * term2 * term2;
    };
};

// 테스트 1: 원형 그릇
TEST_F(TrustRegionTest, QuadraticBowl) {
    std::array<double, 2> x_start = {2.0, 2.0};
    // 원형은 뉴턴 스텝 한두 번이면 끝나야 정상입니다.
    auto x_opt = TrustRegion::optimize(quadratic_bowl, x_start);

    EXPECT_NEAR(x_opt[0], 0.0, 1e-4);
    EXPECT_NEAR(x_opt[1], 0.0, 1e-4);
}

// 테스트 2: 타원형 그릇
TEST_F(TrustRegionTest, EllipticalBowl) {
    std::array<double, 2> x_start = {2.0, 2.0};
    auto x_opt = TrustRegion::optimize(elliptical_bowl, x_start);

    EXPECT_NEAR(x_opt[0], 0.0, 1e-4);
    EXPECT_NEAR(x_opt[1], 0.0, 1e-4);
}

// 테스트 3: 악명 높은 로젠브록 뚫기
TEST_F(TrustRegionTest, Rosenbrock) {
    // 전통적인 로젠브록 시작점 (-1.2, 1.0)
    std::array<double, 2> x_start = {-1.2, 1.0};

    // 이 지형은 매우 험난하므로, 반경(Delta) 조절이 빈번하게 일어납니다.
    // max_iter를 2000으로 넉넉하게 주고 테스트합니다. (로깅을 켜서 구경해도 재밌습니다!)
    auto x_opt = TrustRegion::optimize(rosenbrock, x_start, 2.0, 2000, false);

    EXPECT_NEAR(x_opt[0], 1.0, 1e-3);
    EXPECT_NEAR(x_opt[1], 1.0, 1e-3);
}