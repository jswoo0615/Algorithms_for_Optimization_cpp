#include <gtest/gtest.h>
#include <array>
#include <cmath>

#include "Optimization/GeneralizedPatternSearch.hpp"

namespace Optimization::Tests {

    // 테스트용 타원형 2차 목적 함수
    // f(x) = (x_1 - 2.0)^2 + 3.0 * (x_2 + 1.0)^2
    struct EllipticQuadratic2D {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            T d1 = x[0] - T(2.0);
            T d2 = x[1] + T(1.0);
            return (d1 * d1) + T(3.0) * (d2 * d2);
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(GeneralizedPatternSearchTest, OptimizeWithCompassDirections) {
    std::array<double, 2> x_init = {-5.0, 5.0};
    
    // N=2, M=4 : 2차원 공간을 위한 양의 생성 집합 (Positive Spanning Set)
    // 상, 하, 좌, 우의 Compass 방향 설정
    std::array<std::array<double, 2>, 4> D = {{
        {1.0, 0.0},
        {-1.0, 0.0},
        {0.0, 1.0},
        {0.0, -1.0}
    }};
    
    auto result = GeneralizedPatternSearch::optimize<2, 4>(
        EllipticQuadratic2D{}, x_init, D, 1.0, 1e-6, 0.5, 2000, false
    );

    // 전역 최적점 [2.0, -1.0] 에 도달해야 함
    EXPECT_NEAR(result.x_opt[0], 2.0, 1e-4);
    EXPECT_NEAR(result.x_opt[1], -1.0, 1e-4);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-4);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}