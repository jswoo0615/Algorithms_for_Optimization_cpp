#include <gtest/gtest.h>
#include <array>

#include "Optimization/CyclicCoordinate.hpp"

namespace Optimization::Tests {

    // 템플릿화된 Convex Quadratic Function (AutoDiff 호환)
    struct ConvexQuadratic2D {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            return x[0] * x[0] + T(2.0) * x[1] * x[1];
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(CyclicCoordinateTest, BasicSearchConvergesWithGoldenSection) {
    std::array<double, 2> x_init = {5.0, -3.0};
    
    // GoldenSectionStrategy가 기본값(Default)으로 적용되어 LineSearch::golden_section_search를 호출합니다.
    auto result = CyclicCoordinateSearch::optimize<2>(
        ConvexQuadratic2D{}, x_init, GoldenSectionStrategy{}, 1e-4, 500, false
    );

    // 미분값 없이 Bracket -> Golden Section만으로 수렴하는지 확인
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-3);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-3);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-4);
    EXPECT_GT(result.elapsed_ns, 0);
}

TEST(CyclicCoordinateTest, AcceleratedSearchConvergesWithGoldenSection) {
    std::array<double, 2> x_init = {5.0, -3.0};
    
    auto result = CyclicCoordinateSearch::optimize_accelerated<2>(
        ConvexQuadratic2D{}, x_init, GoldenSectionStrategy{}, 1e-4, 500, false
    );

    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-3);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-3);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-4);
    EXPECT_GT(result.elapsed_ns, 0);
}