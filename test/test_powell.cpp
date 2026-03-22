#include <gtest/gtest.h>

#include <array>

#include "Optimization/Powell.hpp"

namespace Optimization::Tests {

// 템플릿화된 Convex Quadratic Function
// f(x) = x_1^2 + 2x_2^2
struct ConvexQuadratic2D {
    template <typename T, size_t N>
    constexpr T operator()(const std::array<T, N>& x) const noexcept {
        static_assert(N == 2, "Dimension must be 2.");
        return x[0] * x[0] + T(2.0) * x[1] * x[1];
    }
};

}  // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(PowellMethodTest, OptimizeWithGoldenSection) {
    std::array<double, 2> x_init = {5.0, -3.0};

    // GoldenSectionStrategy가 기본값으로 적용되어 미분 없이 최적화를 수행
    auto result = PowellMethod::optimize<2>(ConvexQuadratic2D{}, x_init, GoldenSectionStrategy{},
                                            1e-5, 500, false);

    // 파월의 방법이 최소점으로 올바르게 수렴했는지 확인
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-3);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-3);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-4);

    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}