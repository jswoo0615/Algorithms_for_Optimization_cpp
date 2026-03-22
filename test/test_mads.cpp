#include <gtest/gtest.h>
#include <array>
#include <cmath>

#include "Optimization/MADS.hpp"

namespace Optimization::Tests {

    // 테스트용 2차원 볼록 함수 f(x) = (x_1 - 2)^2 + 3(x_2 + 1)^2
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

TEST(MADSTest, OptimizeEllipticQuadratic) {
    std::array<double, 2> x_init = {-5.0, 5.0};
    
    // MADS 실행
    auto result = MeshAdaptiveDirectSearch::optimize<2>(
        EllipticQuadratic2D{}, x_init, 1e-6, 5000, false
    );

    // 전역 최적점 [2.0, -1.0] 에 수렴해야 함 (0차 무작위 탐색이므로 오차 한계를 1e-3으로 설정)
    EXPECT_NEAR(result.x_opt[0], 2.0, 1e-3);
    EXPECT_NEAR(result.x_opt[1], -1.0, 1e-3);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-4);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}