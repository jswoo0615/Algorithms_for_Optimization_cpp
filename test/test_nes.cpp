#include <gtest/gtest.h>
#include <array>

#include "Optimization/NaturalEvoluationStrategies.hpp"

namespace Optimization::Tests {

    // 테스트용 타원형 2차 함수: f(x) = (x_1 - 3)^2 + 10(x_2 + 2)^2
    struct Elliptic2D {
        template <size_t N>
        constexpr double operator()(const std::array<double, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            double d1 = x[0] - 3.0;
            double d2 = x[1] + 2.0;
            return d1 * d1 + 10.0 * d2 * d2;
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(NESTest, OptimizeEllipticFunction) {
    std::array<double, 2> mu_init = {0.0, 0.0};
    std::array<double, 2> sigma_sq_init = {5.0, 5.0}; // 초기 탐색 범위
    
    // M=200, alpha=0.05 로 설정하여 300번 반복
    auto result = NaturalEvolutionStrategies::optimize<2, 200>(
        Elliptic2D{}, mu_init, sigma_sq_init, 0.05, 300, 12345, false
    );

    // 전역 최적점 [3.0, -2.0] 에 수렴해야 함
    EXPECT_NEAR(result.x_opt[0], 3.0, 1e-1);
    EXPECT_NEAR(result.x_opt[1], -2.0, 1e-1);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-1);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}