#include <gtest/gtest.h>
#include <array>

#include "Optimization/CMA_ES.hpp"

namespace Optimization::Tests {

    // 악명 높은 Rosenbrock Function (협곡 지형)
    struct Rosenbrock2D {
        template <size_t N>
        constexpr double operator()(const std::array<double, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            double d1 = 1.0 - x[0];
            double d2 = x[1] - x[0] * x[0];
            return d1 * d1 + 100.0 * d2 * d2;
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(CMA_ESTest, OptimizeRosenbrockFunction) {
    std::array<double, 2> mu_init = {-1.2, 1.0}; // 협곡 밖의 전형적인 테스트 시작점
    
    // N=2, M=10(샘플 수), M_ELITE=5
    auto result = CMA_ES::optimize<2, 10, 5>(
        Rosenbrock2D{}, mu_init, 0.5, 2000, 1e-4, 12345, false
    );

    // 미분 없이 좁은 협곡을 공분산 행렬로 스스로 적응하며 통과하여 전역 최적점 [1.0, 1.0] 에 도달
    EXPECT_NEAR(result.x_opt[0], 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt[1], 1.0, 1e-2);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-3);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}