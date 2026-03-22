#include <gtest/gtest.h>
#include <array>

#include "Optimization/BFGS.hpp"

namespace Optimization::Tests {

    // 템플릿화된 Rosenbrock Function (AutoDiff 호환용)
    struct RosenbrockFunction {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            static_assert(N == 2, "Rosenbrock is 2D function in this test.");
            const T d1 = T(1.0) - x[0];
            const T d2 = x[1] - (x[0] * x[0]);
            return (d1 * d1) + T(100.0) * (d2 * d2);
        }
    };

    // 템플릿화된 Sphere Function
    struct SphereFunction {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            T sum = T(0.0);
            for (size_t i = 0; i < N; ++i) {
                sum = sum + (x[i] * x[i]);
            }
            return sum;
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(BFGSTest, OptimizeSphereFunction) {
    std::array<double, 3> x_init = {5.0, -3.0, 2.5};
    
    auto result = BFGS::optimize<3>(SphereFunction{}, x_init, 1e-6, 100);

    // BFGS는 2차 함수(Convex)에 대해 매우 빠르게 수렴해야 함
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-4);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-4);
    EXPECT_NEAR(result.x_opt[2], 0.0, 1e-4);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-5);
    
    EXPECT_GT(result.elapsed_ns, 0);
}

TEST(BFGSTest, OptimizeRosenbrockFunction) {
    std::array<double, 2> x_init = {-1.2, 1.0}; // 악명 높은 시작점
    
    auto result = BFGS::optimize<2>(RosenbrockFunction{}, x_init, 1e-6, 500);

    // 전역 최적점 [1.0, 1.0] 검증
    EXPECT_NEAR(result.x_opt[0], 1.0, 1e-3);
    EXPECT_NEAR(result.x_opt[1], 1.0, 1e-3);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-5);
}