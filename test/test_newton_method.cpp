#include <gtest/gtest.h>
#include <array>

#include "Optimization/NewtonMethod.hpp"

namespace Optimization::Tests {

    // 템플릿화된 Sphere Function (AutoDiff 호환)
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

TEST(NewtonMethodTest, Quadratic2DOptimization) {
    std::array<double, 2> x_init = {5.0, -3.0};

    auto result = NewtonMethod::optimize<2>(SphereFunction{}, x_init, 1e-6, 50);

    // 2D 환경에서는 Cramer's Rule이 작동합니다.
    // 중앙 차분법(Central Difference)으로 헤시안을 근사하므로, 허용 오차를 1e-5로 설정합니다.
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-5);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-5);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-5);

    EXPECT_GT(result.elapsed_ns, 0); // 연산 시간이 측정되었는지 검증
}

TEST(NewtonMethodTest, Quadratic3DOptimization) {
    std::array<double, 3> x_init = {5.0, -3.0, 2.5};

    auto result = NewtonMethod::optimize<3>(SphereFunction{}, x_init, 1e-6, 50);

    // 3D 환경에서는 정적 가우스 소거법(Gauss Elimination)이 작동합니다.
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-5);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-5);
    EXPECT_NEAR(result.x_opt[2], 0.0, 1e-5);
}