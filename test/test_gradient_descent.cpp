#include <gtest/gtest.h>

#include "Optimization/GradientDescent.hpp"

using namespace Optimization;

TEST(OptimizerTest, GradientDescentRosenbrock) {
    auto rosenbrock = [](const auto& v) {
        auto term1 = 1.0 - v[0];
        auto term2 = v[1] - v[0] * v[0];
        return term1 * term1 + 100.0 * term2 * term2;
    };

    std::array<double, 2> start = {-1.2, 1.0};
    // 정답은 (1.0, 1.0) 입니다.
    auto result = GradientDescent::optimize<2>(rosenbrock, start, 1e-5, 5000, false);

    EXPECT_NEAR(result[0], 1.0, 1e-3);
    EXPECT_NEAR(result[1], 1.0, 1e-3);
}