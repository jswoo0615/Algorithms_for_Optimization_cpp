#include <gtest/gtest.h>
#include <array>
#include <cmath>

#include "Optimization/HyperGradientDescent.hpp"

namespace Optimization::Tests {

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

    struct RosenbrockFunction {
        template <typename T>
        constexpr T operator()(const std::array<T, 2>& x) const noexcept {
            const T d1 = T(1.0) - x[0];
            const T d2 = x[1] - (x[0] * x[0]);
            return (d1 * d1) + T(100.0) * (d2 * d2);
        }
    };
}

using namespace Optimization;
using namespace Optimization::Tests;

TEST(HyperGradientDescentTest, SphereFunctionOptimization) {
    std::array<double, 3> x_init = {5.0, -3.0, 2.5};

    // Sphere 함수는 안정적이므로 alpha와 mu를 다소 크게 주어도 무방함
    auto result = HyperGradientDescent::optimize<3>(
        SphereFunction{}, x_init, 0.05, 0.01, 1e-5, 5000, false
    );

    for (double val : result) {
        EXPECT_NEAR(val, 0.0, 1e-3);
    }
}

TEST(HyperGradientDescentTest, RosenbrockFunctionOptimization) {
    std::array<double, 2> x_init = {-1.2, 1.0};

    // 튜닝 포인트:
    // Rosenbrock은 급경사 계곡을 가지므로, alpha_init과 mu를 매우 보수적으로 설정해야 합니다.
    // 최대 반복 횟수를 100,000으로 늘려 천천히 수렴하도록 유도합니다.
    auto result = HyperGradientDescent::optimize<2>(
        RosenbrockFunction{}, x_init, 0.0005, 1e-7, 1e-5, 100000, false
    );

    EXPECT_NEAR(result[0], 1.0, 5e-2);
    EXPECT_NEAR(result[1], 1.0, 5e-2);
}