#include <gtest/gtest.h>
#include <array>
#include <cmath>

#include "Optimization/NoisyDescent.hpp"

namespace Optimization::Tests {

    // 템플릿화된 Convex Quadratic Function (AD 호환)
    struct Quadratic2D {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            return x[0] * x[0] + T(2.0) * x[1] * x[1];
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(NoisyDescentTest, ConvergesWithExponentialDecayNoise) {
    std::array<double, 2> x_init = {5.0, -3.0};
    
    // 노이즈 스케줄링: 초기 노이즈 1.0, 매 반복마다 0.99배로 감소
    auto noise_scheduler = [](size_t iter) -> double {
        return 1.0 * std::pow(0.99, static_cast<double>(iter));
    };

    // 학습률 0.05, 수렴 오차 1e-4로 노이즈 하강법 수행
    auto result = NoisyDescent::optimize<2>(
        Quadratic2D{}, x_init, noise_scheduler, 0.05, 1e-4, 5000, false
    );

    // 확률적 노이즈가 개입되었음에도 AD 엔진의 기울기를 따라 전역 최적점 [0.0, 0.0]으로 수렴해야 함
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-2);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-2);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-3);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}