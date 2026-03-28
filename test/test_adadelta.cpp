#include <gtest/gtest.h>
#include <array>

#include "Optimization/AdaDelta.hpp"

namespace Optimization::Tests {

    /**
     * @brief Beale Function
     * @details 평탄한 계곡과 날카로운 경사가 혼재된 비볼록(Non-convex) 함수.
     * 전역 최적점: x* = [3.0, 0.5], f(x*) = 0.0
     */
    struct BealeFunction {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            T term1 = T(1.5) - x[0] + x[0] * x[1];
            T term2 = T(2.25) - x[0] + x[0] * x[1] * x[1];
            T term3 = T(2.625) - x[0] + x[0] * x[1] * x[1] * x[1];
            return term1 * term1 + term2 * term2 + term3 * term3;
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(AdaDeltaTest, OptimizeBealeFunction) {
    std::array<double, 2> x_init = {1.0, 1.0};
    
    // gamma = 0.9, epsilon = 1e-8, 허용 오차 1e-4
    auto result = AdaDelta::optimize<2>(
        BealeFunction{}, x_init, 0.9, 1e-8, 20000, 1e-4, false
    );

    // 학습률(alpha) 세팅 없이 스스로 스케일을 조정하여 최적점에 수렴해야 함
    EXPECT_NEAR(result.x_opt[0], 3.0, 1e-1);
    EXPECT_NEAR(result.x_opt[1], 0.5, 1e-1);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}