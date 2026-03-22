#include <gtest/gtest.h>

#include <array>

#include "Optimization/SecantMethod.hpp"

namespace Optimization::Tests {

/**
 * @brief 1차원 2차 함수 (Quadratic Function)
 * f(x) = (x - 3.0)^2
 * 전역 최적점: x* = 3.0, f(x*) = 0.0
 */
struct Quadratic1D {
    template <typename T, size_t N>
    constexpr T operator()(const std::array<T, N>& x) const noexcept {
        static_assert(N == 1, "Quadratic1D requires exactly 1 dimension.");
        T diff = x[0] - T(3.0);
        return diff * diff;
    }
};

}  // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(SecantMethodTest, Optimize1DQuadratic) {
    // 할선법은 미분값을 직접 계산하지 않는 대신 2개의 초기점이 필요합니다.
    double x0 = 0.0;
    double x1 = 5.0;  // x0와 달라야 합니다.

    // 허용 오차 1e-6으로 Secant Method 최적화 수행
    auto result = SecantMethod::optimize(Quadratic1D{}, x0, x1, 1e-6, 50, false);

    // 정답(x = 3.0)으로 정확히 수렴했는지 검증
    EXPECT_NEAR(result.x_opt, 3.0, 1e-5);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-5);

    // 타이머 및 반복 횟수가 정상 측정되었는지 검증
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}