#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "Optimization/HookeJeeves.hpp"

namespace Optimization::Tests {

/**
 * @brief 미분 불가능한 2차원 목적 함수 (Non-differentiable Function)
 * f(x) = |x_1| + 2 * |x_2 - 1.0|
 * 전역 최적점: x* = [0.0, 1.0], f(x*) = 0.0
 * 뉴턴법(Newton)이나 BFGS는 절대값의 첨점(Cusp)에서 기울기가 파탄나 수렴에 실패하지만,
 * 0차 방법인 Hooke-Jeeves는 완벽하게 뚫고 지나가야 합니다.
 */
struct NonDifferentiable2D {
    template <typename T, size_t N>
    constexpr T operator()(const std::array<T, N>& x) const noexcept {
        static_assert(N == 2, "Dimension must be 2.");
        return std::abs(x[0]) + T(2.0) * std::abs(x[1] - T(1.0));
    }
};

}  // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(HookeJeevesTest, OptimizeNonDifferentiableFunction) {
    std::array<double, 2> x_init = {5.0, -3.0};  // 의도적으로 멀리 떨어진 초기 위치

    auto result =
        HookeJeeves::optimize<2>(NonDifferentiable2D{}, x_init, 0.5, 0.5, 1e-6, 2000, false);

    // 미분 불가능 지형에서도 전역 최적점 [0.0, 1.0] 에 정확히 도달해야 함
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-4);
    EXPECT_NEAR(result.x_opt[1], 1.0, 1e-4);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-4);

    // 연산 소요 시간 및 이터레이션 검증
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}