#include <gtest/gtest.h>
#include <array>
#include <cmath>

#include "Optimization/NelderMead.hpp"

namespace Optimization::Tests {

    // 테스트용 Rosenbrock Function (협곡 지형)
    struct RosenbrockFunction {
        template <typename T, size_t N>
        constexpr T operator()(const std::array<T, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            const T d1 = T(1.0) - x[0];
            const T d2 = x[1] - (x[0] * x[0]);
            return (d1 * d1) + T(100.0) * (d2 * d2);
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(NelderMeadTest, OptimizeRosenbrockFunction) {
    std::array<double, 2> x_init = {-1.2, 1.0}; // 시작점
    
    // Nelder-Mead 알고리즘 실행
    // step = 1.0 (초기 심플렉스 크기), tol = 1e-6 (수렴 오차)
    auto result = NelderMead::optimize<2>(
        RosenbrockFunction{}, x_init, 1.0, 1e-6, 2000, false
    );

    // Rosenbrock의 전역 최적점 [1.0, 1.0]에 수렴했는지 확인 (오차 1e-3 허용)
    // 0차 방법(Direct Method)은 수렴 속도가 느려 오차 허용치를 조금 넉넉히 줍니다.
    EXPECT_NEAR(result.x_opt[0], 1.0, 1e-2);
    EXPECT_NEAR(result.x_opt[1], 1.0, 1e-2);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-3);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}