#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "Optimization/DIRECT.hpp"

namespace Optimization::Tests {

/**
 * @brief Branin Function (전역 최적화 테스트용 벤치마크)
 * @details 여러 개의 국소 최적점(Local minima)이 존재합니다.
 * 전역 최적점: f(x*) ≈ 0.397887
 * 위치: (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
 */
struct BraninFunction {
    template <size_t N>
    constexpr double operator()(const std::array<double, N>& x) const noexcept {
        static_assert(N == 2, "Branin function is defined for 2 dimensions.");

        const double a = 1.0;
        const double b = 5.1 / (4.0 * M_PI * M_PI);
        const double c = 5.0 / M_PI;
        const double r = 6.0;
        const double s = 10.0;
        const double t = 1.0 / (8.0 * M_PI);

        double term1 = x[1] - b * x[0] * x[0] + c * x[0] - r;
        double term2 = s * (1.0 - t) * std::cos(x[0]);

        return term1 * term1 + term2 + s;
    }
};

}  // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(DIRECTTest, OptimizeBraninFunction) {
    // 탐색 경계 (Bounds) 설정
    std::array<double, 2> lower_bound = {-5.0, 0.0};
    std::array<double, 2> upper_bound = {10.0, 15.0};

    // DIRECT 알고리즘 수행 (최대 구간 10000개 허용)
    auto result =
        DIRECT<2, 10000>::optimize(BraninFunction{}, lower_bound, upper_bound, 1e-4, 50, false);

    // 전역 최적값 0.397887 에 도달했는지 검증 (오차 1e-2 허용)
    EXPECT_NEAR(result.f_opt, 0.397887, 1e-2);

    // 소요 시간 및 동작 횟수 검증
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}