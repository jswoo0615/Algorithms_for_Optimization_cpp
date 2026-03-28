#include <gtest/gtest.h>
#include <array>
#include <cmath>

#include "Optimization/CrossEntropy.hpp"

namespace Optimization::Tests {

    // Ackley Function: 전역 최적화 테스트를 위한 악명 높은 다중 국소 최적점 함수
    // 전역 최적점: x* = [0.0, 0.0], f(x*) = 0.0
    struct Ackley2D {
        template <size_t N>
        constexpr double operator()(const std::array<double, N>& x) const noexcept {
            static_assert(N == 2, "Dimension must be 2.");
            double sum1 = x[0] * x[0] + x[1] * x[1];
            double sum2 = std::cos(2.0 * M_PI * x[0]) + std::cos(2.0 * M_PI * x[1]);
            
            return -20.0 * std::exp(-0.2 * std::sqrt(0.5 * sum1)) 
                   - std::exp(0.5 * sum2) + 20.0 + M_E;
        }
    };

} // namespace Optimization::Tests

using namespace Optimization;
using namespace Optimization::Tests;

TEST(CrossEntropyTest, OptimizeAckleyFunction) {
    // 초기 평균 지점을 전역 최적점(0,0)에서 멀리 떨어진 곳으로 설정
    std::array<double, 2> mu_init = {5.0, 5.0};
    
    // 초기 분산을 매우 크게 주어 탐색 공간 전체에 샘플을 흩뿌림
    std::array<double, 2> sigma_sq_init = {10.0, 10.0};
    
    // M=200, M_ELITE=20 으로 세팅하여 CEM 실행 (허용 분산 오차 1e-5)
    auto result = CrossEntropy::optimize<2, 200, 20>(
        Ackley2D{}, mu_init, sigma_sq_init, 200, 1e-5, 12345, false
    );

    // 수많은 가짜 바닥을 뚫고 전역 최적점 [0.0, 0.0] 에 수렴해야 함
    EXPECT_NEAR(result.x_opt[0], 0.0, 1e-2);
    EXPECT_NEAR(result.x_opt[1], 0.0, 1e-2);
    EXPECT_NEAR(result.f_opt, 0.0, 1e-2);
    
    EXPECT_GT(result.elapsed_ns, 0);
    EXPECT_GT(result.iterations, 0);
}