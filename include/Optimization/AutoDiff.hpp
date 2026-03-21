#ifndef OPTIMIZATION_AUTODIFF_HPP_
#define OPTIMIZATION_AUTODIFF_HPP_

#include <array>
#include <functional>

#include "Optimization/Dual.hpp"

namespace Optimization {
/**
 * @brief Automatic Differentiation 인터페이스 제공하는 정적 클래스
 * @note 실시간 제어기 적용을 위해 동적 할당 차단 및 컴파일 타임 최적화 강제
 */
class AutoDiff {
   public:
    // ============================================================
    // 1. Value (함수값만 추출)
    // ============================================================
    template <size_t N, typename Func>
    [[nodiscard]] static constexpr double value(Func f,
                                                const std::array<double, N>& x_point) noexcept {
        return f(x_point);
    }

    // ============================================================
    // 2. Gradient (f: R^N -> R의 미분)
    // ============================================================
    template <size_t N, typename Func>
    [[nodiscard]] static constexpr std::array<double, N> gradient(
        Func f, const std::array<double, N>& x_point) noexcept {
        std::array<DualVec<double, N>, N> x_dual{};  // Zero-initialization
        for (size_t i = 0; i < N; ++i) {
            x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
        }
        return f(x_dual).g;  // RVO (Return Value Optimization)
    }

    // ============================================================
    // 3. Value and Gradient
    // ============================================================
    template <size_t N, typename Func>
    static constexpr void value_and_gradient(Func f, const std::array<double, N>& x_point,
                                             double& v_out, std::array<double, N>& g_out) noexcept {
        std::array<DualVec<double, N>, N> x_dual{};
        for (size_t i = 0; i < N; ++i) {
            x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
        }
        const auto res = f(x_dual);
        v_out = res.v;
        g_out = res.g;
    }
    // ============================================================
    // 4. Jacobian (f: R^N -> R^M의 미분)
    // ============================================================
    template <size_t M, size_t N, typename Func>
    [[nodiscard]] static constexpr std::array<std::array<double, N>, M> jacobian(
        Func f, const std::array<double, N>& x_point) noexcept {
        std::array<DualVec<double, N>, N> x_dual{};
        for (size_t i = 0; i < N; ++i) {
            x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
        }
        const std::array<DualVec<double, N>, M> res_vec = f(x_dual);
        std::array<std::array<double, N>, M> J{};

        for (size_t i = 0; i < M; ++i) {
            J[i] = res_vec[i].g;
        }
        return J;
    }

    // ============================================================
    // 5. Hessian 행렬 계산 (Central Difference)
    // ============================================================
    template <size_t N, typename Func>
    [[nodiscard]] static std::array<std::array<double, N>, N> hessian(
        Func f, const std::array<double, N>& x) noexcept {
        std::array<std::array<double, N>, N> H{};
        constexpr double eps = 1e-5;
        constexpr double inv_two_eps = 1.0 / (2.0 * eps);

        for (size_t i = 0; i < N; ++i) {
            auto x_plus = x;
            x_plus[i] += eps;
            auto x_minus = x;
            x_minus[i] -= eps;

            const auto g_plus = gradient<N>(f, x_plus);
            const auto g_minus = gradient<N>(f, x_minus);

            for (size_t j = 0; j < N; ++j) {
                H[i][j] = (g_plus[j] - g_minus[j]) * inv_two_eps;
            }
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                const double sym = (H[i][j] + H[j][i]) * 0.5;
                H[i][j] = sym;
                H[j][i] = sym;
            }
        }
        return H;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_AUTODIFF_HPP_