#ifndef OPTIMIZATION_AUTODIFF_HPP_
#define OPTIMIZATION_AUTODIFF_HPP_

#include <array>
#include <functional>

#include "Optimization/Dual.hpp"

namespace Optimization {
/**
 * @brief AD 엔진을 캡슐화하여 사용자에게 미분 인터페이스를 제공하는 클래스
 */
class AutoDiff {
   public:
    // ==========================================================
    // 1. Value (함수값만 추출)
    // ==========================================================
    template <size_t N, typename Func>
    static double value(Func f, const std::array<double, N>& x_point) {
        return f(x_point);
    }
    // ==========================================================
    // 2. Gradient (f: R^N -> R의 미분)
    // ==========================================================
    /**
     * @brief 입력 벡터에 대한 스칼라 함수의 Gradient 계산
     */
    template <size_t N, typename Func>
    static std::array<double, N> gradient(Func f, const std::array<double, N>& x_point) {
        // 1. 입력 변수를 DualVec으로 변환 및 Seeding
        std::array<DualVec<double, N>, N> x_dual;
        for (size_t i = 0; i < N; ++i) {
            x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
        }
        // 2. 함수 실행 (사용자 수식이 DualVec 연산으로 수행)
        DualVec<double, N> res = f(x_dual);

        // 3. 결과에서 Gradient 벡터 (.g)만 반환
        return res.g;
    }

    // ==========================================================
    // 3. Value and Gradient (최적화 솔버용 핵심 유틸리티)
    // ==========================================================
    /**
     * @brief 함수값과 Gradient를 한 번의 Forward Pass로 동시에 획득
     */
    template <size_t N, typename Func>
    static void value_and_gradient(Func f, const std::array<double, N>& x_point, double& v_out,
                                   std::array<double, N>& g_out) {
        std::array<DualVec<double, N>, N> x_dual;
        for (size_t i = 0; i < N; ++i) {
            x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
        }
        DualVec<double, N> res = f(x_dual);
        v_out = res.v;  // 함수값 (Value)
        g_out = res.g;  // Gradient
    }

    // ==========================================================
    // 4. Jacobian (f: R^N -> R^M의 미분)
    // ==========================================================
    /**
     * @brief 벡터 함수의 자코비안 행렬 계산
     * @tparam M 출력 차원 (함수의 갯수)
     * @tparam N 입력 차원 (변수의 갯수)
     * @return std::array<std::array<double, N>, M> 행렬 구조
     */
    template <size_t M, size_t N, typename Func>

    static std::array<std::array<double, N>, M> jacobian(Func f,
                                                         const std::array<double, N>& x_point) {
        std::array<DualVec<double, N>, N> x_dual;
        for (size_t i = 0; i < N; ++i) {
            x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
        }
        // 함수 결과는 M개의 요소를 가진 DualVec 배열이어야 함
        std::array<DualVec<double, N>, M> res_vec = f(x_dual);

        std::array<std::array<double, N>, M> J;
        for (size_t i = 0; i < M; ++i) {
            J[i] = res_vec[i].g;
        }
        return J;
    }

    // ==========================================================
    // Hessian 행렬 계산
    // 1차 미분은 AD로 도출, 2차 미분만 수치 미분 (FD)으로 덮어씌움
    // ==========================================================
    template <size_t N, typename Func>
    static std::array<std::array<double, N>, N> hessian(Func f, const std::array<double, N>& x) {
        std::array<std::array<double, N>, N> H;
        const double eps = 1e-5;  // 중앙 차분 미소 간격

        for (size_t i = 0; i < N; ++i) {
            auto x_plus = x;
            x_plus[i] += eps;
            auto x_minus = x;
            x_minus[i] -= eps;

            auto g_plus = gradient<N>(f, x_plus);
            auto g_minus = gradient<N>(f, x_minus);

            for (size_t j = 0; j < N; ++j) {
                H[i][j] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
            }
        }

        // 헤시안 행렬은 대칭행렬이므로, 수치 오차 보정을 위해 대칭화
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sym = (H[i][j] + H[j][i]) / 2.0;
                H[i][j] = sym;
                H[j][i] = sym;
            }
        }
        return H;
    }
};
}  // namespace Optimization
#endif  // OPTIMIZATION_AUTODIFF_HPP_