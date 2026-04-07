#ifndef OPTIMIZATION_AUTODIFF_HPP_
#define OPTIMIZATION_AUTODIFF_HPP_

#include "Optimization/Dual.hpp"  // 스칼라 Dual 구조체 포함
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @brief 정적 메모리 엔진(Layer 1)과 완벽히 호환되는 Auto Differentiation 인터페이스
 * @note std::array를 전면 배제하고 StaticVector / StaticMatrix를 사용.
 * @note SQP의 BFGS 업데이트를 위해 수치적 오차가 큰 Finite Difference Hessian은 삭제함.
 */
class AutoDiff {
   public:
    // ============================================================
    // 1. Value (함수값만 추출, T = double)
    // ============================================================
    template <size_t N, typename Func>
    [[nodiscard]] static double value(Func f, const StaticVector<double, N>& x) {
        return f(x);
    }

    // ============================================================
    // 2. Gradient (f: R^N -> R 의 미분)
    // ============================================================
    // Forward-mode AD를 사용하여 각 변수에 대해 편미분 수행 (N번 순회)
    template <size_t N, typename Func>
    [[nodiscard]] static StaticVector<double, N> gradient(Func f,
                                                          const StaticVector<double, N>& x) {
        StaticVector<double, N> g;
        g.set_zero();

        for (size_t i = 0; i < N; ++i) {
            StaticVector<Dual<double>, N> x_dual;
            for (size_t j = 0; j < N; ++j) {
                // i번째 변수에 대해서만 미분값(d) 1.0 부여 (Seed 주입)
                x_dual(static_cast<int>(j)) =
                    Dual<double>(x(static_cast<int>(j)), (i == j) ? 1.0 : 0.0);
            }
            Dual<double> res = f(x_dual);
            g(static_cast<int>(i)) = res.d;
        }
        return g;
    }

    // ============================================================
    // 3. Value and Gradient 동시 추출
    // ============================================================
    template <size_t N, typename Func>
    static void value_and_gradient(Func f, const StaticVector<double, N>& x, double& v_out,
                                   StaticVector<double, N>& g_out) {
        g_out.set_zero();

        for (size_t i = 0; i < N; ++i) {
            StaticVector<Dual<double>, N> x_dual;
            for (size_t j = 0; j < N; ++j) {
                x_dual(static_cast<int>(j)) =
                    Dual<double>(x(static_cast<int>(j)), (i == j) ? 1.0 : 0.0);
            }
            Dual<double> res = f(x_dual);

            // 첫 번째 순회에서 함수 값(Value) 추출
            if (i == 0) {
                v_out = res.v;
            }
            g_out(static_cast<int>(i)) = res.d;
        }
    }

    // ============================================================
    // 4. Jacobian (f: R^N -> R^M 의 미분)
    // ============================================================
    // 결과: M x N 행렬 (행: 출력차원, 열: 입력차원)
    template <size_t M, size_t N, typename Func>
    [[nodiscard]] static StaticMatrix<double, M, N> jacobian(Func f,
                                                             const StaticVector<double, N>& x) {
        StaticMatrix<double, M, N> J;
        J.set_zero();

        for (size_t i = 0; i < N; ++i) {
            StaticVector<Dual<double>, N> x_dual;
            for (size_t j = 0; j < N; ++j) {
                x_dual(static_cast<int>(j)) =
                    Dual<double>(x(static_cast<int>(j)), (i == j) ? 1.0 : 0.0);
            }

            // f는 StaticVector<Dual<double>, M> 을 반환해야 함
            StaticVector<Dual<double>, M> res_vec = f(x_dual);

            // 각 출력 차원 k에 대하여, i번째 입력 변수에 대한 편미분 값을 J(k, i)에 저장
            for (size_t k = 0; k < M; ++k) {
                J(static_cast<int>(k), static_cast<int>(i)) = res_vec(static_cast<int>(k)).d;
            }
        }
        return J;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_AUTODIFF_HPP_