#ifndef OPTIMIZATION_AUTODIFF_HPP_
#define OPTIMIZATION_AUTODIFF_HPP_

#include "Optimization/Dual.hpp"  // 스칼라 Dual(이원수) 구조체 포함
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @brief 정적 메모리 엔진(Layer 1)과 완벽히 호환되는 Forward-Mode Automatic Differentiation (자동 미분) 클래스
 * 
 * 복잡한 수학적 함수의 도함수(기울기, 야코비안 등)를 해석적으로 구하지 않고, 
 * 이원수(Dual Numbers) 대수학을 활용하여 기계적으로 정확하게 계산(컴퓨터의 소수점 표현 한계인 절단 오차 없이 계산)하는 기법입니다.
 * 유한 차분법(Finite Difference)에서 발생하는 수치적 근사 오차(Truncation Error)가 발생하지 않아
 * 최적화 알고리즘(예: SQP, Newton's Method)에서 매우 안정적인 미분값을 제공합니다.
 * 
 * @note std::array를 전면 배제하고, 수학적 행렬 연산에 최적화된 내부 모듈인 StaticVector / StaticMatrix를 사용합니다.
 * @note SQP의 BFGS 업데이트를 위해 수치적 오차가 큰 유한 차분법(Finite Difference) 기반의 Hessian은 삭제하였습니다.
 */
class AutoDiff {
   public:
    // ============================================================
    // 1. Value (함수값만 추출, T = double)
    // ============================================================
    /**
     * @brief 목적 함수 또는 제약 조건 함수의 단순 스칼라 평가값(Value)을 반환합니다.
     * 
     * @tparam N 입력 벡터의 차원
     * @tparam Func 평가할 함수 타입 (lambda, 펑터 등)
     * @param f 평가할 함수
     * @param x 함수를 평가할 지점의 정적 벡터(StaticVector)
     * @return double 함수 평가 결과값 f(x)
     */
    template <size_t N, typename Func>
    [[nodiscard]] static double value(Func f, const StaticVector<double, N>& x) {
        return f(x);
    }

    // ============================================================
    // 2. Gradient (f: R^N -> R 의 미분)
    // ============================================================
    /**
     * @brief 목적 함수 f: R^N -> R 에 대한 기울기(Gradient) 벡터를 계산합니다.
     * 
     * 전방 자동 미분(Forward-Mode AD)을 사용하여 각 독립 변수에 대해 편미분을 수행합니다.
     * 이를 위해 N개의 차원에 대해 N번의 함수 평가를 수행하며, 매 반복마다 이원수(Dual Number)를 이용해 
     * 한 방향의 편미분 계수(Seed 주입)를 정확하게 추출합니다.
     * 
     * @tparam N 입력 벡터의 차원
     * @tparam Func 평가할 함수 타입
     * @param f 평가할 목적 함수
     * @param x 기울기를 계산할 지점의 정적 벡터
     * @return StaticVector<double, N> x 지점에서의 함수 f의 기울기 벡터 (∇f(x))
     */
    template <size_t N, typename Func>
    [[nodiscard]] static StaticVector<double, N> gradient(Func f,
                                                          const StaticVector<double, N>& x) {
        StaticVector<double, N> g;
        g.set_zero();

        // N개의 입력 변수 각각에 대해 편미분 수행
        for (size_t i = 0; i < N; ++i) {
            StaticVector<Dual<double>, N> x_dual;
            for (size_t j = 0; j < N; ++j) {
                // i번째 변수에 대해서만 미분값(d, Dual Number의 쌍대 부분)을 1.0으로 부여하여 "Seed"를 주입합니다.
                // 이는 수식 상 편미분 ∂f / ∂x_i 를 구하기 위한 설정입니다.
                // j != i 인 다른 변수들의 미분값은 0.0으로 두어 상수로 취급합니다.
                x_dual(static_cast<int>(j)) =
                    Dual<double>(x(static_cast<int>(j)), (i == j) ? 1.0 : 0.0);
            }
            
            // 이원수를 인자로 함수를 평가하면 반환되는 결과 역시 이원수(Dual Number)가 됩니다.
            // 반환된 이원수의 'd' 부분이 바로 우리가 찾고자 하는 i번째 변수에 대한 편미분 값입니다.
            Dual<double> res = f(x_dual);
            g(static_cast<int>(i)) = res.d;
        }
        return g;
    }

    // ============================================================
    // 3. Value and Gradient 동시 추출
    // ============================================================
    /**
     * @brief 스칼라 함수 평가값(Value)과 기울기(Gradient) 벡터를 한 번에 계산합니다.
     * 
     * 기울기 벡터를 계산하는 과정에서 첫 번째 변수의 편미분을 수행할 때,
     * 함수의 스칼라 평가값(Dual의 실수부)도 함께 추출하여 중복 평가를 줄이는 최적화된 함수입니다.
     * 
     * @tparam N 입력 벡터의 차원
     * @tparam Func 평가할 함수 타입
     * @param f 평가할 목적 함수
     * @param x 계산할 지점의 정적 벡터
     * @param v_out 계산된 함수값 f(x)가 저장될 참조 변수
     * @param g_out 계산된 기울기 벡터 ∇f(x)가 저장될 참조 변수
     */
    template <size_t N, typename Func>
    static void value_and_gradient(Func f, const StaticVector<double, N>& x, double& v_out,
                                   StaticVector<double, N>& g_out) {
        g_out.set_zero();

        for (size_t i = 0; i < N; ++i) {
            StaticVector<Dual<double>, N> x_dual;
            for (size_t j = 0; j < N; ++j) {
                // i번째 변수에만 편미분 Seed(1.0)를 주입
                x_dual(static_cast<int>(j)) =
                    Dual<double>(x(static_cast<int>(j)), (i == j) ? 1.0 : 0.0);
            }
            Dual<double> res = f(x_dual);

            // 첫 번째 변수에 대한 순회(i == 0)에서 함수의 스칼라 결과값(실수부 v)을 추출합니다.
            // 모든 반복에서 res.v는 동일한 f(x) 값을 가지므로 한 번만 저장하면 됩니다.
            if (i == 0) {
                v_out = res.v;
            }
            // 쌍대부(d)는 i번째 방향으로의 방향 도함수(편미분 값)입니다.
            g_out(static_cast<int>(i)) = res.d;
        }
    }

    // ============================================================
    // 4. Jacobian (f: R^N -> R^M 의 미분)
    // ============================================================
    /**
     * @brief 다변수 벡터 함수 f: R^N -> R^M 에 대한 야코비안(Jacobian) 행렬을 계산합니다.
     * 
     * 다중 출력(제약 조건, 비선형 시스템 등)을 갖는 함수의 1차 미분 행렬을 구성합니다.
     * 출력 행렬의 크기는 (출력 차원 M) x (입력 차원 N) 이며, 각 성분 J(k, i)는 
     * k번째 출력 함수를 i번째 입력 변수로 편미분한 값을 나타냅니다 ( ∂f_k / ∂x_i ).
     * 
     * @tparam M 출력 벡터의 차원 (예: 제약 조건의 개수)
     * @tparam N 입력 벡터의 차원 (예: 최적화 변수의 개수)
     * @tparam Func 벡터 결과를 반환하는 다변수 함수 타입
     * @param f 평가할 벡터 함수
     * @param x 야코비안 행렬을 계산할 지점의 정적 벡터
     * @return StaticMatrix<double, M, N> 계산된 야코비안 행렬
     */
    template <size_t M, size_t N, typename Func>
    [[nodiscard]] static StaticMatrix<double, M, N> jacobian(Func f,
                                                             const StaticVector<double, N>& x) {
        StaticMatrix<double, M, N> J;
        J.set_zero();

        // 입력 차원 N번의 평가를 통해 전체 야코비안 행렬을 구축합니다. (Forward-Mode AD의 특징)
        // 입력 변수 개수가 출력 함수 개수보다 현저히 많을 경우(M << N), 
        // Reverse-Mode AD보다 계산 비용이 더 클 수 있으나, 소규모 최적화 문제에서는 구현이 단순하고 매우 효율적입니다.
        for (size_t i = 0; i < N; ++i) {
            StaticVector<Dual<double>, N> x_dual;
            for (size_t j = 0; j < N; ++j) {
                // i번째 입력 변수에 대해 Seed(1.0)를 주입
                x_dual(static_cast<int>(j)) =
                    Dual<double>(x(static_cast<int>(j)), (i == j) ? 1.0 : 0.0);
            }

            // 함수 f는 입력으로 전달된 이원수 벡터를 이용해 연산을 수행하고,
            // 출력 차원 M개 만큼의 이원수 벡터를 반환해야 합니다.
            StaticVector<Dual<double>, M> res_vec = f(x_dual);

            // 반환된 각각의 이원수에서 'd'(쌍대부)를 추출하면,
            // 이는 각 출력 함수 f_k 가 i번째 입력 변수 x_i 의 미세한 변화에 어떻게 반응하는지를 의미합니다.
            // 따라서 각 출력 차원 k에 대하여, i번째 열에 편미분 값을 채워 넣습니다.
            for (size_t k = 0; k < M; ++k) {
                J(static_cast<int>(k), static_cast<int>(i)) = res_vec(static_cast<int>(k)).d;
            }
        }
        return J;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_AUTODIFF_HPP_