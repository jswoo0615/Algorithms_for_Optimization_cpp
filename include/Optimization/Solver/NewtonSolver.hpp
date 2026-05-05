#ifndef OPTIMIZATION_NEWTON_SOLVER_HPP_
#define OPTIMIZATION_NEWTON_SOLVER_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
namespace solver {

/**
 * @brief 고속 다변수 뉴턴-랩슨 솔버 (Zero-Allocation & Robust)
 * @details
 * f(x) = 0을 만족하는 N차원 벡터 x를 찾습니다.
 * AD(Auto Differentiation) 엔진과 결합되어 해석적 자코비안을 자동 생성하며,
 * 내부적으로 SIMD 가속된 LU 분해를 사용하여 연산 속도를 극대화했습니다.
 */
template <size_t N, typename Functor>
inline SolverStatus solve_newton(StaticVector<double, N>& x_opt, const Functor& f,
                                 int max_iter = 50, double tol = 1e-6) {
    using ADVar = DualVec<double, N>;

    // [Architect's Update]
    // 루프 내부의 임시 객체 생성을 막기 위해 메모리를 루프 외부로 끌어올림(Hoist)
    StaticVector<ADVar, N> x_dual;
    StaticMatrix<double, N, N> J;
    StaticVector<double, N> neg_F;
    StaticVector<double, N> dx;
    StaticVector<int, N> P;

    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. 현재 x 값을 DualVec 독립 변수로 변환 (시드 주입)
        for (size_t i = 0; i < N; ++i) {
            x_dual(i) = ADVar::make_variable(x_opt(i), i);
        }

        // 2. 비선형 함수 평가 (Value와 Jacobian이 동시에 계산됨)
        // Functor의 반환값에 의한 복사는 컴파일러의 RVO(Return Value Optimization)에 맡깁니다.
        StaticVector<ADVar, N> y_dual = f(x_dual);

        // 3. 선형 시스템 조립 : J * dx = -F
        double max_residual = 0.0;

        for (size_t i = 0; i < N; ++i) {
            // [Architect's Update] MathTraits를 통한 안전한 값 추출 및 절대값 계산
            double f_val = Optimization::get_value(y_dual(i));
            neg_F(i) = -f_val;

            // 수렴 판정용 최대 잔차 (Residual) 업데이트
            double abs_f = MathTraits<double>::abs(f_val);
            if (abs_f > max_residual) {
                max_residual = abs_f;
            }

            // 자코비안 행렬 J 추출
            for (size_t j = 0; j < N; ++j) {
                J(i, j) = y_dual(i).g[j];
            }
        }

        // 4. 수렴 조건 검사 (허용 오차 이내 진입 시 즉시 종료)
        if (max_residual < tol) {
            return SolverStatus::SUCCESS;
        }

        // 5. 탐색 방향 (Search Direction) 계산 : LU 분해 활용
        MathStatus m_status = linalg::LU_decompose(J, P);

        // 수학적으로 발산하면 즉시 실패 선언 (특이 행렬 등)
        if (m_status != MathStatus::SUCCESS) {
            return SolverStatus::MATH_ERROR;
        }

        // [Architect's Update] 무할당(Zero-Allocation) In-place Solve 적용
        linalg::LU_solve(J, P, neg_F, dx);

        // [Architect's Update] Step Size 검증 (Local Minima 및 Stagnation 방어)
        double max_dx = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double abs_dx = MathTraits<double>::abs(dx(i));
            if (abs_dx > max_dx) max_dx = abs_dx;
        }
        if (max_dx < std::numeric_limits<double>::epsilon() * 10.0) {
            return SolverStatus::STEP_SIZE_TOO_SMALL;  // 더 이상 탐색 불가능
        }

        // 6. 상태 업데이트 : x_new = x_old + dx
        // [Architect's Update] SIMD 가속 연산자(operator+=) 활용
        x_opt += dx;
    }

    // 최대 반복 횟수를 채웠으나 수렴하지 못한 경우
    return SolverStatus::MAX_ITERATION_REACHED;
}

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_NEWTON_SOLVER_HPP_