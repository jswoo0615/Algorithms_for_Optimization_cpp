#ifndef OPTIMIZATION_NEWTON_SOLVER_HPP_
#define OPTIMIZATION_NEWTON_SOLVER_HPP_

#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Dual.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
    namespace solver {
        /**
         * @brief 다변수 뉴턴-랩슨 솔버 (Multivariable Newton-Raphson Root Finder)
         * @details f(x) = 0을 만족하는 N차원 벡터 x를 찾습니다
         */
        template <size_t N, typename Functor>
        SolverStatus solve_newton(StaticVector<double, N>& x_opt, const Functor& f, int max_iter = 50, double tol = 1e-6) {
            using ADVar = DualVec<double, N>;
            for (int iter = 0; iter < max_iter; ++iter) {
                // 1. 현재 x 값을 DualVec 독립 변수로 변환 (시드 주입)
                StaticVector<ADVar, N> x_dual;
                for (size_t i = 0; i < N; ++i) {
                    x_dual(static_cast<int>(i)) = ADVar::make_variable(x_opt(static_cast<int>(i)), i);
                }

                // 2. 비선형 함수 평가 (Value와 Jacobian이 동시에 계산됨)
                StaticVector<ADVar, N> y_dual = f(x_dual);

                // 3. 선형 시스템 조립 : J * dx = -F
                StaticMatrix<double, N, N> J;
                StaticVector<double, N> neg_F;
                double max_residual = 0.0;

                for (size_t i = 0; i < N; ++i) {
                    double f_val = y_dual(static_cast<int>(i)).v;
                    neg_F(static_cast<int>(i)) = -f_val;            // -F 벡터 구성

                    // 수렴 판정용 최대 잔차 (Residual) 업데이트
                    double abs_f = std::abs(f_val);
                    if (abs_f > max_residual) {
                        max_residual = abs_f;
                    }

                    // 자코비안 행렬 J 추출
                    for (size_t j = 0; j < N; ++j) {
                        J(static_cast<int>(i), static_cast<int>(j)) = y_dual(static_cast<int>(i)).g[j];
                    }
                }

                // 4. 수렴 조건 검사 (허용 오차 이내 진입 시 즉시 종료)
                if (max_residual < tol) {
                    return SolverStatus::SUCCESS;
                }

                // 5. 탐색 방향 (Search Direction) 계산 : LU 분해 활용 (비대칭 행렬 가능성이 높으므로 LU 사용)
                StaticVector<int, N> P;
                MathStatus m_status = linalg::LU_decompose(J, P);

                // 수학적으로 발산하면 즉시 실패 선언
                if (m_status != MathStatus::SUCCESS) {
                    return SolverStatus::MATH_ERROR;
                }

                StaticVector<double, N> dx = linalg::LU_solve(J, P, neg_F);

                // 6. 상태 업데이트 : x_new = x_old + dx
                for (size_t i = 0; i < N; ++i) {
                    x_opt(static_cast<int>(i)) += dx(static_cast<int>(i));
                }
            }
            // 최대 반복 횟수를 채웠으나, 수렴하지 못한 경우
            return SolverStatus::MAX_ITERATION_REACHED;
        }
        
    } // namespace solver
} // namespace Optimization
#endif // OPTIMIZATION_NEWTON_SOLVER_HPP_