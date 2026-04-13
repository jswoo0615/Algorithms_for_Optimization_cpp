#ifndef OPTIMIZATION_SPARSE_IPM_QP_SOLVER_HPP_
#define OPTIMIZATION_SPARSE_IPM_QP_SOLVER_HPP_

#include "Optimization/Matrix/SparseMatrixEngine.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"
#include <cmath>
#include <algorithm>

namespace Optimization {

    /**
     * @brief 희소 행렬(Sparse Matrix) 기반 내부 점 기법 (Interior Point Method) QP 솔버
     * 
     * 이 클래스는 다음과 같은 형태의 2차 계획법(Quadratic Programming) 문제를 풉니다.
     *   Minimize    1/2 * x^T * P * x + q^T * x
     *   Subject to  A_ineq * x <= b_ineq
     * 
     * 희소 행렬(Sparse Matrix) 연산과 Matrix-Free 켤레 기울기법(Conjugate Gradient)을
     * 결합하여, 대규모의 제어 문제에서도 메모리를 적게 쓰고 빠르게 최적해를 계산합니다.
     * 
     * @tparam N_vars 최적화 변수의 개수 (Dimension of x)
     * @tparam N_ineq 부등식 제약조건의 개수
     * @tparam MaxNNZ_P 목적 함수 헤시안 P의 최대 비영요소(Non-zero) 허용 개수
     * @tparam MaxNNZ_A 제약 조건 행렬 A_ineq의 최대 비영요소(Non-zero) 허용 개수
     */
    template <size_t N_vars, size_t N_ineq, size_t MaxNNZ_P, size_t MaxNNZ_A>
    class SparseIPMQPSolver {
        public:
            // ==============================================================================
            // 1. QP(Quadratic Programming) 문제 정의를 위한 행렬 및 벡터
            // ==============================================================================
            /// @brief 목적 함수의 2차항 계수 행렬 (Hessian, 희소 행렬)
            StaticSparseMatrix<double, N_vars, N_vars, MaxNNZ_P> P;
            /// @brief 목적 함수의 1차항 계수 벡터 (Gradient)
            StaticVector<double, N_vars> q;
            /// @brief 부등식 제약 조건 행렬 (희소 행렬)
            StaticSparseMatrix<double, N_ineq, N_vars, MaxNNZ_A> A_ineq;
            /// @brief 부등식 제약 조건의 우변 한계값 벡터
            StaticVector<double, N_ineq> b_ineq;

            // ==============================================================================
            // 2. Primal-Dual 상태 변수 (연속적인 NMPC 제어 시 Warm-start를 위해 객체 상태로 유지)
            // ==============================================================================
            /// @brief 최적화 대상인 Primal 변수 벡터
            StaticVector<double, N_vars> x;
            /// @brief 부등식 제약 조건에 대한 Dual 변수 (Lagrange Multipliers)
            StaticVector<double, N_ineq> lambda;
            /// @brief 부등식 제약 조건을 등식으로 바꾸기 위한 여유 변수 (Slack Variables)
            StaticVector<double, N_ineq> s;

            /**
             * @brief 생성자: Primal-Dual IPM의 초기 상태 설정
             * @note 내부 점 기법(IPM)은 슬랙 변수와 듀얼 변수가 항상 양수(Strictly Positive) 영역에
             *       있어야 하므로 1.0으로 초기화하여 안전한 내부(Interior)에서 탐색을 시작합니다.
             */
            SparseIPMQPSolver() {
                x.set_zero();
                for (size_t i = 0; i < N_ineq; ++i) {
                    lambda(i) = 1.0;
                    s(i) = 1.0;
                }
            }

            /**
             * @brief [Matrix-Free 연산] 축소된 KKT 시스템의 행렬-벡터 곱셈 (Schur Complement)
             * 
             * 메모리에 거대한 KKT 행렬을 명시적으로 직접 만들지 않고 연산만 수행하는 기법입니다.
             * 우리가 풀어야 할 축소된 헤시안 행렬은 H_sys = P + A^T * W * A 입니다.
             * 
             * @param p 곱할 입력 벡터 (탐색 방향)
             * @param W 가중치 벡터 (대각 행렬 W = lambda * S^{-1} 의 대각 성분)
             * @param y 계산 결과가 저장될 출력 벡터 (y = H_sys * p)
             */
            void apply_H_sys(const StaticVector<double, N_vars>& p, const StaticVector<double, N_ineq>& W, StaticVector<double, N_vars>& y) const {
                // 1. y = P * p (목적 함수 곡률 방향 투영)
                P.multiply(p, y);
                
                // 2. temp1 = A * p (제약 조건 방향 투영)
                StaticVector<double, N_ineq> Ap;
                A_ineq.multiply(p, Ap);

                // 3. temp2 = W * (A * p) (가중치 대각 행렬 W 적용)
                for (size_t i = 0; i < N_ineq; ++i) {
                    Ap(i) *= W(i);
                }

                // 4. y += A^T * temp2 (다시 원래 변수 공간으로 역투영하여 누적 합산)
                StaticVector<double, N_vars> AT_Ap;
                A_ineq.multiply_transpose(Ap, AT_Ap);
                for (size_t i = 0; i < N_vars; ++i) {
                    y(i) += AT_Ap(i);
                }
            }

            // [Conjugate Gradient] 축소된 KKT 시스템 해결
            bool solve_implicit_cg(const StaticVector<double, N_vars>& rhs, StaticVector<double, N_ineq>& W, StaticVector<double, N_vars>& dx) const {
                StaticVector<double, N_vars> r = rhs;  // 초기 추정치 dx = 0
                StaticVector<double, N_vars> p_cg = r;
                StaticVector<double, N_vars> Ap_cg;

                double rsold = 0.0;
                for (size_t i = 0; i < N_vars; ++i) {
                    rsold += r(i) * r(i);
                }
                if (rsold < 1e-8) {
                    return true;
                }

                for (int iter = 0; iter < 30; ++iter) { // 고정 반복 (WCET 보장)
                    apply_H_sys(p_cg, W, Ap_cg);
                    double pAp = 0.0;
                    for (size_t i = 0; i < N_vars; ++i) {
                        pAp += p_cg(i) * Ap_cg(i);
                    }
                    if (std::abs(pAp) < 1e-12) {
                        break;
                    }
                    double alpha = rsold / pAp;
                    double rsnew = 0.0;
                    for (size_t i = 0; i < N_vars; ++i) {
                        dx(i) += alpha * p_cg(i);
                        r(i) -= alpha * Ap_cg(i);
                        rsnew += r(i) * r(i);
                    }
                    if (rsnew < 1e-6) {
                        return true;
                    }
                    double beta = rsnew / rsold;
                    for (size_t i = 0; i < N_vars; ++i) {
                        p_cg(i) = r(i) + beta * p_cg(i);
                    }
                    rsold = rsnew;
                }
                return false;
            }

            bool solve(StaticVector<double, N_vars>& out_x, int max_iter = 10, double tol = 1e-3) {
                // Warm-start 
                for (size_t i = 0; i < N_vars; ++i) {
                    x(i) = out_x(i);
                }

                for (int iter = 0; iter < max_iter; ++iter) {
                    // 1. Residual 계산 (KKT 시스템의 잔차)
                    StaticVector<double, N_vars> Px;
                    P.multiply(x, Px);
                    StaticVector<double, N_vars> AT_lambda; 
                    A_ineq.multiply_transpose(lambda, AT_lambda);
                    StaticVector<double, N_vars> r_d; // r_d = Px + q + A^T lambda
                    for (size_t i = 0; i < N_vars; ++i) {
                        r_d(i) = Px(i) + q(i) + AT_lambda(i);
                    }

                    StaticVector<double, N_ineq> Ax;
                    A_ineq.multiply(x, Ax);
                    StaticVector<double, N_ineq> r_p; // r_p = Ax - b + s
                    for (size_t i = 0; i < N_ineq; ++i) {
                        r_p(i) = Ax(i) - b_ineq(i) + s(i);
                    }

                    double gap = 0.0;
                    for (size_t i = 0; i < N_ineq; ++i) {
                        gap += lambda(i) * s(i);
                    }
                    double mu = gap / N_ineq;

                    if (mu < tol) {     // 수렴 시
                        break;
                    }

                    double sigma = 0.1; // Centering 파라미터
                    double target_mu = sigma * mu;

                    // 2. Weight Matrix (W = lambda * S^{-1}) 및 RHS 조립
                    StaticVector<double, N_ineq> W;
                    StaticVector<double, N_vars> rhs = r_d;     // rhs = -r_d - A^T (S^{-1} (Lambda r_p - r_c))
                    StaticVector<double, N_ineq> temp_vec;
                    for (size_t i = 0; i < N_ineq; ++i) {
                        W(i) = lambda(i) / s(i);
                        double r_c = lambda(i) * s(i) - target_mu;
                        temp_vec(i) = (lambda(i) * r_p(i) - r_c) / s(i);
                    }
                    
                    StaticVector<double, N_vars> AT_temp;
                    A_ineq.multiply_transpose(temp_vec, AT_temp);
                    for (size_t i = 0; i < N_vars; ++i) {
                        rhs(i) = -r_d(i) - AT_temp(i);
                    }

                    // 3. Newton Step 계산 (Implicit CG)
                    StaticVector<double, N_vars> dx;
                    dx.set_zero();
                    solve_implicit_cg(rhs, W, dx);

                    // 4. 슬랙 및 듀얼 스텝 복원
                    StaticVector<double, N_ineq> A_dx; 
                    A_ineq.multiply(dx, A_dx);
                    StaticVector<double, N_ineq> ds, dlambda;
                    for (size_t i = 0; i < N_ineq; ++i) {
                        ds(i) = -r_p(i) - A_dx(i);
                        double r_c = lambda(i) * s(i) - target_mu;
                        dlambda(i) = -(r_c + lambda(i) * ds(i)) / s(i);
                    }

                    // 5. Fraction-to-the-boundary step size
                    double alpha_p = 1.0, alpha_d = 1.0;
                    for (size_t i = 0; i < N_ineq; ++i) {
                        if (ds(i) < 0.0) {
                            alpha_p = std::min(alpha_p, -0.99 * s(i) / ds(i));
                        }
                        if (dlambda(i) < 0.0) {
                            alpha_d = std::min(alpha_d, -0.99 * lambda(i) / dlambda(i));
                        }
                    }

                    // 6. 업데이트
                    for (size_t i = 0; i < N_vars; ++i) {
                        x(i) += alpha_p * dx(i);
                    }
                    for (size_t i = 0; i < N_ineq; ++i) {
                        s(i) += alpha_p * ds(i);
                        lambda(i) += alpha_d * dlambda(i);
                    }
                }

                // 결과 반환
                for (size_t i = 0; i < N_vars; ++i) {
                    out_x(i) = x(i);
                }
                return true;
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_SPARSE_IPM_QP_SOLVER_HPP_