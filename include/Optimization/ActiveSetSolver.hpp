#ifndef ACTIVE_SET_SOLVER_HPP_
#define ACTIVE_SET_SOLVER_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"

/**
 * @brief Primal Active-Set QP Solver
 * @details J(u) = 1/2 u^T P u + q^T u
 * subject to A_eq u = b_eq
 * A_ineq u <= b_ineq
 */
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class ActiveSetSolver {
   public:
    // ==================================================================
    // 1. 문제 정의 (Hessian, Gradient, Eq/Ineq Constraints)
    // ==================================================================
    StaticMatrix<double, N_vars, N_vars> P;
    StaticVector<double, N_vars> q;

    StaticMatrix<double, N_eq, N_vars> A_eq;
    StaticVector<double, N_eq> b_eq;

    StaticMatrix<double, N_ineq, N_vars> A_ineq;
    StaticVector<double, N_ineq> b_ineq;

    // ==================================================================
    // 2. 상태 변수 (Working Set)
    // ==================================================================
    // true이면 해당 부등식 제약이 활성화되어 등식처럼 취급
    bool working_set[N_ineq > 0 ? N_ineq : 1] = {false};

   private:
    // 최대 KKT 시스템 크기 : 변수 + 등식 + 부등식 (전체 활성화 가정 시 최대치)
    static constexpr size_t KKT_Size = N_vars + N_eq + N_ineq;

    StaticMatrix<double, KKT_Size, KKT_Size> KKT_Matrix;
    StaticVector<double, KKT_Size> KKT_rhs;
    StaticVector<double, KKT_Size> KKT_solution;

   public:
    /**
     * @brief 현재 Working Set의 상태를 반영하여 KKT 시스템을 동적으로 조립 (Masking 기법)
     * @param u_k 현재의 제어 변수 상태 (탐색 방향 p를 구하기 위한 잔차 계산용)
     */
    void build_masked_KKT(const StaticVector<double, N_vars>& u_k) {
        KKT_Matrix.set_zero();
        KKT_rhs.set_zero();

        // --------------------------------------------------------
        // Step 1 : 좌상단 블록 P (Hessian) 및 비용 함수 Gradient 잔차
        // --------------------------------------------------------
        KKT_Matrix.insert_block(0, 0, P);

        // 탐색 방향 p에 대한 RHS : - (P * u_k + q)
        StaticVector<double, N_vars> grad = (P * u_k) + q;
        grad = grad * static_cast<double>(-1.0);
        KKT_rhs.insert_block(0, 0, grad);

        // --------------------------------------------------------
        // Step 2 : 등식 제약 블록 (항상 활성화 상태로 매핑)
        // --------------------------------------------------------
        if constexpr (N_eq > 0) {
            KKT_Matrix.insert_transposed_block(0, N_vars, A_eq);
            KKT_Matrix.insert_block(N_vars, 0, A_eq);

            // RHS 잔차 : b_eq - A_eq * u_k
            StaticVector<double, N_eq> eq_res = b_eq - (A_eq * u_k);
            KKT_rhs.insert_block(N_vars, 0, eq_res);
        }

        // --------------------------------------------------------
        // Step 3 : 부등식 제약 블록 (Working Set 배열에 따른 런타임 마스킹)
        // --------------------------------------------------------
        if constexpr (N_ineq > 0) {
            const size_t ineq_offset = N_vars + N_eq;
            for (size_t i = 0; i < N_ineq; ++i) {
                if (working_set[i]) {
                    // 활성화 됨 : 실제 A_ineq의 해당 행을 시스템에 삽입
                    for (size_t j = 0; j < N_vars; ++j) {
                        KKT_Matrix(static_cast<int>(ineq_offset + i), static_cast<int>(j)) =
                            A_ineq(static_cast<int>(i), static_cast<int>(j));
                        KKT_Matrix(static_cast<int>(j), static_cast<int>(ineq_offset + i)) =
                            A_ineq(static_cast<int>(i), static_cast<int>(j));
                    }

                    // RHS 잔차 : 활성화된 제약의 거리 b_ineq - A_ineq * u_k
                    double ineq_val = 0;
                    for (size_t j = 0; j < N_vars; ++j) {
                        ineq_val += A_ineq(static_cast<int>(i), static_cast<int>(j)) *
                                    u_k(static_cast<int>(j));
                    }
                    KKT_rhs(static_cast<int>(ineq_offset + i)) =
                        b_ineq(static_cast<int>(i)) - ineq_val;
                } else {
                    // 비활성화 됨 : 역행렬 특이성 (Singular)을 방지하기 위해 대각 원소 1.0 삽입
                    KKT_Matrix(static_cast<int>(ineq_offset + i),
                               static_cast<int>(ineq_offset + i)) = 1.0;
                    KKT_rhs(static_cast<int>(ineq_offset + i)) = 0.0;
                }
            }
        }
    }

    /**
     * @brief Primal Active-Set QP 최적화 루프
     * @param u [in, out] 초기 추정치 (Initial Guess)로 시작하여 최적해로 업데이트 됨
     * @param max_iter 무한 루프 방지용 최대 반복 횟수 (Worst-case Iteration 통제)
     * @return bool 최적해 수렴 여부
     */
    bool solve(StaticVector<double, N_vars>& u, int max_iter = 100) {
        for (int iter = 0; iter < max_iter; ++iter) {
            // 1. 현재 Working Set과 상태 u를 기반으로 KKT 시스템 조립
            build_masked_KKT(u);

            if (!KKT_Matrix.LDLT_decompose()) {
                // 특이 행렬 발생 (제약 조건의 선형 종속 등) -> Fallback 전략 필요
                return false;
            }

            KKT_solution = KKT_Matrix.LDLT_solve(KKT_rhs);

            // 탐색 방향 p 추출
            StaticVector<double, N_vars> p = KKT_solution.template extract_block<N_vars, 1>(0, 0);

            // p의 무한 노름 (Infinity Norm) 계산
            double p_norm = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                p_norm = std::max(p_norm, std::abs(p(i)));
            }

            // -------------------------------------------------------------
            // Branch A : p == 0 (현재 Working Set 내에서 최적점에 도달)
            // -------------------------------------------------------------
            if (p_norm < 1e-6) {
                double min_lambda = 0.0;
                int drop_idx = -1;
                const size_t ineq_offset = N_vars + N_eq;

                // KKT 조건의 Dual Feasibility 검사 (활성화된 부등식의 λ >= 0 인가?)
                for (size_t i = 0; i < N_ineq; ++i) {
                    if (working_set[i]) {
                        double lambda_i = KKT_solution(static_cast<int>(ineq_offset + i));
                        // 가장 음수인 라그랑주 승수를 찾음
                        if (lambda_i < min_lambda) {
                            min_lambda = lambda_i;
                            drop_idx = static_cast<int>(i);
                        }
                    }
                }

                if (drop_idx == -1) {
                    // 모든 활성 부등식 제약의 λ >= 0 -> KKT 최적성 조건 완벽 충족
                    return true;
                } else {
                    // 해를 영역 안쪽으로 끌어당기는 '나쁜' 제약 조건 해제 (Drop)
                    working_set[drop_idx] = false;
                }
            }
            // -------------------------------------------------------------
            // Branch B : p != 0 (아직 더 나은 비용 함수 값을 향해 이동 가능)
            // -------------------------------------------------------------
            else {
                double alpha = 1.0;
                int blocking_idx = -1;

                StaticVector<double, N_ineq> a_p = A_ineq * p;
                StaticVector<double, N_ineq> a_u = A_ineq * u;

                for (size_t i = 0; i < N_ineq; ++i) {
                    // 비활성화된 제약 중, 이동 방향 p가 제약 경계면을 향해 갈 때만 (a_p > 0) 검사
                    if (!working_set[i] && a_p(i) > 1e-8) {
                        double alpha_i = (b_ineq(i) - a_u(i)) / a_p(i);

                        // 가장 먼저 부딪히는 경계면 찾기
                        if (alpha_i < alpha) {
                            alpha = alpha_i;
                            blocking_idx = static_cast<int>(i);
                        }
                    }
                }

                // 상태 변수 업데이트
                u = u + (p * alpha);

                // 스텝이 가로막혔다면 해당 제약을 Working Set에 추가 (Add)
                if (blocking_idx != -1) {
                    working_set[blocking_idx] = true;
                }
            }
        }
        // Iteration Limit 초과 (Cycling 발생 혹은 스텝 폭발)
        return false;
    }
};

#endif  // ACTIVE_SET_SOLVER_HPP_