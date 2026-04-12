#ifndef ACTIVE_SET_SOLVER_HPP_
#define ACTIVE_SET_SOLVER_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"

/**
 * @brief Primal Active-Set 방식을 이용한 2차 계획법(Quadratic Programming, QP) 솔버
 * 
 * 부등식 제약 조건이 포함된 2차 최적화 문제를 푸는 클래스입니다.
 * 목적 함수: J(u) = 1/2 * u^T * P * u + q^T * u 를 최소화합니다.
 * 제약 조건:
 *   - 등식 제약: A_eq * u = b_eq
 *   - 부등식 제약: A_ineq * u <= b_ineq
 * 
 * Active-Set 알고리즘은 부등식 제약 중 현재 해(u)에서 정확히 경계면에 도달한 제약(활성화된 제약, Active Constraint)을
 * 등식 제약처럼 취급(Working Set)하여 하위 문제(KKT 시스템)를 풀고, 
 * 라그랑주 승수(Lagrange Multiplier)와 탐색 방향(Search Direction)을 평가하여
 * 제약을 추가(Add)하거나 제거(Drop)하는 과정을 반복하며 최적해를 찾는 기법입니다.
 * 
 * @tparam N_vars 최적화 변수의 개수
 * @tparam N_eq 등식 제약 조건의 개수
 * @tparam N_ineq 부등식 제약 조건의 개수
 */
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class ActiveSetSolver {
   public:
    // ==================================================================
    // 1. 문제 정의 (Hessian, Gradient, Eq/Ineq Constraints)
    // ==================================================================
    StaticMatrix<double, N_vars, N_vars> P; ///< 목적 함수의 2차항 계수 행렬 (Hessian). 양의 정부호(Positive Definite)여야 함.
    StaticVector<double, N_vars> q;         ///< 목적 함수의 1차항 계수 벡터 (Gradient).

    StaticMatrix<double, N_eq, N_vars> A_eq; ///< 등식 제약 조건의 계수 행렬
    StaticVector<double, N_eq> b_eq;         ///< 등식 제약 조건의 상수 벡터

    StaticMatrix<double, N_ineq, N_vars> A_ineq; ///< 부등식 제약 조건의 계수 행렬
    StaticVector<double, N_ineq> b_ineq;         ///< 부등식 제약 조건의 상수 벡터

    // ==================================================================
    // 2. 상태 변수 (Working Set)
    // ==================================================================
    /**
     * @brief 현재 활성화된 부등식 제약 조건을 추적하는 배열
     * 
     * 배열의 요소가 true이면 해당 인덱스의 부등식 제약이 현재 경계면(A_ineq * u == b_ineq)에
     * 도달했다고 간주하여 KKT 시스템 조립 시 등식 제약처럼 시스템에 포함시킵니다.
     * N_ineq가 0일 경우 컴파일 오류를 방지하기 위해 최소 크기 1로 선언합니다.
     */
    bool working_set[N_ineq > 0 ? N_ineq : 1] = {false};

   private:
    /**
     * @brief KKT(Karush-Kuhn-Tucker) 행렬의 최대 크기
     * 
     * KKT 시스템은 [ 변수(N_vars) + 등식 제약(N_eq) + 부등식 제약(N_ineq) ] 차원을 가집니다.
     * 부등식 제약이 모두 활성화될 수 있는 최악의 경우(Worst-case)를 대비하여 정적 메모리를 미리 할당합니다.
     */
    static constexpr size_t KKT_Size = N_vars + N_eq + N_ineq;

    StaticMatrix<double, KKT_Size, KKT_Size> KKT_Matrix; ///< 매 반복마다 구성되는 KKT 시스템 좌변 행렬
    StaticVector<double, KKT_Size> KKT_rhs;              ///< 매 반복마다 구성되는 KKT 시스템 우변 벡터 (잔차)
    StaticVector<double, KKT_Size> KKT_solution;         ///< 풀이 결과 (탐색 방향 p 및 제약 조건의 라그랑주 승수 λ)

   public:
    /**
     * @brief 현재 Working Set의 상태를 반영하여 KKT 시스템을 동적으로 조립 (Masking 기법)
     * 
     * 활성화되지 않은 부등식 제약 조건의 행과 열은 KKT_Matrix에서 제거하지 않고, 
     * 대각 원소를 1.0, 우변을 0.0으로 설정하는 마스킹(Masking) 기법을 사용하여
     * 정적 행렬 크기를 유지하면서 역행렬 계산 시 특이성(Singularity)이 발생하지 않도록 처리합니다.
     * 
     * @param u_k 현재의 최적화 변수 상태 (탐색 방향 p를 구하기 위한 우변 잔차 계산에 사용)
     */
    void build_masked_KKT(const StaticVector<double, N_vars>& u_k) {
        KKT_Matrix.set_zero();
        KKT_rhs.set_zero();

        // --------------------------------------------------------
        // Step 1 : 좌상단 블록 P (Hessian) 및 변수에 대한 우변 잔차
        // --------------------------------------------------------
        // KKT 행렬의 (0, 0) 위치에 P 행렬 삽입
        KKT_Matrix.insert_block(0, 0, P);

        // 탐색 방향 p에 대한 우변(RHS)은 현재 위치에서의 목적 함수 기울기의 음수: - (P * u_k + q)
        StaticVector<double, N_vars> grad = (P * u_k) + q;
        grad = grad * static_cast<double>(-1.0);
        KKT_rhs.insert_block(0, 0, grad);

        // --------------------------------------------------------
        // Step 2 : 등식 제약 블록 (항상 활성화 상태이므로 무조건 KKT 행렬에 포함)
        // --------------------------------------------------------
        if constexpr (N_eq > 0) {
            // KKT 행렬의 좌하단 및 우상단 블록에 A_eq 및 A_eq^T 삽입
            KKT_Matrix.insert_transposed_block(0, N_vars, A_eq); // 우상단: A_eq^T
            KKT_Matrix.insert_block(N_vars, 0, A_eq);            // 좌하단: A_eq

            // 등식 제약에 대한 우변(RHS) 잔차 : b_eq - A_eq * u_k
            StaticVector<double, N_eq> eq_res = b_eq - (A_eq * u_k);
            KKT_rhs.insert_block(N_vars, 0, eq_res);
        }

        // --------------------------------------------------------
        // Step 3 : 부등식 제약 블록 (Working Set 배열에 따른 런타임 마스킹)
        // --------------------------------------------------------
        if constexpr (N_ineq > 0) {
            const size_t ineq_offset = N_vars + N_eq; // 부등식 제약 조건이 삽입될 시작 인덱스
            for (size_t i = 0; i < N_ineq; ++i) {
                if (working_set[i]) {
                    // [활성화 됨] : 현재 해가 부등식 경계에 도달해 있으므로 등식 제약처럼 시스템에 삽입
                    for (size_t j = 0; j < N_vars; ++j) {
                        // 좌하단 (A_ineq) 및 우상단 (A_ineq^T) 삽입
                        KKT_Matrix(static_cast<int>(ineq_offset + i), static_cast<int>(j)) =
                            A_ineq(static_cast<int>(i), static_cast<int>(j));
                        KKT_Matrix(static_cast<int>(j), static_cast<int>(ineq_offset + i)) =
                            A_ineq(static_cast<int>(i), static_cast<int>(j));
                    }

                    // 활성화된 제약의 우변(RHS) 잔차 : b_ineq - A_ineq * u_k
                    // p를 이동하여 이 잔차를 정확히 0으로 만들고자 함.
                    double ineq_val = 0;
                    for (size_t j = 0; j < N_vars; ++j) {
                        ineq_val += A_ineq(static_cast<int>(i), static_cast<int>(j)) *
                                    u_k(static_cast<int>(j));
                    }
                    KKT_rhs(static_cast<int>(ineq_offset + i)) =
                        b_ineq(static_cast<int>(i)) - ineq_val;
                } else {
                    // [비활성화 됨] : 제약 조건 내부 영역에 있어 여유가 있는 상태
                    // KKT 행렬의 크기를 동적으로 줄이지 않고 대각 원소를 1.0으로, 우변을 0.0으로 두어
                    // 해당 라그랑주 승수가 0이 되도록 유도하고 행렬이 가역행렬(Invertible)이 되도록 보장함.
                    KKT_Matrix(static_cast<int>(ineq_offset + i),
                               static_cast<int>(ineq_offset + i)) = 1.0;
                    KKT_rhs(static_cast<int>(ineq_offset + i)) = 0.0;
                }
            }
        }
    }

    /**
     * @brief Primal Active-Set QP 최적화 메인 루프
     * 
     * @param u [in, out] 초기 추정치 (Initial Guess)로 시작하여 최적해로 업데이트 됩니다. 
     *                    반드시 모든 부등식 제약을 만족하는 Feasible한 초기값이어야 합니다 (Primal 방식의 전제조건).
     * @param max_iter 무한 루프 방지용 최대 반복 횟수 (제약 조건의 추가/제거 반복을 통제)
     * @return bool 최적해 수렴에 성공하면 true, KKT 행렬 분해 실패 혹은 최대 반복 초과 시 false 반환
     */
    bool solve(StaticVector<double, N_vars>& u, int max_iter = 100) {
        for (int iter = 0; iter < max_iter; ++iter) {
            // 1. 현재 Working Set과 변수 u를 기반으로 KKT 시스템 조립
            build_masked_KKT(u);

            // 2. KKT 행렬을 풀기 위해 LDLT 분해(Decomposition) 시도
            // Active-Set 방법에서 KKT 행렬은 Indefinite 형태(양수 및 음수 고윳값 혼재)를 띠므로,
            // Cholesky 대신 LDLT 분해 등 더 강력한 솔버가 필요합니다.
            if (!KKT_Matrix.LDLT_decompose()) {
                // 특이 행렬(Singular Matrix) 발생. 주로 제약 조건들이 선형 종속이거나
                // P 행렬이 양의 정부호가 아닐 때 발생합니다.
                return false;
            }

            // 3. 탐색 방향 p 및 라그랑주 승수 λ 계산
            KKT_solution = KKT_Matrix.LDLT_solve(KKT_rhs);

            // KKT_solution 벡터의 상단 N_vars 만큼이 실제 공간에서의 탐색 방향 p에 해당합니다.
            StaticVector<double, N_vars> p = KKT_solution.template extract_block<N_vars, 1>(0, 0);

            // 탐색 방향 벡터 p의 크기(L-infinity norm) 계산
            double p_norm = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                p_norm = std::max(p_norm, std::abs(p(static_cast<int>(i))));
            }

            // -------------------------------------------------------------
            // Branch A : p == 0 (현재 활성화된 제약 조건들의 부분 공간 내에서 최적점에 도달)
            // -------------------------------------------------------------
            // 탐색 방향이 거의 0이라는 것은 현재의 Working Set 하에서는 더 이상 갈 곳이 없음을 의미합니다.
            if (p_norm < 1e-6) {
                double min_lambda = 0.0;
                int drop_idx = -1;
                const size_t ineq_offset = N_vars + N_eq;

                // KKT 최적성 조건 중 쌍대 실행 가능성(Dual Feasibility) 검사:
                // 부등식 제약(g_i(x) <= 0)에 대한 라그랑주 승수 λ_i 는 반드시 0 이상이어야 합니다.
                // 만약 λ_i < 0 이라면, 해당 제약 조건을 경계에서 해제(Drop)했을 때 
                // 목적 함수 값을 더 낮출 수 있는 방향으로 진행할 수 있다는 뜻입니다.
                for (size_t i = 0; i < N_ineq; ++i) {
                    if (working_set[i]) {
                        double lambda_i = KKT_solution(static_cast<int>(ineq_offset + i));
                        // 가장 음수 값이 큰(즉, 해제했을 때 가장 이득이 큰) 제약 조건을 찾습니다.
                        if (lambda_i < min_lambda) {
                            min_lambda = lambda_i;
                            drop_idx = static_cast<int>(i);
                        }
                    }
                }

                if (drop_idx == -1) {
                    // 음수인 라그랑주 승수가 없다면 (모든 λ_i >= 0),
                    // 현재 해가 전역적인 최적점(Global Optimum)이며 KKT 조건을 완벽히 충족한 것입니다.
                    return true;
                } else {
                    // 가장 '나쁜' 영향을 주는 제약 조건 하나를 Working Set에서 제거(Drop)하여
                    // 다음 반복에서 해당 제약 경계면 안쪽으로 파고들어갈 수 있게 합니다.
                    working_set[drop_idx] = false;
                }
            }
            // -------------------------------------------------------------
            // Branch B : p != 0 (탐색 방향 p를 따라 더 나은 목적 함수 값을 향해 이동 가능)
            // -------------------------------------------------------------
            else {
                double alpha = 1.0; // 기본 스텝 사이즈 (최대 이동 거리)
                int blocking_idx = -1; // 이동 중에 부딪히게 될 새로운 제약 조건의 인덱스

                // p 방향으로 이동했을 때의 각 부등식 제약 조건의 변화량(a_p)과 현재 제약 조건 값(a_u)
                StaticVector<double, N_ineq> a_p = A_ineq * p;
                StaticVector<double, N_ineq> a_u = A_ineq * u;

                for (size_t i = 0; i < N_ineq; ++i) {
                    // 비활성화된 제약 중, 탐색 방향 p가 제약 경계면을 향해 가고 있을 때만 (a_p_i > 0) 검사합니다.
                    // a_p_i <= 0 인 경우는 제약 조건 안쪽(더 안전한 방향)으로 가고 있으므로 충돌할 일이 없습니다.
                    if (!working_set[i] && a_p(static_cast<int>(i)) > 1e-8) {
                        // 현재 위치에서 경계면까지 도달하는 데 필요한 이동 비율(거리) 계산
                        // (A_ineq * (u + alpha * p) = b_ineq) 를 alpha에 대해 푼 결과입니다.
                        double alpha_i = (b_ineq(static_cast<int>(i)) - a_u(static_cast<int>(i))) / a_p(static_cast<int>(i));

                        // 현재까지 찾은 가장 짧은 거리보다 짧다면 갱신 (가장 먼저 부딪히는 경계면 찾기)
                        if (alpha_i < alpha) {
                            alpha = alpha_i;
                            blocking_idx = static_cast<int>(i);
                        }
                    }
                }

                // 상태 변수 u를 방향 p를 따라 스텝 사이즈 alpha 만큼 이동시킵니다.
                for (size_t i = 0; i < N_vars; ++i) {
                    u(static_cast<int>(i)) = u(static_cast<int>(i)) + (p(static_cast<int>(i)) * alpha);
                }

                // 스텝 사이즈가 1.0 미만으로 잘렸다는 것은 이동 중에 새로운 부등식 경계면과 부딪혔음을 의미합니다.
                // 부딪힌 제약 조건을 Working Set에 추가(Add)하여, 
                // 다음 반복부터는 이 제약을 넘어서 밖으로 나가지 못하도록 등식 제약처럼 고정시킵니다.
                if (blocking_idx != -1) {
                    working_set[blocking_idx] = true;
                }
            }
        }
        // 반복 횟수가 max_iter에 도달했다면 수렴에 실패한 것입니다.
        // 흔한 실패 원인으로는 제약 조건이 상충(Infeasible)하여 Cycling(Add/Drop 반복 무한루프)이 발생한 경우가 있습니다.
        return false;
    }
};

#endif  // ACTIVE_SET_SOLVER_HPP_