#ifndef EQP_SOLVER_HPP_
#define EQP_SOLVER_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"
/**
 * @brief Equality-Constrainted Quadratic Programming Solver
 * @details J(u) = 1/2 u^T P u + q^T u
 * subject to Au = b
 */
template <size_t N_vars, size_t N_cons>
class EQPSolver {
    public:
        // ===========================================================
        // 1. Input Interface (문제 정의 파라미터)
        // ===========================================================
        StaticMatrix<double, N_vars, N_vars> P;         // Hessian 근사치 (J^T J + R)
        StaticVector<double, N_vars> q;                 // 선형 비용 (보통 에러 벡터의 투영)
        StaticMatrix<double, N_cons, N_vars> A;         // 등식 제약 행렬 (Dynamic 제약)
        StaticVector<double, N_cons> b;                 // 등식 제약 경곗값

        // ===========================================================
        // 2. Output Interface (최적화 결과)
        // ===========================================================
        StaticVector<double, N_vars> u_opt;             // 최적 제어 입력 (Optimal Control)
        StaticVector<double, N_cons> lambda_opt;        // 라그랑주 상수 (Dual Variables)

    private:
        // ===========================================================
        // 3. KKT System 정적 메모리
        // ===========================================================
        static constexpr size_t KKT_Size = N_vars + N_cons;

        StaticMatrix<double, KKT_Size, KKT_Size> KKT_Matrix;
        StaticVector<double, KKT_Size> KKT_rhs;
        StaticVector<double, KKT_Size> KKT_solution;

    public:
        /**
         * @brief KKT 시스템을 조립하고 최적해를 도출합니다
         * @return bool 분해 및 수렴 성공 여부 (Positive Definite 실패 시 false)
         */
        bool solve() {
            // LDLT_decompose는 행렬을 인플레이스 (In-place)로 덮어쓰며 파괴합니다.
            // 따라서 매 solve 호출 시마다 이전 데이터의 찌꺼기를 완전히 지워야 합니다.
            KKT_Matrix.set_zero();
            KKT_rhs.set_zero();

            // -------------------------------------------------------
            // Step 1 : KKT_Matrix 블록 조립
            // [P   A^T]
            // [A   0  ]
            // -------------------------------------------------------
            KKT_Matrix.insert_block(0, 0, P);
            KKT_Matrix.insert_transposed_block(0, N_vars, A);   // Zero-copy Transpose
            KKT_Matrix.insert_block(N_vars, 0, A);
            // 우하단 N_cons x N_cons 크기의 '0' 블록은 set_zero()로 인해 이미 0입니다

            // -------------------------------------------------------
            // Step 2 : RHS 블록 조립
            // [ -q ]
            // [  b ]
            // -------------------------------------------------------
            StaticVector<double, N_vars> neg_q = q * static_cast<double>(-1.0);
            KKT_rhs.insert_block(0, 0, neg_q);
            KKT_rhs.insert_block(N_vars, 0, b);

            // -------------------------------------------------------
            // Step 3 : 최적화 엔진 가동 (LDLT 분해)
            // -------------------------------------------------------
            // 주의 : KKT 행렬의 우하단이 0이므로 전체 행렬은 엄밀히 양정치 (PD)가 아닙니다.
            // LDLT가 중간에 터진다면 KKT의 0 위치에 Regularization (예 : 1e-8 * I)이 필요할 수 있습니다.
            if (!KKT_Matrix.LDLT_decompose()) {
                return false;       // 특이 행렬이거나 솔버 실패 (Fallback 로직 발동 필요)
            }

            KKT_solution = KKT_Matrix.LDLT_solve(KKT_rhs);

            // -------------------------------------------------------
            // Step 4 : 최적해 (u, lambda) 분리 추출
            // -------------------------------------------------------
            u_opt = KKT_solution.template extract_block<N_vars, 1>(0, 0);
            lambda_opt = KKT_solution.template extract_block<N_cons, 1>(N_vars, 0);

            return true;
        }
};

#endif // EQP_SOLVER_HPP_