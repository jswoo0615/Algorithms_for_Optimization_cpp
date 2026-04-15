#ifndef EQP_SOLVER_HPP_
#define EQP_SOLVER_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"

/**
 * @brief 등식 제약 조건이 있는 이차 계획법(Equality-Constrained Quadratic Programming, EQP) 솔버
 * 클래스
 * @details
 * 다음과 같은 형태의 볼록 최적화(Convex Optimization) 문제를 풉니다.
 *
 * 최소화(Minimize):
 *   J(u) = 1/2 u^T P u + q^T u
 *
 * 제약조건(Subject to):
 *   A u = b
 *
 * 여기서 u는 구하고자 하는 제어 변수(Optimal Control Vector)이며,
 * P는 양의 정부호(Positive Definite) 혹은 양의 준정부호(Positive Semi-Definite) 대칭 행렬인
 * 헤시안(Hessian) 행렬입니다.
 *
 * 이 문제는 라그랑주 승수법(Lagrange Multipliers)을 사용하여 KKT(Karush-Kuhn-Tucker) 시스템이라는
 * 선형 연립방정식을 블록 행렬 형태로 구성하고, 그 해를 구함으로써 해석적으로 단번에(closed-form)
 * 최적해를 얻을 수 있습니다. 주로 비선형 제어(NMPC)나 순차적 이차 계획법(SQP), 내점법(Interior
 * Point Method)의 내부 부분 문제(Subproblem)로 활용됩니다.
 *
 * @tparam N_vars 최적화할 변수(u)의 차원 개수
 * @tparam N_cons 등식 제약 조건의 개수
 */
template <size_t N_vars, size_t N_cons>
class EQPSolver {
   public:
    // ===========================================================
    // 1. Input Interface (문제 정의 파라미터)
    // ===========================================================

    /**
     * @brief 목적 함수의 이차항 계수 행렬 (Hessian Matrix Approximation)
     * @details 비용 함수의 곡률(curvature)을 나타내며, 대칭 행렬(Symmetric Matrix)이어야 합니다.
     * (예: Gauss-Newton 방식에서 J^T J + R 형태로 구성됨)
     */
    StaticMatrix<double, N_vars, N_vars> P;

    /**
     * @brief 목적 함수의 선형항 계수 벡터 (Linear Cost Vector / Gradient)
     * @details 목적 함수의 일차 미분항(Gradient 방향) 혹은 에러 벡터의 투영값을 나타냅니다.
     */
    StaticVector<double, N_vars> q;

    /**
     * @brief 등식 제약 조건 행렬 (Equality Constraint / Dynamic 제약)
     * @details 제약 조건 방정식 A u = b 에서 좌변 행렬을 구성합니다. (예: 동역학 모델의 선형화 된
     * 상태 전이 행렬 등)
     */
    StaticMatrix<double, N_cons, N_vars> A;

    /**
     * @brief 등식 제약 조건의 경곗값 벡터 (Equality Constraint Bound Vector)
     * @details 제약 조건 방정식 A u = b 에서 우변 벡터를 구성합니다.
     */
    StaticVector<double, N_cons> b;

    // ===========================================================
    // 2. Output Interface (최적화 결과)
    // ===========================================================

    /**
     * @brief 계산된 최적의 제어 입력 벡터 (Optimal Control / Primal Variables)
     * @details KKT 연립방정식을 풀어낸 후 도출된 변수 u 가 이곳에 저장됩니다.
     */
    StaticVector<double, N_vars> u_opt;

    /**
     * @brief 계산된 라그랑주 승수 벡터 (Lagrange Multipliers / Dual Variables)
     * @details 각 제약 조건에 대응되는 쌍대 변수(Dual Variable) 값들이며, 최적점에서 제약 조건이
     * 목적 함수 비용에 미치는 민감도(Sensitivity / Shadow Price)를 의미합니다.
     */
    StaticVector<double, N_cons> lambda_opt;

   private:
    // ===========================================================
    // 3. KKT System 정적 메모리 (Static Memory Allocation)
    // ===========================================================

    /// @brief KKT 행렬 전체의 차원 크기 (제어 변수의 개수 + 제약 조건의 개수)
    static constexpr size_t KKT_Size = N_vars + N_cons;

    // 동적 힙 메모리 할당(new/malloc)을 배제하여 실시간(Real-Time) 제어 환경에서의 안전성과 속도를
    // 확보합니다.
    StaticMatrix<double, KKT_Size, KKT_Size> KKT_Matrix;  ///< KKT 선형 연립방정식의 좌변 블록 행렬
    StaticVector<double, KKT_Size> KKT_rhs;  ///< KKT 선형 연립방정식의 우변 벡터 (Right-Hand Side)
    StaticVector<double, KKT_Size>
        KKT_solution;  ///< KKT 시스템의 전체 해 벡터 (u 와 lambda 가 병합되어 있음)

   public:
    /**
     * @brief KKT 시스템을 조립하고 선형대수 솔버(LDLT 분해)를 통해 최적해를 도출합니다.
     * @details
     * KKT 시스템(Lagrange 1차 필요조건)은 다음과 같은 형태의 연립방정식으로 도출됩니다:
     *
     * [ P    A^T ] [ u      ]   = [ -q ]
     * [ A     0  ] [ lambda ]     [  b ]
     *
     * 이 블록 행렬을 구성한 후, `MatrixEngine`에서 제공하는 `LDLT_decompose()` 및 `LDLT_solve()`를
     * 이용하여 단번에 해를 구합니다.
     *
     * @note 주의사항 (Regularization Caveats):
     * KKT 행렬은 우하단 블록이 영행렬(0)이므로 전체적으로 양의 정부호(Positive Definite)가 될 수
     * 없는 대칭 부정부호 행렬(Symmetric Indefinite Matrix, 안장점 구조)입니다. 따라서 일반적인
     * Cholesky (LL^T) 분해는 불가능하며 LDL^T 분해를 사용해야 합니다. 제약 조건 행렬 A의
     * 행(Row)들이 선형 독립(Linear Independent)이 아니거나, P가 제약조건의 Null Space에서 충분히
     * 양의 정부호가 아닐 경우 분해가 터지거나(실패) 특이 행렬(Singular)이 될 수 있습니다.
     *
     * @return bool KKT 시스템 분해 및 해 도출 성공 여부
     * (성공 시 true. 행렬이 특이 상태이거나 수치적 문제로 분해 실패 시 false 반환)
     */
    bool solve() {
        // -------------------------------------------------------
        // [초기화]
        // -------------------------------------------------------
        // LDLT_decompose는 계산 효율을 위해 대상 행렬을 덮어쓰면서(In-place 파괴) 진행됩니다.
        // 따라서 같은 EQPSolver 인스턴스를 재사용하여 매번 solve()를 호출할 때,
        // 이전 루프에서 덮어씌워진 찌꺼기(데이터 쓰레기값)가 남지 않도록 완전히 0으로 초기화해야
        // 합니다.
        KKT_Matrix.set_zero();
        KKT_rhs.set_zero();

        // -------------------------------------------------------
        // Step 1 : KKT_Matrix 블록(Block) 단위 조립
        // 행렬 구조:
        // [ P    A^T ]
        // [ A     0  ]
        // -------------------------------------------------------

        // (1) 좌상단에 비용 함수 헤시안 행렬 P 삽입
        KKT_Matrix.insert_block(0, 0, P);

        // (2) 우상단에 제약 조건 행렬 A의 전치(A^T) 삽입
        // (Zero-copy Transpose와 같은 최적화 기능이 포함된 insert_transposed_block 활용 권장)
        KKT_Matrix.insert_transposed_block(0, N_vars, A);

        // (3) 좌하단에 제약 조건 행렬 A 삽입
        KKT_Matrix.insert_block(N_vars, 0, A);

        // (4) 우하단 N_cons x N_cons 크기의 블록 행렬은, KKT_Matrix.set_zero()로
        // 이미 메모리가 0으로 덮어씌워져 있으므로 추가적인 연산 없이 자연스럽게 '0 블록'이 됩니다.

        // -------------------------------------------------------
        // Step 2 : 우변 벡터(RHS Vector) 블록 조립
        // 벡터 구조:
        // [ -q ]
        // [  b ]
        // -------------------------------------------------------

        // (1) 상단 파트: 비용 벡터 q의 부호 반전(-q) 삽입
        StaticVector<double, N_vars> neg_q = q * static_cast<double>(-1.0);
        KKT_rhs.insert_block(0, 0, neg_q);

        // (2) 하단 파트: 제약 조건 경곗값 벡터 b 삽입
        KKT_rhs.insert_block(N_vars, 0, b);

        // -------------------------------------------------------
        // Step 3 : 최적화 연립방정식 풀이 (LDLT 분해 및 전/후진 대입)
        // -------------------------------------------------------

        // KKT 행렬을 L(하삼각행렬) * D(대각행렬) * L^T 꼴로 분해합니다.
        // 분해가 터지는 경우(Zero Pivot 발생 등)는 수치적으로 풀 수 없음을 의미합니다.
        // 실무에서는 이런 현상을 대비하기 위해 우하단 0 블록에 (1e-8 * I) 같은 작은 값을 더해주는
        // 정규화(Regularization) 기법(Fallback 로직)을 추가로 구성하기도 합니다.
        if (!KKT_Matrix.LDLT_decompose()) {
            return false;  // 특이 행렬이거나 솔버가 파탄(Singularity)나면 실패 처리
        }

        // 분해된 행렬 인자들(L, D)을 이용하여 O(N^2) 속도로 RHS에 대한 해를 구합니다.
        KKT_solution = KKT_Matrix.LDLT_solve(KKT_rhs);

        // -------------------------------------------------------
        // Step 4 : 결과 도출 (Primal & Dual 최적해 분리 및 추출)
        // -------------------------------------------------------

        // KKT 해 벡터의 앞부분(상단 N_vars 개)은 실제 우리가 구하고자 하는 최적 제어 변수(u)
        // 입니다.
        u_opt = KKT_solution.template extract_block<N_vars, 1>(0, 0);

        // KKT 해 벡터의 뒷부분(하단 N_cons 개)은 최적 라그랑주 승수(lambda) 입니다.
        lambda_opt = KKT_solution.template extract_block<N_cons, 1>(N_vars, 0);

        return true;
    }
};

#endif  // EQP_SOLVER_HPP_