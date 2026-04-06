/**
 * @file  test_static_matrix.cpp
 * @brief StaticMatrix.hpp 수치 검증 테스트 스위트
 *
 * 빌드:
 *   g++ -std=c++17 -O2 -DNDEBUG -o test_static_matrix test_static_matrix.cpp && ./test_static_matrix
 *
 * 검증 항목:
 *   [1] 메모리 레이아웃  — Col-major 인덱싱 / alignas(64)
 *   [2] 산술 연산자     — +, -, *(scalar), /(scalar), *(matrix)
 *   [3] LU 분해/해법   — 정방, 특이행렬, 해 정확도
 *   [4] Cholesky      — SPD, 비양정치 거부
 *   [5] LDLT          — SPD, 음정치 거부 (패치 검증), 특이 거부
 *   [6] MGS-QR        — 직교성 Q^T Q = I, Rx = Q^T b
 *   [7] Householder QR — 직교성, 해 정확도
 *   [8] 엣지케이스     — 1x1, 비정방, 제로 나눗셈 예외
 */

#include "Optimization/Matrix/MatrixEngine.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>

// ================================================================
// 테스트 프레임워크 (의존성 없는 단독 실행)
// ================================================================
static int  g_total  = 0;
static int  g_passed = 0;
static int  g_failed = 0;

static constexpr double TOL = 1e-9;   // 수치 허용 오차

void check(bool cond, const std::string& name) {
    ++g_total;
    if (cond) {
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        ++g_failed;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

void section(const std::string& title) {
    std::cout << "\n── " << title << " ──\n";
}

// ================================================================
// 보조 함수
// ================================================================

/** 두 행렬의 Frobenius 오차가 tol 이내인지 확인 */
template <typename T, size_t R, size_t C>
bool mat_near(const StaticMatrix<T,R,C>& A,
              const StaticMatrix<T,R,C>& B,
              double tol = TOL) {
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            if (std::abs(A(static_cast<int>(i), static_cast<int>(j))
                       - B(static_cast<int>(i), static_cast<int>(j))) > tol)
                return false;
    return true;
}

/** 단위행렬 생성 */
template <typename T, size_t N>
StaticMatrix<T,N,N> eye() {
    StaticMatrix<T,N,N> I;
    for (size_t i = 0; i < N; ++i)
        I(static_cast<int>(i), static_cast<int>(i)) = static_cast<T>(1);
    return I;
}

/** 벡터 두 원소 간 오차 확인 */
template <typename T, size_t N>
bool vec_near(const StaticVector<T,N>& a,
              const StaticVector<T,N>& b,
              double tol = TOL) {
    for (size_t i = 0; i < N; ++i)
        if (std::abs(a(i) - b(i)) > tol)
            return false;
    return true;
}

// ================================================================
// [1] 메모리 레이아웃
// ================================================================
void test_memory_layout() {
    section("[1] 메모리 레이아웃");

    // Col-major 인덱싱 검증: (r, c) → data[c*Rows + r]
    StaticMatrix<double, 3, 3> A;
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;

    const double* p = A.data_ptr();
    // Col-major: 0열 [1,4,7], 1열 [2,5,8], 2열 [3,6,9]
    check(p[0] == 1.0 && p[1] == 4.0 && p[2] == 7.0, "Col-major 0열 순서");
    check(p[3] == 2.0 && p[4] == 5.0 && p[5] == 8.0, "Col-major 1열 순서");
    check(p[6] == 3.0 && p[7] == 6.0 && p[8] == 9.0, "Col-major 2열 순서");

    // alignas(64) 확인
    check(reinterpret_cast<uintptr_t>(A.data_ptr()) % 64 == 0, "64바이트 정렬 보장");

    // 선형 인덱서 operator()(size_t) 일관성
    check(A(0u) == 1.0 && A(1u) == 4.0, "선형 인덱서 일관성");
}

// ================================================================
// [2] 산술 연산자
// ================================================================
void test_arithmetic() {
    section("[2] 산술 연산자");

    StaticMatrix<double, 2, 2> A, B;
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;

    // 덧셈
    auto C = A + B;
    check(C(0,0)==6 && C(0,1)==8 && C(1,0)==10 && C(1,1)==12, "operator+");

    // 뺄셈
    auto D = B - A;
    check(D(0,0)==4 && D(0,1)==4 && D(1,0)==4 && D(1,1)==4, "operator-");

    // 스칼라 곱
    auto E = A * 3.0;
    check(E(0,0)==3 && E(1,1)==12, "operator*(scalar)");

    // 스칼라 나눗셈
    auto F = B / 2.0;
    check(std::abs(F(0,0)-2.5)<TOL && std::abs(F(1,1)-4.0)<TOL, "operator/(scalar)");

    // 제로 나눗셈 예외
    bool threw = false;
    try { auto G = A / 0.0; }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "operator/(0) — 예외 발생");

    // 행렬 곱: 2x2 × 2x2
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // AB = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    auto AB = A * B;
    check(AB(0,0)==19 && AB(0,1)==22 && AB(1,0)==43 && AB(1,1)==50, "operator*(matrix) 2x2");

    // 비정방 행렬 곱: 2x3 × 3x2
    StaticMatrix<double, 2, 3> P;
    StaticMatrix<double, 3, 2> Q;
    P(0,0)=1; P(0,1)=2; P(0,2)=3;
    P(1,0)=4; P(1,1)=5; P(1,2)=6;
    Q(0,0)=7; Q(0,1)=8;
    Q(1,0)=9; Q(1,1)=10;
    Q(2,0)=11; Q(2,1)=12;
    // PQ[0][0] = 1*7+2*9+3*11 = 58
    // PQ[0][1] = 1*8+2*10+3*12 = 64
    // PQ[1][0] = 4*7+5*9+6*11 = 139
    // PQ[1][1] = 4*8+5*10+6*12 = 154
    auto PQ = P * Q;
    check(PQ(0,0)==58 && PQ(0,1)==64 && PQ(1,0)==139 && PQ(1,1)==154,
          "operator*(matrix) 2x3 × 3x2 비정방");

    // A*I = A 항등원 검증
    auto I2 = eye<double, 2>();
    check(mat_near(A * I2, A), "A * I = A (항등원)");
}

// ================================================================
// [3] LU 분해 / 해법
// ================================================================
void test_LU() {
    section("[3] LU 분해 / 해법");

    // 3x3 일반 행렬
    // A = [[2,1,1],[4,3,3],[8,7,9]]
    // 해: Ax = b,  b = [1,1,1]^T  →  x = [1, -1, 0]^T (손계산)
    StaticMatrix<double,3,3> A;
    A(0,0)=2; A(0,1)=1; A(0,2)=1;
    A(1,0)=4; A(1,1)=3; A(1,2)=3;
    A(2,0)=8; A(2,1)=7; A(2,2)=9;

    StaticVector<double,3> b;
    b(0u)=1; b(1u)=1; b(2u)=1;

    bool ok = A.LU_decompose();
    check(ok, "LU_decompose 성공 반환");

    auto x = A.LU_solve(b);
    // residual 검증: 원본 A를 따로 보관해야 하므로, Ax-b 대신
    // 알려진 해로 직접 검증
    // A = [[2,1,1],[4,3,3],[8,7,9]], x = [1,-1,0] → Ax = [1,1,1]
    StaticVector<double,3> x_ref;
    x_ref(0u)=1.0; x_ref(1u)=-1.0; x_ref(2u)=0.0;
    check(vec_near(x, x_ref, 1e-9), "LU_solve 해 정확도 (3x3)");

    // 특이행렬 거부
    StaticMatrix<double,2,2> S;
    S(0,0)=1; S(0,1)=2;
    S(1,0)=2; S(1,1)=4;  // rank-1
    check(!S.LU_decompose(), "특이행렬 → LU_decompose false 반환");

    // 1x1 엣지케이스
    StaticMatrix<double,1,1> M1;
    M1(0,0) = 7.0;
    StaticVector<double,1> b1; b1(0u) = 21.0;
    M1.LU_decompose();
    auto x1 = M1.LU_solve(b1);
    check(std::abs(x1(0u) - 3.0) < TOL, "LU 1x1 엣지케이스");
}

// ================================================================
// [4] Cholesky 분해 / 해법
// ================================================================
void test_Cholesky() {
    section("[4] Cholesky 분해 / 해법");

    // SPD 행렬: A = [[4,2],[2,3]]
    // L = [[2,0],[1, sqrt(2)]]
    // b = [2,3]^T → x = A^{-1}b
    StaticMatrix<double,2,2> A;
    A(0,0)=4; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;

    StaticVector<double,2> b;
    b(0u)=2; b(1u)=3;

    bool ok = A.Cholesky_decompose();
    check(ok, "Cholesky_decompose SPD 성공");

    // L[0][0] = 2, L[1][0] = 1, L[1][1] = sqrt(2)
    check(std::abs(A(0,0) - 2.0) < TOL, "L[0][0] = 2");
    check(std::abs(A(1,0) - 1.0) < TOL, "L[1][0] = 1");
    check(std::abs(A(1,1) - std::sqrt(2.0)) < TOL, "L[1][1] = sqrt(2)");

    auto x = A.Cholesky_solve(b);
    // A^{-1} [[4,2],[2,3]] x = [2,3]
    // det=8, inv = (1/8)[[3,-2],[-2,4]]
    // x = (1/8)[3*2-2*3, -2*2+4*3] = (1/8)[0,8] = [0,1]
    StaticVector<double,2> x_ref;
    x_ref(0u)=0.0; x_ref(1u)=1.0;
    check(vec_near(x, x_ref, 1e-9), "Cholesky_solve 해 정확도");

    // 4x4 SPD 검증 (A = R^T R, 임의 R)
    StaticMatrix<double,4,4> R4, A4;
    // 간단한 대각 우세 SPD
    A4(0,0)=4; A4(0,1)=2; A4(0,2)=0; A4(0,3)=0;
    A4(1,0)=2; A4(1,1)=5; A4(1,2)=1; A4(1,3)=0;
    A4(2,0)=0; A4(2,1)=1; A4(2,2)=6; A4(2,3)=1;
    A4(3,0)=0; A4(3,1)=0; A4(3,2)=1; A4(3,3)=7;
    check(A4.Cholesky_decompose(), "Cholesky 4x4 SPD 성공");

    // 비양정치 거부: A = [[1,2],[2,1]] (eigenvalues: 3, -1)
    StaticMatrix<double,2,2> B;
    B(0,0)=1; B(0,1)=2;
    B(1,0)=2; B(1,1)=1;
    check(!B.Cholesky_decompose(), "비양정치 행렬 → Cholesky false 반환");

    // 제로 대각 행렬 거부
    StaticMatrix<double,2,2> Z;
    Z(0,0)=0; Z(0,1)=0;
    Z(1,0)=0; Z(1,1)=1;
    check(!Z.Cholesky_decompose(), "d=0 → Cholesky false 반환");
}

// ================================================================
// [5] LDLT 분해 / 해법 (음정치 거부 패치 핵심 검증)
// ================================================================
void test_LDLT() {
    section("[5] LDLT 분해 / 해법");

    // SPD 행렬: A = [[4,2],[2,3]]
    // D = [4, 2], L = [[1,0],[0.5,1]]
    StaticMatrix<double,2,2> A;
    A(0,0)=4; A(0,1)=2;
    A(1,0)=2; A(1,1)=3;

    StaticVector<double,2> b;
    b(0u)=2; b(1u)=3;

    bool ok = A.LDLT_decompose();
    check(ok, "LDLT_decompose SPD 성공");

    // D[0][0] = 4, L[1][0] = 0.5, D[1][1] = 3 - 0.5²*4 = 2
    check(std::abs(A(0,0) - 4.0) < TOL, "D[0] = 4");
    check(std::abs(A(1,0) - 0.5) < TOL, "L[1][0] = 0.5");
    check(std::abs(A(1,1) - 2.0) < TOL, "D[1] = 2");

    auto x = A.LDLT_solve(b);
    // A^{-1}b = [0,1] (Cholesky와 동일한 행렬)
    StaticVector<double,2> x_ref;
    x_ref(0u)=0.0; x_ref(1u)=1.0;
    check(vec_near(x, x_ref, 1e-9), "LDLT_solve 해 정확도");

    // ── 핵심 패치 검증: 음정치 행렬 거부 ──
    // A = [[1,2],[2,1]]: D_jj = 1 (pass), D_jj = 1 - 4*1 = -3 (음수 → 거부)
    StaticMatrix<double,2,2> Neg;
    Neg(0,0)=1; Neg(0,1)=2;
    Neg(1,0)=2; Neg(1,1)=1;
    check(!Neg.LDLT_decompose(), "음정치 행렬 → LDLT false 반환 [패치 검증]");

    // 특이행렬 거부: D_jj = 0
    StaticMatrix<double,2,2> Sing;
    Sing(0,0)=0; Sing(0,1)=0;
    Sing(1,0)=0; Sing(1,1)=1;
    check(!Sing.LDLT_decompose(), "D=0 특이행렬 → LDLT false 반환");

    // 4x4 SPD
    StaticMatrix<double,4,4> A4;
    A4(0,0)=4; A4(0,1)=2; A4(0,2)=0; A4(0,3)=0;
    A4(1,0)=2; A4(1,1)=5; A4(1,2)=1; A4(1,3)=0;
    A4(2,0)=0; A4(2,1)=1; A4(2,2)=6; A4(2,3)=1;
    A4(3,0)=0; A4(3,1)=0; A4(3,2)=1; A4(3,3)=7;
    check(A4.LDLT_decompose(), "LDLT 4x4 SPD 성공");
}

// ================================================================
// [6] MGS-QR 분해 / 해법
// ================================================================
void test_QR_MGS() {
    section("[6] MGS-QR 분해 / 해법");

    // 3x2 과결정 시스템 (Rows > Cols)
    StaticMatrix<double,3,2> A;
    A(0,0)=1; A(0,1)=0;
    A(1,0)=0; A(1,1)=1;
    A(2,0)=1; A(2,1)=1;

    StaticMatrix<double,3,2> Q = A;  // in-place → Q
    StaticMatrix<double,2,2> R;

    bool ok = Q.QR_decompose_MGS(R);
    check(ok, "QR_MGS decompose 성공");

    // Q^T Q = I 검증 (직교성)
    // Q^T 는 2x3, Q는 3x2, 결과는 2x2
    StaticMatrix<double,2,2> QtQ;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
                s += Q(k,i) * Q(k,j);
            QtQ(i,j) = s;
        }
    check(mat_near(QtQ, eye<double,2>(), 1e-9), "Q^T Q = I (MGS 직교성)");

    // R이 상삼각인지 확인
    check(std::abs(R(1,0)) < TOL, "R 하삼각 원소 = 0");

    // 최소제곱 해 검증: A의 full column rank → 유일 최소제곱해 존재
    // Ax = b, b = [1,1,0]^T
    // 정규방정식 A^T A x = A^T b
    StaticVector<double,3> b;
    b(0u)=1; b(1u)=1; b(2u)=0;

    auto x = Q.QR_solve(R, b);
    // 검증: ||Ax - b||를 직접 계산하여 최소제곱 조건 확인
    // A^T(Ax - b) = 0 이어야 함 (법선방정식)
    // A = 원본 A (MGS 전), x를 대입
    StaticMatrix<double,3,2> A_orig;
    A_orig(0,0)=1; A_orig(0,1)=0;
    A_orig(1,0)=0; A_orig(1,1)=1;
    A_orig(2,0)=1; A_orig(2,1)=1;

    double res0 = 0, res1 = 0;
    for (int k = 0; k < 3; ++k) {
        double r = b(static_cast<size_t>(k));
        for (int j = 0; j < 2; ++j)
            r -= A_orig(k,j) * x(static_cast<size_t>(j));
        res0 += A_orig(k,0) * r;
        res1 += A_orig(k,1) * r;
    }
    check(std::abs(res0) < 1e-9 && std::abs(res1) < 1e-9,
          "QR_MGS 최소제곱 법선방정식 A^T(Ax-b)=0");

    // 선형독립 아닌 열 → 거부
    StaticMatrix<double,3,2> Dep;
    Dep(0,0)=1; Dep(0,1)=2;
    Dep(1,0)=2; Dep(1,1)=4;
    Dep(2,0)=3; Dep(2,1)=6;
    StaticMatrix<double,2,2> Rdep;
    check(!Dep.QR_decompose_MGS(Rdep), "선형종속 열 → QR_MGS false 반환");
}

// ================================================================
// [7] Householder QR 분해 / 해법
// ================================================================
void test_QR_Householder() {
    section("[7] Householder QR 분해 / 해법");

    // 4x3 과결정 시스템
    StaticMatrix<double,4,3> A;
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    A(2,0)=7; A(2,1)=8; A(2,2)=10;
    A(3,0)=0; A(3,1)=1; A(3,2)=0;

    StaticMatrix<double,4,3> A_orig = A;
    StaticVector<double,3> tau;

    bool ok = A.QR_decompose_Householder(tau);
    check(ok, "QR_Householder decompose 성공");

    // Householder In-place 표현 특성 확인:
    //   상삼각 A(i,j) (j>=i) 가 R이고,
    //   하삼각 A(k,i) (k>i) 는 정규화된 Householder vector 저장에 사용됨.
    //   → 하삼각이 0이 아닌 것이 정상. 대신 대각 원소 R[i][i]의 부호(sign 반전) 확인.
    // R[0][0]은 - sign * norm이므로 절댓값이 0보다 커야 함.
    check(std::abs(A(0,0)) > 1e-9, "R[0][0] ≠ 0 (Householder 대각)");
    check(std::abs(A(1,1)) > 1e-9, "R[1][1] ≠ 0 (Householder 대각)");
    check(std::abs(A(2,2)) > 1e-9, "R[2][2] ≠ 0 (Householder 대각)");

    // 최소제곱 해: b = [1,2,3,4]^T
    StaticVector<double,4> b;
    b(0u)=1; b(1u)=2; b(2u)=3; b(3u)=4;

    auto x = A.QR_solve_Householder(tau, b);

    // 법선방정식 A_orig^T (A_orig x - b) = 0 검증
    double res[3] = {0,0,0};
    for (int k = 0; k < 4; ++k) {
        double r = b(static_cast<size_t>(k));
        for (int j = 0; j < 3; ++j)
            r -= A_orig(k,j) * x(static_cast<size_t>(j));
        for (int i = 0; i < 3; ++i)
            res[i] += A_orig(k,i) * r;
    }
    bool normal_ok = std::abs(res[0]) < 1e-8
                  && std::abs(res[1]) < 1e-8
                  && std::abs(res[2]) < 1e-8;
    check(normal_ok, "QR_Householder 최소제곱 법선방정식 A^T(Ax-b)=0");

    // 정방 풀랭크 시스템 (유일해)
    // A = [[2,1],[1,3]], b = [5,10]  → x = [1,3]
    StaticMatrix<double,2,2> A2;
    A2(0,0)=2; A2(0,1)=1;
    A2(1,0)=1; A2(1,1)=3;
    StaticVector<double,2> b2, tau2;
    b2(0u)=5; b2(1u)=10;
    A2.QR_decompose_Householder(tau2);
    auto x2 = A2.QR_solve_Householder(tau2, b2);
    // 2x+y=5, x+3y=10 → x=1, y=3
    check(std::abs(x2(0u)-1.0)<1e-9 && std::abs(x2(1u)-3.0)<1e-9,
          "QR_Householder 정방 유일해 정확도");
}

// ================================================================
// [8] 엣지케이스
// ================================================================
void test_edge_cases() {
    section("[8] 엣지케이스");

    // 1x1 모든 분해
    StaticMatrix<double,1,1> M;
    M(0,0) = 4.0;
    StaticVector<double,1> bv; bv(0u) = 8.0;

    StaticMatrix<double,1,1> Mlu = M;
    check(Mlu.LU_decompose(), "1x1 LU_decompose");
    auto xlu = Mlu.LU_solve(bv);
    check(std::abs(xlu(0u)-2.0)<TOL, "1x1 LU_solve = 2");

    StaticMatrix<double,1,1> Mch = M;
    check(Mch.Cholesky_decompose(), "1x1 Cholesky_decompose");
    auto xch = Mch.Cholesky_solve(bv);
    check(std::abs(xch(0u)-2.0)<TOL, "1x1 Cholesky_solve = 2");

    StaticMatrix<double,1,1> Mld = M;
    check(Mld.LDLT_decompose(), "1x1 LDLT_decompose");
    auto xld = Mld.LDLT_solve(bv);
    check(std::abs(xld(0u)-2.0)<TOL, "1x1 LDLT_solve = 2");

    // StaticVector 타입 확인 (N×1 특수화)
    StaticVector<double,3> v;
    v(0u)=1; v(1u)=2; v(2u)=3;
    check(v(0,0)==1.0 && v(1,0)==2.0, "StaticVector (r,c=0) 접근");

    // 대형 정방 행렬 기본 연산 컴파일·실행 확인 (8x8)
    StaticMatrix<double,8,8> Big;
    for (int i = 0; i < 8; ++i)
        Big(i,i) = static_cast<double>(i+2);  // 대각 우세
    check(Big.LU_decompose(), "8x8 대각 행렬 LU_decompose");
}

// ================================================================
// main
// ================================================================
int main() {
    std::cout << std::string(52, '=') << "\n";
    std::cout << "  StaticMatrix.hpp 수치 검증 테스트 스위트\n";
    std::cout << std::string(52, '=') << "\n";

    test_memory_layout();
    test_arithmetic();
    test_LU();
    test_Cholesky();
    test_LDLT();
    test_QR_MGS();
    test_QR_Householder();
    test_edge_cases();

    std::cout << "\n" << std::string(52, '=') << "\n";
    std::cout << "  결과: " << g_passed << " / " << g_total << " PASSED";
    if (g_failed > 0)
        std::cout << "  (" << g_failed << " FAILED)";
    std::cout << "\n" << std::string(52, '=') << "\n";

    return g_failed == 0 ? 0 : 1;
}