#include <gtest/gtest.h>
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"

using namespace Optimization;
using namespace Optimization::linalg;

// =========================================================================
// [Test Suite 1] Layer 1: Memory & Block Operations
// =========================================================================
TEST(StaticMatrixTest, MemoryAlignmentAndZeroing) {
    StaticMatrix<double, 4, 4> A;
    
    // 1. 메모리 64-byte 정렬 확인 (SIMD 대비)
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(A.data_ptr()) % 64, 0) 
        << "CRITICAL: Matrix memory is not 64-byte aligned!";

    // 2. 초기화 확인
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(A(i), 0.0);
    }
}

TEST(StaticMatrixTest, BlockInsertAndExtract) {
    StaticMatrix<double, 5, 5> KKT; 
    StaticMatrix<double, 2, 2> H_uu; 
    
    H_uu(0, 0) = 1.0; H_uu(0, 1) = 2.0;
    H_uu(1, 0) = 3.0; H_uu(1, 1) = 4.0;

    // 1. 블록 삽입
    KKT.insert_block(2, 2, H_uu);
    EXPECT_DOUBLE_EQ(KKT(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(KKT(2, 3), 2.0);
    EXPECT_DOUBLE_EQ(KKT(3, 2), 3.0);
    EXPECT_DOUBLE_EQ(KKT(3, 3), 4.0);
    EXPECT_DOUBLE_EQ(KKT(0, 0), 0.0); 

    // 2. 블록 추출
    auto Extracted = KKT.extract_block<2, 2>(2, 2);
    EXPECT_DOUBLE_EQ(Extracted(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(Extracted(1, 1), 4.0);

    // 3. 전치 블록 삽입
    StaticMatrix<double, 5, 5> KKT2;
    KKT2.insert_transposed_block(2, 2, H_uu);
    EXPECT_DOUBLE_EQ(KKT2(2, 2), 1.0); // (0,0)
    EXPECT_DOUBLE_EQ(KKT2(2, 3), 3.0); // (1,0)이 전치됨
    EXPECT_DOUBLE_EQ(KKT2(3, 2), 2.0); // (0,1)이 전치됨
    EXPECT_DOUBLE_EQ(KKT2(3, 3), 4.0); // (1,1)
}

// =========================================================================
// [Test Suite 2] Layer 2: Core Linear Algebra
// =========================================================================
TEST(LinearAlgebraCoreTest, ArithmeticAndMultiplication) {
    StaticMatrix<double, 2, 3> A;
    StaticMatrix<double, 3, 2> B;

    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    B(0,0)=7;  B(0,1)=8;
    B(1,0)=9;  B(1,1)=10;
    B(2,0)=11; B(2,1)=12;

    // J-K-I 루프 캐시 최적화 곱셈 검증
    auto C = A * B; 
    EXPECT_DOUBLE_EQ(C(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 154.0);

    // 스칼라 연산 검증
    auto D = C / 2.0;
    EXPECT_DOUBLE_EQ(D(0, 0), 29.0);
    EXPECT_DOUBLE_EQ(D(1, 1), 77.0);

    // 전치 연산 검증
    auto A_T = transpose(A); 
    EXPECT_DOUBLE_EQ(A_T(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A_T(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(A_T(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(A_T(0, 1), 4.0);
}

// =========================================================================
// [Test Suite 3] Layer 2: LU Solver (Partial Pivoting)
// =========================================================================
TEST(LinearAlgebraLUSolverTest, SuccessCase) {
    StaticMatrix<double, 3, 3> A;
    StaticVector<double, 3> b;
    StaticVector<int, 3> P;

    A(0,0)=3; A(0,1)=2; A(0,2)=-1;
    A(1,0)=2; A(1,1)=-2; A(1,2)=4;
    A(2,0)=-1; A(2,1)=0.5; A(2,2)=-1;

    b(0)=1; b(1)=-2; b(2)=0;

    MathStatus status = LU_decompose(A, P);
    EXPECT_EQ(status, MathStatus::SUCCESS);

    auto x = LU_solve(A, P, b);
    EXPECT_NEAR(x(0), 1.0, 1e-6);
    EXPECT_NEAR(x(1), -2.0, 1e-6);
    EXPECT_NEAR(x(2), -2.0, 1e-6);
}

TEST(LinearAlgebraLUSolverTest, SingularCase) {
    StaticMatrix<double, 3, 3> A;
    StaticVector<int, 3> P;

    // 선형 종속인 행렬 구성 (Row 0과 Row 1이 동일)
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=1; A(1,1)=2; A(1,2)=3;
    A(2,0)=4; A(2,1)=5; A(2,2)=6;

    MathStatus status = LU_decompose(A, P);
    EXPECT_EQ(status, MathStatus::SINGULAR) << "Fail-fast mechanism broken: Solver did not detect singular matrix.";
}

// =========================================================================
// [Test Suite 4] Layer 2: Cholesky & LDLT Solvers
// =========================================================================
TEST(LinearAlgebraCholeskySolverTest, SuccessCase) {
    StaticMatrix<double, 3, 3> A;
    StaticVector<double, 3> b;

    A(0,0)=4;  A(0,1)=12; A(0,2)=-16;
    A(1,0)=12; A(1,1)=37; A(1,2)=-43;
    A(2,0)=-16; A(2,1)=-43; A(2,2)=98;

    b(0)=1; b(1)=2; b(2)=3;

    MathStatus status = Cholesky_decompose(A);
    EXPECT_EQ(status, MathStatus::SUCCESS);

    auto x = Cholesky_solve(A, b);

    // Ax = b 역검증
    StaticMatrix<double, 3, 3> A_orig;
    A_orig(0,0)=4; A_orig(0,1)=12; A_orig(0,2)=-16;
    A_orig(1,0)=12; A_orig(1,1)=37; A_orig(1,2)=-43;
    A_orig(2,0)=-16; A_orig(2,1)=-43; A_orig(2,2)=98;

    auto b_calc = A_orig * x;
    EXPECT_NEAR(b_calc(0), b(0), 1e-6);
    EXPECT_NEAR(b_calc(1), b(1), 1e-6);
    EXPECT_NEAR(b_calc(2), b(2), 1e-6);
}

TEST(LinearAlgebraLDLTSolverTest, SuccessCase) {
    StaticMatrix<double, 3, 3> A;
    StaticVector<double, 3> b;

    A(0,0)=4;  A(0,1)=12; A(0,2)=-16;
    A(1,0)=12; A(1,1)=37; A(1,2)=-43;
    A(2,0)=-16; A(2,1)=-43; A(2,2)=98;

    b(0)=1; b(1)=2; b(2)=3;

    MathStatus status = LDLT_decompose(A);
    EXPECT_EQ(status, MathStatus::SUCCESS);

    auto x = LDLT_solve(A, b);

    StaticMatrix<double, 3, 3> A_orig;
    A_orig(0,0)=4; A_orig(0,1)=12; A_orig(0,2)=-16;
    A_orig(1,0)=12; A_orig(1,1)=37; A_orig(1,2)=-43;
    A_orig(2,0)=-16; A_orig(2,1)=-43; A_orig(2,2)=98;

    auto b_calc = A_orig * x;
    EXPECT_NEAR(b_calc(0), b(0), 1e-6);
    EXPECT_NEAR(b_calc(1), b(1), 1e-6);
    EXPECT_NEAR(b_calc(2), b(2), 1e-6);
}

// =========================================================================
// [Test Suite 5] Layer 2: QR Solvers
// =========================================================================
TEST(LinearAlgebraQRSolverTest, MGS_SuccessCase) {
    StaticMatrix<double, 4, 3> A;
    StaticMatrix<double, 3, 3> R;
    StaticVector<double, 4> b;

    A(0,0)=1; A(0,1)=-1; A(0,2)=4;
    A(1,0)=1; A(1,1)=4;  A(1,2)=-2;
    A(2,0)=1; A(2,1)=4;  A(2,2)=2;
    A(3,0)=1; A(3,1)=-1; A(3,2)=0;

    b(0)=1; b(1)=2; b(2)=3; b(3)=4;

    MathStatus status = QR_decompose_MGS(A, R);
    EXPECT_EQ(status, MathStatus::SUCCESS);

    auto x = QR_solve(A, R, b);
    
    // x가 계산되었는지 단순 검증 (MGS 최소자승해)
    EXPECT_FALSE(std::isnan(x(0)));
    EXPECT_FALSE(std::isnan(x(1)));
    EXPECT_FALSE(std::isnan(x(2)));
}

TEST(LinearAlgebraQRSolverTest, Householder_SuccessCase) {
    StaticMatrix<double, 4, 3> A;
    StaticVector<double, 3> tau;
    StaticVector<double, 4> b;

    A(0,0)=1; A(0,1)=-1; A(0,2)=4;
    A(1,0)=1; A(1,1)=4;  A(1,2)=-2;
    A(2,0)=1; A(2,1)=4;  A(2,2)=2;
    A(3,0)=1; A(3,1)=-1; A(3,2)=0;

    b(0)=1; b(1)=2; b(2)=3; b(3)=4;

    MathStatus status = QR_decompose_Householder(A, tau);
    EXPECT_EQ(status, MathStatus::SUCCESS);

    auto x = QR_solve_Householder(A, tau, b);
    
    EXPECT_FALSE(std::isnan(x(0)));
    EXPECT_FALSE(std::isnan(x(1)));
    EXPECT_FALSE(std::isnan(x(2)));
}

// =========================================================================
// [Test Suite 6] Layer 2: NMPC Specific Assembly
// =========================================================================
TEST(LinearAlgebraNMPCTest, QuadraticMultiply) {
    StaticMatrix<double, 2, 3> A;
    StaticMatrix<double, 2, 2> P;

    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    P(0,0)=2; P(0,1)=0;
    P(1,0)=0; P(1,1)=3;

    // Result = A^T * P * A (3x2 * 2x2 * 2x3 -> 3x3)
    auto Result = quadratic_multiply(A, P);

    // 수동 계산: P*A = [2 4 6; 12 15 18]
    // A^T * (P*A) 의 (0,0) 원소 = 1*2 + 4*12 = 50
    EXPECT_DOUBLE_EQ(Result(0, 0), 50.0);
}

TEST(LinearAlgebraNMPCTest, SolveMultiple) {
    StaticMatrix<double, 3, 3> A;
    StaticMatrix<double, 3, 2> B;

    A(0,0)=4;  A(0,1)=12; A(0,2)=-16;
    A(1,0)=12; A(1,1)=37; A(1,2)=-43;
    A(2,0)=-16; A(2,1)=-43; A(2,2)=98;

    B(0,0)=1; B(0,1)=2;
    B(1,0)=2; B(1,1)=4;
    B(2,0)=3; B(2,1)=6; // B의 두 번째 열은 첫 번째 열의 2배

    MathStatus status = LDLT_decompose(A);
    EXPECT_EQ(status, MathStatus::SUCCESS);

    auto X = solve_multiple(A, B);

    // X의 두 번째 열 결과는 첫 번째 열 결과의 2배여야 함 (선형성 검증)
    EXPECT_NEAR(X(0, 1), X(0, 0) * 2.0, 1e-6);
    EXPECT_NEAR(X(1, 1), X(1, 0) * 2.0, 1e-6);
    EXPECT_NEAR(X(2, 1), X(2, 0) * 2.0, 1e-6);
}