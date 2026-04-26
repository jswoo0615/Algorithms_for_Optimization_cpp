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
    StaticMatrix<double, 5, 5> KKT; // 거대 타겟 행렬
    StaticMatrix<double, 2, 2> H_uu; // 작은 블록

    H_uu(0, 0) = 1.0; H_uu(0, 1) = 2.0;
    H_uu(1, 0) = 3.0; H_uu(1, 1) = 4.0;

    // 1. 블록 삽입 (std::copy 고속 복사 검증)
    KKT.insert_block(2, 2, H_uu);

    EXPECT_DOUBLE_EQ(KKT(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(KKT(2, 3), 2.0);
    EXPECT_DOUBLE_EQ(KKT(3, 2), 3.0);
    EXPECT_DOUBLE_EQ(KKT(3, 3), 4.0);
    EXPECT_DOUBLE_EQ(KKT(0, 0), 0.0); // 엉뚱한 곳이 오염되지 않았는지 확인

    // 2. 블록 추출 (std::copy 고속 복사 검증)
    auto Extracted = KKT.extract_block<2, 2>(2, 2);
    EXPECT_DOUBLE_EQ(Extracted(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(Extracted(1, 1), 4.0);
}

// =========================================================================
// [Test Suite 2] Layer 2: Basic Linear Algebra
// =========================================================================
TEST(LinearAlgebraTest, MatrixMultiplication) {
    // J-K-I 루프 캐시 최적화 곱셈 검증
    StaticMatrix<double, 2, 3> A;
    StaticMatrix<double, 3, 2> B;

    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    B(0,0)=7;  B(0,1)=8;
    B(1,0)=9;  B(1,1)=10;
    B(2,0)=11; B(2,1)=12;

    auto C = A * B; // Layer 2 오버로딩 호출

    EXPECT_DOUBLE_EQ(C(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 154.0);
}

TEST(LinearAlgebraTest, Transpose) {
    StaticMatrix<double, 2, 3> A;
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;

    auto A_T = transpose(A); // Layer 2 자유 함수 호출

    EXPECT_DOUBLE_EQ(A_T(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A_T(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(A_T(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(A_T(0, 1), 4.0);
}

// =========================================================================
// [Test Suite 3] Layer 2: Decomposition & Solvers
// =========================================================================
TEST(LinearAlgebraTest, LUSolver) {
    StaticMatrix<double, 3, 3> A;
    StaticVector<double, 3> b;
    StaticVector<int, 3> P; // [추가] 피벗 배열

    A(0,0)=3; A(0,1)=2; A(0,2)=-1;
    A(1,0)=2; A(1,1)=-2; A(1,2)=4;
    A(2,0)=-1; A(2,1)=0.5; A(2,2)=-1;

    b(0)=1; b(1)=-2; b(2)=0;

    bool success = LU_decompose(A, P); // [수정] 피벗 배열 전달
    EXPECT_TRUE(success) << "LU Decomposition failed on a non-singular matrix.";

    auto x = LU_solve(A, P, b); // [수정] 피벗 배열 전달

    // 정답: x = [1, -2, -2]
    EXPECT_NEAR(x(0), 1.0, 1e-6);
    EXPECT_NEAR(x(1), -2.0, 1e-6);
    EXPECT_NEAR(x(2), -2.0, 1e-6);
}

TEST(LinearAlgebraTest, LDLTSolver) {
    StaticMatrix<double, 3, 3> A;
    StaticVector<double, 3> b;

    // Symmetric Positive Definite 행렬 구성
    A(0,0)=4;  A(0,1)=12; A(0,2)=-16;
    A(1,0)=12; A(1,1)=37; A(1,2)=-43;
    A(2,0)=-16; A(2,1)=-43; A(2,2)=98;

    b(0)=1; b(1)=2; b(2)=3;

    bool success = LDLT_decompose(A);
    EXPECT_TRUE(success) << "LDLT Decomposition failed on an SPD matrix.";

    auto x = LDLT_solve(A, b);

    // Ax = b 역검증 (Residual 체크)
    StaticMatrix<double, 3, 3> A_orig;
    A_orig(0,0)=4; A_orig(0,1)=12; A_orig(0,2)=-16;
    A_orig(1,0)=12; A_orig(1,1)=37; A_orig(1,2)=-43;
    A_orig(2,0)=-16; A_orig(2,1)=-43; A_orig(2,2)=98;

    auto b_calc = A_orig * x;

    EXPECT_NEAR(b_calc(0), b(0), 1e-6);
    EXPECT_NEAR(b_calc(1), b(1), 1e-6);
    EXPECT_NEAR(b_calc(2), b(2), 1e-6);
}