#include <iostream>
#include <iomanip>
#include <string>
#include "Optimization/Matrix/SparseMatrixEngine.hpp"

using namespace Optimization;

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  [Layer 2] Static CSR (Compressed Sparse Row) Engine Test\n";
    std::cout << std::string(70, '=') << "\n";

    // 테스트 행렬 A (3x3)
    // [ 10.0    0.0    0.0 ]
    // [  0.0   20.0   30.0 ]
    // [  0.0    0.0   40.0 ]
    // 총 9개의 원소 중 Non-zero는 4개. 
    
    constexpr size_t Rows = 3;
    constexpr size_t Cols = 3;
    constexpr size_t MaxNNZ = 4;

    StaticSparseMatrix<double, Rows, Cols, MaxNNZ> A;

    // 1. 값 주입 (반드시 Row 순서대로 주입해야 함)
    A.add_value(0, 0, 10.0); // 0행 0열
    A.add_value(1, 1, 20.0); // 1행 1열
    A.add_value(1, 2, 30.0); // 1행 2열
    A.add_value(2, 2, 40.0); // 2행 2열

    // 2. CSR 구조 압축 완료 (row_ptr 계산)
    A.finalize();

    // 3. 압축된 내부 구조 확인
    std::cout << "  [*] CSR Memory Layout Verification\n";
    
    std::cout << "      Values    : [ ";
    for(size_t i = 0; i < A.nnz_count; ++i) std::cout << std::setw(4) << A.values(static_cast<int>(i)) << " ";
    std::cout << "]\n";

    std::cout << "      Col_Index : [ ";
    for(size_t i = 0; i < A.nnz_count; ++i) std::cout << std::setw(4) << A.col_index(static_cast<int>(i)) << " ";
    std::cout << "]\n";

    std::cout << "      Row_Ptr   : [ ";
    for(size_t i = 0; i <= Rows; ++i) std::cout << std::setw(4) << A.row_ptr(static_cast<int>(i)) << " ";
    std::cout << "]\n\n";

    // 4. 초고속 행렬-벡터 곱셈 (SpMV) 테스트
    // x = [1.0, 2.0, 3.0]^T
    StaticVector<double, Cols> x;
    x(0) = 1.0; x(1) = 2.0; x(2) = 3.0;

    StaticVector<double, Rows> y;
    A.multiply(x, y);

    // 예상 결과 (y = A * x):
    // y_0 = 10*1 = 10
    // y_1 = 20*2 + 30*3 = 130
    // y_2 = 40*3 = 120
    std::cout << "  [*] SpMV (Sparse Matrix-Vector Multiplication) Test\n";
    std::cout << "      Input x   : [ " << x(0) << ", " << x(1) << ", " << x(2) << " ]\n";
    std::cout << "      Output y  : [ " << y(0) << ", " << y(1) << ", " << y(2) << " ]\n\n";

    // 5. 검증
    bool pass = (y(0) == 10.0) && (y(1) == 130.0) && (y(2) == 120.0);

    if (pass) {
        std::cout << "  [PASS] 동적 할당 없는 정적 CSR 행렬 압축 및 연산 검증 완료.\n";
        return 0;
    } else {
        std::cout << "  [FAIL] CSR 행렬 연산 오류 발생.\n";
        return 1;
    }
}