// include/Optimization/Matrix/SparseMatrixEngine.hpp
#ifndef OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_
#define OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

// [Architect's True Sparse] 동적 할당이 완벽히 배제된 정적 CSR 행렬 구조
template <typename T, size_t Rows, size_t Cols, size_t MaxNNZ>
class StaticSparseMatrix {
public:
    StaticVector<T, MaxNNZ> values;       // 0이 아닌 실제 값들
    StaticVector<int, MaxNNZ> col_index;  // 해당 값의 열(Column) 인덱스
    StaticVector<int, Rows + 1> row_ptr;  // 각 행(Row)이 시작되는 인덱스 포인터
    size_t nnz_count;                     // 현재 채워진 Non-zero 개수

    StaticSparseMatrix() : nnz_count(0) {
        values.set_zero();
        col_index.set_zero();
        row_ptr.set_zero();
    }

    // 0이 아닌 값만 밀어 넣음 (메모리 폭발 방지)
    void add_value(int row, int col, T val) {
        if (nnz_count >= MaxNNZ) return; // 오버플로우 방탄조끼
        values(nnz_count) = val;
        col_index(nnz_count) = col;
        row_ptr(row + 1)++;
        nnz_count++;
    }

    void finalize() {
        for (size_t i = 0; i < Rows; ++i) {
            row_ptr(i + 1) += row_ptr(i);
        }
    }

    // 핵심: O(N^2)이 아닌 O(NNZ) 속도의 초고속 행렬-벡터 곱셈 (SpMV)
    void multiply(const StaticVector<T, Cols>& x, StaticVector<T, Rows>& y) const {
        y.set_zero();
        for (size_t i = 0; i < Rows; ++i) {
            T sum = 0.0;
            for (int j = row_ptr(i); j < row_ptr(i + 1); ++j) {
                sum += values(j) * x(col_index(j));
            }
            y(static_cast<int>(i)) = sum;
        }
    }
};

} // namespace Optimization
#endif