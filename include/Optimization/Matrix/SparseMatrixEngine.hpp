#ifndef OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_
#define OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_

#include <cmath>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {
template <typename T, size_t Rows, size_t Cols, size_t MaxNNZ>
class StaticSparseMatrix {
   public:
    StaticVector<T, MaxNNZ> values;
    StaticVector<int, MaxNNZ> col_index;
    StaticVector<int, Rows + 1> row_ptr;
    size_t nnz_count;

    StaticSparseMatrix() : nnz_count(0) {
        values.set_zero();
        col_index.set_zero();
        row_ptr.set_zero();
    }

    void finalize() {
        for (size_t i = 0; i < Rows; ++i) {
            row_ptr(i + 1) += row_ptr(i);
        }
    }

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

    // 전치 행렬-벡터 곱셈 (y = A^T * x)
    void multiply_transpose(const StaticVector<T, Rows>& x, StaticVector<T, Cols>& y) const {
        y.set_zero();
        for (size_t i = 0; i < Rows; ++i) {
            T xi = x(i);
            for (int j = row_ptr(i); j < row_ptr(i + 1); ++j) {
                y(col_index(j)) += values(j) * xi;
            }
        }
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_