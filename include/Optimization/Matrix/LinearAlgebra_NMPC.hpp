#ifndef OPTIMIZATION_LINEAR_ALGEBRA_NMPC_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_NMPC_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"
#include "Optimization/Matrix/LinearAlgebra_LDLT.hpp"

namespace Optimization {
namespace linalg {
template <typename T, size_t Rows, size_t Cols, size_t P_Dim>
StaticMatrix<T, Cols, Cols> quadratic_multiply(const StaticMatrix<T, Rows, Cols>& A,
                                               const StaticMatrix<T, P_Dim, P_Dim>& P) {
    StaticMatrix<T, P_Dim, Cols> Temp = P * A;
    return transpose(A) * Temp;
}

template <typename T, size_t Rows, size_t Cols, size_t B_Cols>
StaticMatrix<T, Rows, B_Cols> solve_multiple(const StaticMatrix<T, Rows, Cols>& mat,
                                             const StaticMatrix<T, Rows, B_Cols>& B) {
    static_assert(Rows == Cols, "Solver requires a square matrix");
    StaticMatrix<T, Rows, B_Cols> X;
    for (size_t j = 0; j < B_Cols; ++j) {
        StaticVector<T, Rows> b_col;
        for (size_t i = 0; i < Rows; ++i) {
            b_col(i) = B(static_cast<int>(i), static_cast<int>(j));
        }
        StaticVector<T, Rows> x_col = LDLT_solve(mat, b_col);

        for (size_t i = 0; i < Rows; ++i) {
            X(static_cast<int>(i), static_cast<int>(j)) = x_col(i);
        }
    }
    return X;
}
}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_NMPC_HPP_