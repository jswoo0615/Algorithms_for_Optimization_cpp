#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"

namespace Optimization {
namespace linalg {
template <typename T, size_t Rows, size_t Cols>
MathStatus LDLT_decompose(StaticMatrix<T, Rows, Cols>& mat) {
    static_assert(Rows == Cols, "LDLT requires a square matrix");
    int N = static_cast<int>(Rows);

    for (int j = 0; j < N; ++j) {
        T sum_D = static_cast<T>(0);
        for (int k = 0; k < j; ++k) {
            T L_jk = mat(j, k);
            sum_D += L_jk * L_jk * mat(k, k);
        }
        T D_jj = mat(j, j) - sum_D;

        if (MathTraits<T>::near_zero(D_jj)) {
            return MathStatus::SINGULAR;
        }
        mat(j, j) = D_jj;

        for (int i = j + 1; i < N; ++i) {
            T sum_L = static_cast<T>(0);
            for (int k = 0; k < j; ++k) {
                sum_L += mat(i, k) * mat(j, k) * mat(k, k);
            }
            mat(i, j) = (mat(i, j) - sum_L) / mat(j, j);
        }
    }
    return MathStatus::SUCCESS;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Rows> LDLT_solve(const StaticMatrix<T, Rows, Cols>& mat,
                                 const StaticVector<T, Rows>& b) {
    static_assert(Rows == Cols, "Solve requires a square matrix");
    StaticVector<T, Rows> x, y, z;
    int N = static_cast<int>(Rows);

    for (int i = 0; i < N; ++i) {
        T sum = static_cast<T>(0);
        for (int k = 0; k < i; ++k) {
            sum += mat(i, k) * z(k);
        }
        z(i) = b(i) - sum;
    }
    for (int i = 0; i < N; ++i) {
        y(i) = z(i) / mat(i, i);
    }
    for (int i = N - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (int k = i + 1; k < N; ++k) {
            sum += mat(k, i) * x(k);
        }
        x(i) = y(i) - sum;
    }
    return x;
}
}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_
