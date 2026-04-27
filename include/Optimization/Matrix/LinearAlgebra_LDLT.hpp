#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"

namespace Optimization {
namespace linalg {
template <typename T, size_t Rows, size_t Cols>
MathStatus LDLT_decompose(StaticMatrix<T, Rows, Cols>& mat) {
    static_assert(Rows == Cols, "LDLT requires a square matrix");
    for (size_t j = 0; j < Cols; ++j) {
        T sum_D = static_cast<T>(0);
        for (size_t k = 0; k < j; ++k) {
            T L_jk = mat(static_cast<int>(j), static_cast<int>(k));
            sum_D = L_jk * L_jk * mat(static_cast<int>(k), static_cast<int>(k));
        }
        T D_jj = mat(static_cast<int>(j), static_cast<int>(j)) - sum_D;
        if (MathTraits<T>::near_zero(D_jj)) {
            return MathStatus::SINGULAR;
        }
        mat(static_cast<int>(j), static_cast<int>(j)) = D_jj;

        for (size_t i = j + 1; i < Rows; ++i) {
            T sum_L = static_cast<T>(0);
            for (size_t k = 0; k < j; ++k) {
                sum_L += mat(static_cast<int>(i), static_cast<int>(k)) *
                         mat(static_cast<int>(j), static_cast<int>(k)) *
                         mat(static_cast<int>(k), static_cast<int>(k));
            }
            mat(static_cast<int>(i), static_cast<int>(j)) =
                (mat(static_cast<int>(i), static_cast<int>(j)) - sum_L) /
                mat(static_cast<int>(j), static_cast<int>(j));
        }
    }
    return MathStatus::SUCCESS;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Rows> LDLT_solve(const StaticMatrix<T, Rows, Cols>& mat,
                                 const StaticVector<T, Rows>& b) {
    static_assert(Rows == Cols, "Solve requires a square matrix");
    StaticVector<T, Rows> x, y, z;
    for (size_t i = 0; i < Rows; ++i) {
        T sum = static_cast<T>(0);
        for (size_t k = 0; k < i; ++k) {
            sum += mat(static_cast<int>(i), static_cast<int>(k)) * z(k);
        }
        z(i) = b(i) - sum;
    }

    for (size_t i = 0; i < Rows; ++i) {
        y(i) = z(i) / mat(static_cast<int>(i), static_cast<int>(i));
    }
    for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
            sum += mat(static_cast<int>(k), i) * x(k);
        }
        x(static_cast<size_t>(i)) = y(static_cast<size_t>(i)) - sum;
    }
    return x;
}
}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_