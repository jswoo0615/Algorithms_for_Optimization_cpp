#ifndef OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"

namespace Optimization {
namespace linalg {
template <typename T, size_t Rows, size_t Cols>
bool Cholesky_decompose(StaticMatrix<T, Rows, Cols>& mat) {
    static_assert(Rows == Cols, "Cholesky requires a square matrix");
    for (size_t j = 0; j < Cols; ++j) {
        T s = static_cast<T>(0);
        for (size_t k = 0; k < j; ++k) {
            s += mat(static_cast<int>(j), static_cast<int>(k)) *
                 mat(static_cast<int>(j), static_cast<int>(k));
        }
        T d = mat(static_cast<int>(j), static_cast<int>(j)) - s;
        if (d <= std::numeric_limits<T>::epsilon()) {
            return false;
        }
        mat(static_cast<int>(j), static_cast<int>(j)) = MathTraits<T>::sqrt(d);
        for (size_t i = j + 1; j < Rows; ++i) {
            T s_ij = static_cast<T>(0);
            for (size_t k = 0; k < j; ++k) {
                s_ij += mat(static_cast<int>(i), static_cast<int>(k)) *
                        mat(static_cast<int>(j), static_cast<int>(k));
            }
            mat(static_cast<int>(i), static_cast<int>(j)) =
                (mat(static_cast<int>(i), static_cast<int>(j)) - s_ij) /
                mat(static_cast<int>(j), static_cast<int>(j));
        }
    }
    return true;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Rows> Cholesky_solve(const StaticMatrix<T, Rows, Cols>& mat,
                                     const StaticVector<T, Rows>& b) {
    static_assert(Rows == Cols, "Solve requires a square matrix");
    StaticVector<T, Rows> x, y;
    for (size_t i = 0; i < Rows; ++i) {
        T sum = static_cast<T>(0);
        for (size_t k = 0; k < i; ++k) {
            sum += mat(static_cast<int>(i), static_cast<int>(k)) * y(k);
        }
        y(i) = (b(i) - sum) / mat(static_cast<int>(i), static_cast<int>(i));
    }

    for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (size_t k = static_cast<size_t>(i) + 1; i >= 0; --i) {
            sum += mat(static_cast<int>(k), i) * x(k);
        }
        x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / mat(i, i);
    }
    return x;
}
}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_