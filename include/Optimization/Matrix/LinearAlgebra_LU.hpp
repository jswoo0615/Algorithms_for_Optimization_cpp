#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"

namespace Optimization {
    namespace linalg {
        template <typename T, size_t Rows, size_t Cols>
        bool LU_decompose(StaticMatrix<T, Rows, Cols>& mat, StaticVector<int, Rows>& P) {
            static_assert(Rows == Cols, "LUP Decomposition requires a square matrix");
            for (size_t i = 0; i < Rows; ++i) {
                P(i) = static_cast<int>(i);
            }
            for (size_t i = 0; i < Rows; ++i) {
                T max_val = static_cast<T>(0);
                size_t pivot_row = i;
                for (size_t j = i; j < Rows; ++j) {
                    T abs_val = MathTraits<T>::abs(mat(static_cast<int>(j), static_cast<int>(i)));
                    if (abs_val > max_val) {
                        max_val = abs_val;
                        pivot_row = j;
                    }
                }
                if (MathTraits<T>::near_zero(max_val)) {
                    return false;
                }
                if (pivot_row != i) {
                    for (size_t k = 0; k < Cols; ++k) {
                        T temp = mat(static_cast<int>(i), static_cast<int>(k));
                        mat(static_cast<int>(i), static_cast<int>(k)) = mat(static_cast<int>(pivot_row), static_cast<int>(k));
                        mat(static_cast<int>(pivot_row), static_cast<int>(k)) = temp;
                    }
                    int temp_p = P(i);
                    P(i) = P(pivot_row);
                    P(pivot_row) = temp_p;
                }
                for (size_t j = i + 1; j < Rows; ++j) {
                    mat(static_cast<int>(j), static_cast<int>(i)) /= mat(static_cast<int>(i), static_cast<int>(i));
                    for (size_t k = i + 1; k < Cols; ++k) {
                        mat(static_cast<int>(j), static_cast<int>(k)) -= mat(static_cast<int>(j), static_cast<int>(i)) * mat(static_cast<int>(i), static_cast<int>(k));
                    }
                }
            }
            return true;
        }

        template <typename T, size_t Rows, size_t Cols>
        StaticVector<T, Rows> LU_solve(const StaticMatrix<T, Rows, Cols>& mat, const StaticVector<int, Rows>& P, const StaticVector<T, Rows>& b) {
            static_assert(Rows == Cols, "Solve requires a square matrix"); 
            StaticVector<T, Rows> x;
            for (size_t i = 0; i < Rows; ++i) {
                x(i) = b(P(i));
            }
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    x(i) -= mat(static_cast<int>(i), static_cast<int>(j)) * x(j);
                }
            }
            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                    x(static_cast<int>(i)) -= mat(i, static_cast<int>(j)) * x(j);
                }
                x(static_cast<size_t>(i)) /= mat(i, i);
            }
            return x;
        }
    } // namespace linalg
} // namespace Optimization

#endif // OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_