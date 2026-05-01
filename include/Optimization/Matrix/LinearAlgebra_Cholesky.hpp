#ifndef OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"

namespace Optimization {
    namespace linalg {
        template <typename T, size_t Rows, size_t Cols>
        MathStatus Cholesky_decompose(StaticMatrix<T, Rows, Cols>& mat) {
            static_assert(Rows == Cols, "Cholesky requires a square matrix");
            int N = static_cast<int>(Rows);

            for (int j = 0; j < N; ++j) {
                T s = static_cast<T>(0);
                for (int k = 0; k < j; ++k) {
                    s += mat(j, k) * mat(j, k);
                }
                T d = mat(j, j) - s;

                if (d <= std::numeric_limits<T>::epsilon()) {
                    return MathStatus::SINGULAR;
                }

                mat(j, j) = MathTraits<T>::sqrt(d);

                for (int i = j + 1; i < N; ++i) {
                    T s_ij = static_cast<T>(0);
                    for (int k = 0; k < j; ++k) {
                        s_ij += mat(i, k) * mat(j, k);
                    }
                    mat(i, j) = (mat(i, j) - s_ij) / mat(j, j);
                }
            }
            return MathStatus::SUCCESS;
        }

        template <typename T, size_t Rows, size_t Cols>
        StaticVector<T, Rows> Cholesky_solve(const StaticMatrix<T, Rows, Cols>& mat, const StaticVector<T, Rows>& b) {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> x, y;
            int N = static_cast<int>(Rows);

            // 전진 대입
            for (int i = 0; i < N; ++i) {
                T sum = static_cast<T>(0);
                for (int k = 0; k < i; ++k) {
                    sum += mat(i, k) * y(k);
                }
                y(i) = (b(i) - sum) / mat(i, i);
            }
            // 후진 대입
            for (int i = N - 1; i >= 0; --i) {
                T sum = static_cast<T>(0);
                for (int k = i + 1; k < N; ++k) {
                    sum += mat(k, i) * x(k);
                }
                x(i) = (y(i) - sum) / mat(i, i);
            }
            return x;
        }
    } // namespace linalg
} // namespace Optimization

#endif // OPTIMIZATION_LINEAR_ALGEBRA_CHOLESKY_HPP_