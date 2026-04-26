#ifndef OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Matrix/MathTraits.hpp"

namespace Optimization {
    // ===============================================
    // 기본 선형 대수 연산 (Algebraic Operators)
    // ===============================================
    template <typename T, size_t Rows, size_t Cols>
    StaticMatrix<T, Rows, Cols> operator+(const StaticMatrix<T, Rows, Cols>& A, 
                                          const StaticMatrix<T, Rows, Cols>& B) {
        StaticMatrix<T, Rows, Cols> Result;
        const size_t total = Rows * Cols;
        for (size_t i = 0; i < total; ++i) {
            Result(i) = A(i) + B(i);
        }
        return Result;
    }

    template <typename T, size_t Rows, size_t Cols>
    StaticMatrix<T, Rows, Cols> operator-(const StaticMatrix<T, Rows, Cols>& A,
                                          const StaticMatrix<T, Rows, Cols>& B) {
        StaticMatrix<T, Rows, Cols> Result;
        const size_t total = Rows * Cols;
        for (size_t i = 0; i < total; ++i) {
            Result(i) = A(i) - B(i);
        }
        return Result;
    }

    template <typename T, size_t Rows, size_t Cols>
    StaticMatrix<T, Rows, Cols> operator*(const StaticMatrix<T, Rows, Cols>& A, T scalar) {
        StaticMatrix<T, Rows, Cols> Result;
        const size_t total = Rows * Cols;
        for (size_t i = 0; i < total; ++i) {
            Result(i) = A(i) * scalar;
        }
        return Result;
    }

    template <typename T, size_t Rows, size_t Cols>
    StaticMatrix<T, Rows, Cols> operator/(const StaticMatrix<T, Rows, Cols>& A, T scalar) {
        if (MathTraits<T>::near_zero(scalar)) {
            throw std::invalid_argument("Division by near-zero scalar in StaticMatrix::operator/");
        }
        StaticMatrix<T, Rows, Cols> Result;
        const size_t total = Rows * Cols;
        T inv_scalar = 1 / scalar;
        for (size_t i = 0; i < total; ++i) {
            Result(i) = A(i) * inv_scalar;
        }
        return Result;
    }

    template <typename T, size_t Rows, size_t Cols, size_t OtherCols> 
    StaticMatrix<T, Rows, OtherCols> operator*(const StaticMatrix<T, Rows, Cols>& A, 
                                               const StaticMatrix<T, Cols, OtherCols>& B) {
        StaticMatrix<T, Rows, OtherCols> Result;
        T* res_base = Result.data_ptr();
        const T* a_base = A.data_ptr();

        // J-K-I 루프 캐시 최적화 적용
        for (size_t j = 0; j < OtherCols; ++j) {
            T* res_col = res_base + (j * Rows);
            for (size_t k = 0; k < Cols; ++k) {
                const T* a_col = a_base + (k * Rows);
                const T b_kj = B(static_cast<int>(k), static_cast<int>(j));
                for (size_t i = 0; i < Rows; ++i) {
                    res_col[i] += a_col[i] * b_kj;
                }
            }
        }
        return Result;
    }
    namespace linalg {
        // ================================================
        // 행렬 전치 (Matrix Transpose)
        // ================================================
        template <typename T, size_t Rows, size_t Cols>
        StaticMatrix<T, Cols, Rows> transpose(const StaticMatrix<T, Rows, Cols>& mat) {
            StaticMatrix<T, Cols, Rows> Result;
            for (size_t j = 0; j < Cols; ++j) {
                const T* src_col = mat.data_ptr() + (j * Rows);
                for (size_t i = 0; i < Rows; ++i) {
                    Result(static_cast<int>(j), static_cast<int>(i)) = src_col[i];
                }
            }
            return Result;
        }
    } // namespace linalg
} // namespace Optimization

#endif // OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_