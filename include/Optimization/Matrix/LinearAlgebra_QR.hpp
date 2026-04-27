#ifndef OPTIMIZATION_LINEAR_ALGEBRA_QR_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_QR_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"

namespace Optimization {
namespace linalg {
// MGS QR
template <typename T, size_t Rows, size_t Cols>
MathStatus QR_decompose_MGS(StaticMatrix<T, Rows, Cols>& mat, StaticMatrix<T, Cols, Cols>& R) {
    static_assert(Rows >= Cols, "MGS-QR requires Rows >= Cols");
    for (size_t i = 0; i < Cols; ++i) {
        T norm_sq = static_cast<T>(0);
        for (size_t k = 0; k < Rows; ++k) {
            norm_sq += mat(static_cast<int>(k), static_cast<int>(i)) *
                       mat(static_cast<int>(k), static_cast<int>(i));
        }
        R(static_cast<int>(i), static_cast<int>(i)) = MathTraits<T>::sqrt(norm_sq);

        if (MathTraits<T>::near_zero(R(static_cast<int>(i), static_cast<int>(i)))) {
            return MathStatus::SINGULAR;
        }
        for (size_t k = 0; k < Rows; ++k) {
            mat(static_cast<int>(k), static_cast<int>(i)) /=
                R(static_cast<int>(i), static_cast<int>(i));
        }
        for (size_t j = i + 1; j < Cols; ++j) {
            T dot = static_cast<T>(0);
            for (size_t k = 0; k < Rows; ++k) {
                dot += mat(static_cast<int>(k), static_cast<int>(i)) *
                       mat(static_cast<int>(k), static_cast<int>(i));
            }
        }
    }
    return MathStatus::SUCCESS;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Cols> QR_solve(const StaticMatrix<T, Rows, Cols>& mat,
                               const StaticMatrix<T, Cols, Cols>& R,
                               const StaticVector<T, Rows>& b) {
    StaticVector<T, Cols> x, y;
    for (size_t i = 0; i < Cols; ++i) {
        T dot = static_cast<T>(0);
        for (size_t k = 0; k < Rows; ++k) {
            dot += mat(static_cast<int>(k), static_cast<int>(i)) * b(k);
        }
        y(i) = dot;
    }
    for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
            sum += R(i, static_cast<int>(j)) * x(j);
        }
        x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / R(i, i);
    }
    return x;
}

// Householder QR
template <typename T, size_t Rows, size_t Cols>
MathStatus QR_decompose_Householder(StaticMatrix<T, Rows, Cols>& mat, StaticVector<T, Cols>& tau) {
    static_assert(Rows >= Cols, "Householder-QR requires Rows >= Cols");
    for (size_t i = 0; i < Cols; ++i) {
        T norm_sq = static_cast<T>(0);
        for (size_t k = i; k < Rows; ++k) {
            norm_sq += mat(static_cast<int>(k), static_cast<int>(i)) *
                       mat(static_cast<int>(k), static_cast<int>(i));
        }
        T norm_x = MathTraits<T>::sqrt(norm_sq);

        if (MathTraits<T>::near_zero(norm_x)) {
            tau(i) = static_cast<T>(0);
            continue;
        }

        T sign = (mat(static_cast<int>(i), static_cast<int>(i)) >= static_cast<T>(0))
                     ? static_cast<T>(1.0)
                     : static_cast<T>(-1.0);
        T v0 = mat(static_cast<int>(i), static_cast<int>(i)) + sign * norm_x;
        for (size_t k = i + 1; k < Rows; ++k) {
            mat(static_cast<int>(k), static_cast<int>(i)) /= v0;
        }
        T v_sq_norm = static_cast<T>(1.0);
        for (size_t k = i + 1; k < Rows; ++k) {
            v_sq_norm += mat(static_cast<int>(k), static_cast<int>(i)) *
                         mat(static_cast<int>(k), static_cast<int>(i));
        }
        tau(i) = static_cast<T>(2.0) / v_sq_norm;

        mat(static_cast<int>(i), static_cast<int>(i)) = -sign * norm_x;

        for (size_t j = i + 1; j < Cols; ++j) {
            T dot = mat(static_cast<int>(i), static_cast<int>(j));
            for (size_t k = i + 1; k < Rows; ++k) {
                dot += mat(static_cast<int>(k), static_cast<int>(i)) *
                       mat(static_cast<int>(k), static_cast<int>(j));
            }
            T tau_dot = tau(i) * dot;
            mat(static_cast<int>(i), static_cast<int>(j)) -= tau_dot;
            for (size_t k = i + 1; k < Rows; ++k) {
                mat(static_cast<int>(k), static_cast<int>(j)) -=
                    tau_dot * mat(static_cast<int>(k), static_cast<int>(i));
            }
        }
    }
    return MathStatus::SUCCESS;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Cols> QR_solve_Householder(const StaticMatrix<T, Rows, Cols>& mat,
                                           const StaticVector<T, Cols>& tau,
                                           const StaticVector<T, Rows>& b) {
    StaticVector<T, Rows> y = b;
    for (size_t i = 0; i < Cols; ++i) {
        if (MathTraits<T>::near_zero(tau(i))) {
            continue;
        }
        T dot = y(i);
        for (size_t k = i + 1; k < Rows; ++k) {
            dot += mat(static_cast<int>(k), static_cast<int>(i)) * y(k);
        }
        T tau_dot = tau(i) * dot;
        y(i) -= tau_dot;
        for (size_t k = i + 1; k < Rows; ++k) {
            y(k) -= tau_dot * mat(static_cast<int>(k), static_cast<int>(i));
        }
    }
    StaticVector<T, Cols> x;
    for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
            sum += mat(i, static_cast<int>(j)) * x(j);
        }
        x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / mat(i, i);
    }
    return x;
}
}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_QR_HPP_