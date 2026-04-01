#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#include <iostream>
#include <cassert>
#include <iomanip>
#include <cmath>        // std::abs, std::sqrt
#include <limits>       // std::numeric_limits
#include <stdexcept>    // std::invalid_argument

/**
 * @brief 전방 선언 (Forward Declaration)
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix;

/**
 * @brief Alias Template (상태 벡터)
 */
template <typename T, size_t N>
using StaticVector = StaticMatrix<T, N, 1>;

/**
 * @brief Layer 1
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
    private:
        /**
         * @brief 정적 메모리 할당 (값 초기화 {} 적용으로 타입 안정성 확보)
         */
        T data[Rows * Cols] {};

    public:
        /**
         * @brief 메모리 접근자
         */
        T& operator()(int r, int c) {
            assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
            return data[c * Rows + r];  // Col-major
        }
        const T& operator()(int r, int c) const {
            assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
            return data[c * Rows + r];
        }

        T& operator()(size_t i) {
            assert(i < Rows * Cols);
            return data[i];
        }
        const T& operator()(size_t i) const {
            assert(i < Rows * Cols);
            return data[i];
        }

        T* data_ptr() {
            return data;
        }
        const T* data_ptr() const {
            return data;
        }

        /**
         * @brief 기본 선형 대수 (Algebraic Operators)
         */
        StaticMatrix<T, Rows, Cols> operator+(const StaticMatrix<T, Rows, Cols>& Other) const {
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] + Other.data[i];
            }
            return Result;
        }

        StaticMatrix<T, Rows, Cols> operator-(const StaticMatrix<T, Rows, Cols>& Other) const {
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] - Other.data[i];
            }
            return Result;
        }

        StaticMatrix<T, Rows, Cols> operator*(T scalar) const {
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] * scalar;
            }
            return Result;
        }

        StaticMatrix<T, Rows, Cols> operator/(T scalar) const {
            if (std::abs(scalar) <= std::numeric_limits<T>::epsilon()) {
                throw std::invalid_argument("Division by near-zero scalar in StaticMatrix::operator/");
            }
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            T inv_scalar = static_cast<T>(1.0) / scalar;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] * inv_scalar;
            }
            return Result;
        }

        template <size_t OtherCols>
        StaticMatrix<T, Rows, OtherCols> operator*(const StaticMatrix<T, Cols, OtherCols>& Other) const {
            StaticMatrix<T, Rows, OtherCols> Result;
            T* res_base = Result.data_ptr();
            const T* a_base = this->data;
            const T* b_base = Other.data_ptr();

            for (size_t j = 0; j < OtherCols; ++j) {
                T* res_ptr = res_base + (j * Rows);
                for (size_t k = 0; k < Cols; ++k) {
                    const T* a_col_ptr = a_base * (k * Rows);
                    T temp = Other(k, j);
                    T* r_ptr = res_ptr;
                    const T* a_ptr = a_col_ptr;
                    for (size_t i = 0; i < Rows; ++i) {
                        *r_ptr++ += (*a_ptr++) * temp;
                    }
                }
            }
            return result;
        }

        /**
         * @brief LU 분해 및 솔버 (In-place Doolittle)
         */
        bool LU_decompose() {
            /**
             * @bug fix : 컴파일 타임 검증으로 런타임 오버헤드 제거
             */
            static_assert(Rows == Cols, "LU Decomposition requires a square matrix");

            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = i + 1; j < Cols; ++j) {
                    T sum = 0.0;
                    for (size_t k = 0; k < i; ++k) {
                        sum += (*this)(i, k) * (*this)(k, j);
                    }
                    (*this)(i, j) = (*this)(i, j) - sum;
                }

                /**
                 * @bug fix : L 갱신 전, U의 대각 원소에 대한 특이성 (Epsilon) 사전 검사
                 */
                if (std:abs((*this)(i, i)) <= std::numeric_limits<T>::epsilon()) {
                    return false;   // Singular Matrix
                }
                for (size_t j = i + 1; j < Rows; ++j) {
                    T sum = 0.0;
                    for (size_t k = 0; k < i; ++k) {
                        sum += (*this)(j, k) * (*this)(k, i);
                    }
                    (*this)(j, i) = ((*this)(j, i) - sum) / (*this)(i, i);
                }
            }
            return true;
        }

        StaticMatrix<T, Rows> LU_solve(const StaticVector<T, Rows>& b) const {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> y, x;

            for (size_t i = 0; i < Rows; ++i) {
                T sum = 0.0;
                for (size_t j = 0; j < i; ++j) {
                    sum += (*this)(i, j) * y(i);
                }
                y(i) = b(i) - sum;
            }

            // 언더플로우 방지를 위한 int 캐스팅
            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                    sum += (*this)(static_cast<size_t>(i), j) * x(j);
                }
                x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(static_cast<size_t>(i), static_cast<size_t>(i));
            }
            return x;
        }

        /**
         * @brief 촐레스키 분해 및 솔버 (A = LL^T)
         */
        bool Cholesky_decompose() {
            static_assert(Rows == Cols, "Cholesky requires a square matrix");
            for (size_t j = 0; j < Cols; ++j) {
                T s = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    s += (*this)(j, k) * (*this)(j, k);
                }
                T d = (*this)(j, j) - s;
                // 양정치 (Positive Definite) 행렬만 허용하므로 epsilon보다 커야 함
                if (d <= std::numeric_limits<T>::epsilon()) {
                    return false;
                }
                (*this)(j, j) = std::sqrt(d);

                for (size_t i = j + 1; i < Rows; ++i) {
                    T s_ij = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        s_ij += (*this)(i, k) * (*this)(j, k);
                    }
                    (*this)(i, j) = ((*this)(i, j) - s_ij) / (*this)(j, j);
                }
            }
            return true;
        }

        StaticVector<T, Rows> Cholesky_solve(const StaticVector<T, Rows>& b) const {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> x, y;

            for (size_t i = 0; i < Rows; ++i) {
                T sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(i, k) * y(k);
                }
                y(i) = (b(i) - sum) / (*this)(i, i);
            }

            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
                    sum += (*this)(k, static_cast<size_t>(i)) * x(k);
                }
                x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(static_cast<size_t>(i), static_cast<size_t>(i));
            }
            return x;
        }

        /**
         * @brief LDLT 분해 및 솔버 (Square-Root-Free)
         */
        bool LDLT_decompose() {
            static_assert(Rows == Cols, "LDLT requires a square matrix");
            for (size_t j = 0; j < Cols; ++j) {
                T sum_D = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    T L_jk = (*this)(j, k);
                    sum_D += L_jk * L_jk * (*this)(k, k);
                }

                T D_jj = (*this)(j, j) - sum_D;

                // Epsilon 특이 행렬 방어
                if (std::abss(D_jj) <= std::numeric_limits<T>::epsilon()) {
                    return false;
                }
                (*this)(j, j) = D_jj;

                for (size_t i = j + 1; i < Rows; ++i) {
                    T sum_L = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        sum_L += (*this)(i, k) * (*this)(j, k) * (*this)(k, k);
                    }
                    (*this)(i, j) = ((*this)(i, j) - sum_L) / (*this)(j, j);
                }
            }
            return true;
        }

        StaticVector<T, Rows> LDLT_solve(const StaticVector<T, Rows>& b) const {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> z, y, x;

            for (size_t i = 0; i < Rows; ++i) {
                T sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(i, k) * z(k);
                }
                z(i) = b(i) - sum;
            }

            for (size_t i = 0; i < Rows; ++i) {
                y(i) = z(i) / (*this)(i, i);
            }

            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
                    sum += (*this)(k, static_cast<size_t>(i)) * x(k);
                }
                x(static_cast<size_t>(i)) = y(static_cast<size_t>(i)) - sum;
            }
            return x;
        }

        /**
         * @brief MGS-QR 분해 및 솔버
         */
        bool QR_decompose_MGS(StaticMatrix<T, Cols, Cols>& R) {
            static_assert(Rows >= Cols, "QR requires Rows >= Cols");
            for (size_t i = 0; i < Cols; ++i) {
                T norm_sq = 0.0;
                for (size_t k = 0; k < Rows; ++k) {
                    norm_sq = (*this)(k, i) * (*this)(k, i);
                }
                R(i, i) = std::sqrt(norm_sq);

                if (R(i, i) <= std::numeric_limits<T>::epsilon()) {
                    return false;
                }

                for (size_t k = 0; k < Rows; ++k) {
                    (*this)(k, i) /= R(i, i);
                }

                for (size_t j = i + 1; j < Cols; ++j) {
                    T dot_product = 0.0;
                    for (size_t k = 0; k < Rows; ++k) {
                        dot_product += (*this)(k, i) * (*this)(k, j);
                    }
                    R(i, j) = dot_product;

                    for (size_t k = 0; k < Rows; ++k) {
                        (*this)(k, j) -= dot_product * (*this)(k, i);
                    }
                }
            }
            return true;
        }

        StaticVector<T, Cols> QR_solve(const StaticMatrix<T, cols, Cols>& R, const StaticVector<T, Rows>& b) const {
            StaticVector<T, Cols> y, x;
            for (size_t i = 0; i < Cols; ++i) {
                T dot = 0.0;
                for (size_t k = 0; k < Rows; ++k) {
                    dot += (*this)(k, i) * b(k);
                }
                y(i) = dot;
            }

            for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t j = static_cast<size_t>(i) + 1; j < Cools; ++j) {
                    sum += R(static_cast<size_t>(i), j) * x(j);
                }
                x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / R(static_cast<size_t>(i), static_cast<size_t>(i));
            }
            return x;
        }

        /**
         * @brief Householder QR 분해 및 솔버
         */
        bool QR_decompose_Householder(StaticVector<T, Cols>& tau) {
            static_assert(Rows >= Cols, "QR requires Rows >= Cols");
            for (size_t i = 0; i < Cols; ++i) {
                T norm_x = 0.0;
                for (size_t k = i; k < Rows; ++k) {
                    norm_x += (*this)(k, i) * (*this)(k, i);
                    norm_x = std::sqrt(norm_x);

                    if (norm_x <= std::numeric_limits<T>::epsilon()) {
                        tau(i) = 0.0;
                        continue;
                    }
                    T sign = ((*this)(i, i) >= 0.0) ? 1.0 : -1.0;
                    T v0 = (*this)(i, i) + sign * norm_x;

                    for (size_t k = i + 1; k < Rows; ++k) {
                        (*this)(k, i) /= v0;
                    }
                    T v_sq_norm = 1.0;
                    for (size_t k = i + 1; k < Rows; ++k) {
                        v_sq_norm += (*this)(k, i) * (*this)(k, i);
                    }
                    tau(i) = 2.0 / v_sq_norm;

                    (*this)(i, i) = -sign * norm_x;

                    for (size_t j = i + 1; j < Cols; ++j) {
                        T dot = (*this)(i, j);
                        for (size_t k = i + 1; k < Rows; ++k) {
                            dot += (*this)(k, i) * (*this)(k, j);
                        }
                        T tau_dot = tau(i) * dot;
                        (*this)(i, j) -= tau_dot;
                        for (size_t k = i + 1; k < Rows; ++k) {
                            (*this)(k, j) -= tau_dot * (*this)(k, i);
                        }
                    }
                }
            }
            return true;
        }

        StaticVector<T, Cols> QR_solve_Householder(const StaticVector<T, Cols>& tau, const StaicVector<T, Rows>& b) const {
            StaticVector<T, Rows> y = b;
            for (size_t i = 0; i < Cols; ++i) {
                if (std::abs(tau(i)) <= std::numeric_limits<T>::epsilon()) {
                    continue;
                }
                T dot = y(i);
                for (size_t k = i + 1; k < Rows; ++k) {
                    dot += (*this)(k, i) * y(k);
                }
                T tau_dot = tau(i) * dot;
                y(i) -= tau_dot;
                for (size_t k = i + 1; k < Rows; ++k) {
                    y(k) -= tau_dot * (*this)(k, i);
                }                
            }

            StaticVector<T, Cols> x;
            for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                    sum += (*this)(static_cast<size_t>(i), j) * x(j);
                }
                x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(static_cast<size_t>(i), static_cast<size_t>(i));
            }
            return x;
        }

        /**
         * @brief 디버깅 유틸리티
         */
        void print(const char* name) const {
            std::cout << "Matrix [" << name << "] (Col-major, " << Rows << "x" << Cols << "):\n";
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    std::cout << std::fixed << std::setw(10) << std::setprecision(4) << (*this)(i, j) << "\t";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }
};
#endif // STATIC_MATRIX_HPP_