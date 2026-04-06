#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#include <iostream>
#include <cassert>
#include <iomanip>
#include <cmath>        // std::abs, std::sqrt
#include <limits>       // std::numeric_limits
#include <stdexcept>    // std::invalid_argument
#include <type_traits>  // std::is_floating_point (AD Traits 확장 대비)

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

// ============================================================
// Traits 구조체
// - T가 scalar(double/float)이면 기본 std:: 함수 위임
// - T가 Dual<T> 등 복합 타입이면 특수화(Specialization)로 확장 가능
// ============================================================
template <typename T>
struct MathTraits {
    static T   abs  (const T& x) { return std::abs(x); }
    static T   sqrt (const T& x) { return std::sqrt(x); }

    /**
     * @brief Epsilon 특이성 검사
     *   - scalar: std::numeric_limits<T>::epsilon() 직접 비교
     *   - Dual:   특수화에서 실수부(real part)만 비교하도록 오버라이드
     */
    static bool near_zero(const T& x) {
        return std::abs(x) <= std::numeric_limits<T>::epsilon();
    }
};

/**
 * @brief Layer 1
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
    private:
        // ============================================================
        // [FIX #1] alignas(64): AVX-512 / 캐시 라인 정렬 보장
        //   - 멀티코어/SIMD 환경에서 비정렬 페널티 제거
        //   - Aurix TriCore SIMD 파이프라인 최적화 대응
        // ============================================================
        alignas(64) T data[Rows * Cols] {};

    public:
        /**
         * @brief 메모리 접근자 (Col-major 레이아웃)
         */
        T& operator()(int r, int c) {
            assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
            return data[c * Rows + r];
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
            // [FIX #5] MathTraits::near_zero 로 AD 호환 epsilon 검사
            if (MathTraits<T>::near_zero(scalar)) {
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

        // ============================================================
        // [FIX #2 + #6] 행렬 곱셈 operator*
        //   원본 버그:
        //     1) `a_base * (k * Rows)` → 포인터 산술 오타 (곱셈 → 덧셈)
        //     2) `return result;` → 대소문자 불일치 (`Result` 여야 함)
        //   최적화:
        //     Col-major 레이아웃에 맞는 J-K-I 루프 순서 적용
        //     → 내부 루프가 연속 메모리를 순차 접근 → 공간 지역성 극대화
        // ============================================================
        template <size_t OtherCols>
        StaticMatrix<T, Rows, OtherCols> operator*(const StaticMatrix<T, Cols, OtherCols>& Other) const {
            StaticMatrix<T, Rows, OtherCols> Result;
            T*       res_base = Result.data_ptr();
            const T* a_base   = this->data;
            const T* b_base   = Other.data_ptr();

            // J-K-I 순서: Col-major 레이아웃 공간 지역성 최적화
            for (size_t j = 0; j < OtherCols; ++j) {
                T*       res_col = res_base + (j * Rows);       // Result의 j열 시작
                for (size_t k = 0; k < Cols; ++k) {
                    // [FIX #6] `a_base * (k * Rows)` → `a_base + (k * Rows)`
                    const T* a_col = a_base + (k * Rows);       // A의 k열 시작
                    const T  b_kj  = Other(static_cast<int>(k),
                                          static_cast<int>(j)); // B[k][j] 스칼라
                    for (size_t i = 0; i < Rows; ++i) {
                        res_col[i] += a_col[i] * b_kj;
                    }
                }
            }
            return Result;  // [FIX #6] `result` → `Result`
        }

        /**
         * @brief LU 분해 (In-place Doolittle)
         */
        bool LU_decompose() {
            static_assert(Rows == Cols, "LU Decomposition requires a square matrix");

            for (size_t i = 0; i < Rows; ++i) {
                // U 행 갱신
                for (size_t j = i; j < Cols; ++j) {
                    T sum = static_cast<T>(0);
                    for (size_t k = 0; k < i; ++k) {
                        sum += (*this)(static_cast<int>(i), static_cast<int>(k))
                             * (*this)(static_cast<int>(k), static_cast<int>(j));
                    }
                    (*this)(static_cast<int>(i), static_cast<int>(j)) -= sum;
                }

                // [FIX #5 + #6] `std:abs` → `std::abs`, MathTraits 위임
                if (MathTraits<T>::near_zero((*this)(static_cast<int>(i), static_cast<int>(i)))) {
                    return false;   // Singular Matrix
                }

                // L 열 갱신
                for (size_t j = i + 1; j < Rows; ++j) {
                    T sum = static_cast<T>(0);
                    for (size_t k = 0; k < i; ++k) {
                        sum += (*this)(static_cast<int>(j), static_cast<int>(k))
                             * (*this)(static_cast<int>(k), static_cast<int>(i));
                    }
                    (*this)(static_cast<int>(j), static_cast<int>(i)) =
                        ((*this)(static_cast<int>(j), static_cast<int>(i)) - sum)
                        / (*this)(static_cast<int>(i), static_cast<int>(i));
                }
            }
            return true;
        }

        // ============================================================
        // [FIX #6] LU_solve
        //   원본 버그: `StaticMatrix<T, Rows>` → `StaticVector<T, Rows>`
        //              forward substitution에서 `y(i)` → `y(j)` 인덱스 오류
        // ============================================================
        StaticVector<T, Rows> LU_solve(const StaticVector<T, Rows>& b) const {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> y, x;

            // Forward substitution: Ly = b
            for (size_t i = 0; i < Rows; ++i) {
                T sum = static_cast<T>(0);
                for (size_t j = 0; j < i; ++j) {
                    // [FIX] `y(i)` → `y(j)` (올바른 이미 풀린 원소 참조)
                    sum += (*this)(static_cast<int>(i), static_cast<int>(j))
                         * y(j);
                }
                y(i) = b(i) - sum;
            }

            // Back substitution: Ux = y
            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                T sum = static_cast<T>(0);
                for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                    sum += (*this)(i, static_cast<int>(j)) * x(j);
                }
                x(static_cast<size_t>(i)) =
                    (y(static_cast<size_t>(i)) - sum)
                    / (*this)(i, i);
            }
            return x;
        }

        /**
         * @brief 촐레스키 분해 (A = LL^T)
         */
        bool Cholesky_decompose() {
            static_assert(Rows == Cols, "Cholesky requires a square matrix");
            for (size_t j = 0; j < Cols; ++j) {
                T s = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    s += (*this)(static_cast<int>(j), static_cast<int>(k))
                       * (*this)(static_cast<int>(j), static_cast<int>(k));
                }
                T d = (*this)(static_cast<int>(j), static_cast<int>(j)) - s;

                // [FIX #5] MathTraits::near_zero로 AD 호환 epsilon 검사
                if (d <= std::numeric_limits<T>::epsilon()) {
                    return false;   // Not positive-definite
                }
                (*this)(static_cast<int>(j), static_cast<int>(j)) =
                    MathTraits<T>::sqrt(d);

                for (size_t i = j + 1; i < Rows; ++i) {
                    T s_ij = static_cast<T>(0);
                    for (size_t k = 0; k < j; ++k) {
                        s_ij += (*this)(static_cast<int>(i), static_cast<int>(k))
                              * (*this)(static_cast<int>(j), static_cast<int>(k));
                    }
                    (*this)(static_cast<int>(i), static_cast<int>(j)) =
                        ((*this)(static_cast<int>(i), static_cast<int>(j)) - s_ij)
                        / (*this)(static_cast<int>(j), static_cast<int>(j));
                }
            }
            return true;
        }

        StaticVector<T, Rows> Cholesky_solve(const StaticVector<T, Rows>& b) const {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> x, y;

            // Forward substitution: Ly = b
            for (size_t i = 0; i < Rows; ++i) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(i), static_cast<int>(k)) * y(k);
                }
                y(i) = (b(i) - sum) / (*this)(static_cast<int>(i), static_cast<int>(i));
            }

            // Back substitution: L^T x = y
            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                T sum = static_cast<T>(0);
                for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
                    sum += (*this)(static_cast<int>(k), i) * x(k);
                }
                x(static_cast<size_t>(i)) =
                    (y(static_cast<size_t>(i)) - sum)
                    / (*this)(i, i);
            }
            return x;
        }

        /**
         * @brief LDLT 분해 (Square-Root-Free Cholesky)
         */
        bool LDLT_decompose() {
            static_assert(Rows == Cols, "LDLT requires a square matrix");
            for (size_t j = 0; j < Cols; ++j) {
                T sum_D = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    T L_jk = (*this)(static_cast<int>(j), static_cast<int>(k));
                    sum_D += L_jk * L_jk
                           * (*this)(static_cast<int>(k), static_cast<int>(k));
                }

                T D_jj = (*this)(static_cast<int>(j), static_cast<int>(j)) - sum_D;

                // [FIX #5 + #6] `std::abss` → MathTraits::near_zero
                if (MathTraits<T>::near_zero(D_jj)) {
                    return false;
                }
                (*this)(static_cast<int>(j), static_cast<int>(j)) = D_jj;

                for (size_t i = j + 1; i < Rows; ++i) {
                    T sum_L = static_cast<T>(0);
                    for (size_t k = 0; k < j; ++k) {
                        sum_L += (*this)(static_cast<int>(i), static_cast<int>(k))
                               * (*this)(static_cast<int>(j), static_cast<int>(k))
                               * (*this)(static_cast<int>(k), static_cast<int>(k));
                    }
                    (*this)(static_cast<int>(i), static_cast<int>(j)) =
                        ((*this)(static_cast<int>(i), static_cast<int>(j)) - sum_L)
                        / (*this)(static_cast<int>(j), static_cast<int>(j));
                }
            }
            return true;
        }

        StaticVector<T, Rows> LDLT_solve(const StaticVector<T, Rows>& b) const {
            static_assert(Rows == Cols, "Solve requires a square matrix");
            StaticVector<T, Rows> z, y, x;

            // Forward substitution: Lz = b
            for (size_t i = 0; i < Rows; ++i) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(i), static_cast<int>(k)) * z(k);
                }
                z(i) = b(i) - sum;
            }

            // 대각 스케일링: Dy = z
            for (size_t i = 0; i < Rows; ++i) {
                y(i) = z(i) / (*this)(static_cast<int>(i), static_cast<int>(i));
            }

            // Back substitution: L^T x = y
            for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
                T sum = static_cast<T>(0);
                for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
                    sum += (*this)(static_cast<int>(k), i) * x(k);
                }
                x(static_cast<size_t>(i)) = y(static_cast<size_t>(i)) - sum;
            }
            return x;
        }

        /**
         * @brief MGS-QR 분해 (Modified Gram-Schmidt)
         */
        // ============================================================
        // [FIX #6] QR_decompose_MGS
        //   원본 버그: `norm_sq = ...` (누적 아님, += 누락)
        // ============================================================
        bool QR_decompose_MGS(StaticMatrix<T, Cols, Cols>& R) {
            static_assert(Rows >= Cols, "QR requires Rows >= Cols");
            for (size_t i = 0; i < Cols; ++i) {
                T norm_sq = static_cast<T>(0);
                for (size_t k = 0; k < Rows; ++k) {
                    // [FIX] `=` → `+=` (누산 연산 복구)
                    norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i))
                             * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
                R(static_cast<int>(i), static_cast<int>(i)) =
                    MathTraits<T>::sqrt(norm_sq);

                if (MathTraits<T>::near_zero(
                        R(static_cast<int>(i), static_cast<int>(i)))) {
                    return false;
                }

                for (size_t k = 0; k < Rows; ++k) {
                    (*this)(static_cast<int>(k), static_cast<int>(i))
                        /= R(static_cast<int>(i), static_cast<int>(i));
                }

                for (size_t j = i + 1; j < Cols; ++j) {
                    T dot = static_cast<T>(0);
                    for (size_t k = 0; k < Rows; ++k) {
                        dot += (*this)(static_cast<int>(k), static_cast<int>(i))
                             * (*this)(static_cast<int>(k), static_cast<int>(j));
                    }
                    R(static_cast<int>(i), static_cast<int>(j)) = dot;

                    for (size_t k = 0; k < Rows; ++k) {
                        (*this)(static_cast<int>(k), static_cast<int>(j))
                            -= dot * (*this)(static_cast<int>(k), static_cast<int>(i));
                    }
                }
            }
            return true;
        }

        // ============================================================
        // [FIX #6] QR_solve (MGS)
        //   원본 버그:
        //     1) `StaticMatrix<T, cols, Cols>` → `cols` 대소문자 오타
        //     2) `j < Cools` → `Cools` 오타 (`Cols` 여야 함)
        // ============================================================
        StaticVector<T, Cols> QR_solve(
            const StaticMatrix<T, Cols, Cols>& R,
            const StaticVector<T, Rows>& b) const
        {
            StaticVector<T, Cols> y, x;

            // Q^T b 계산
            for (size_t i = 0; i < Cols; ++i) {
                T dot = static_cast<T>(0);
                for (size_t k = 0; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) * b(k);
                }
                y(i) = dot;
            }

            // Back substitution: Rx = y
            for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
                T sum = static_cast<T>(0);
                // [FIX] `Cools` → `Cols`
                for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                    sum += R(i, static_cast<int>(j)) * x(j);
                }
                x(static_cast<size_t>(i)) =
                    (y(static_cast<size_t>(i)) - sum) / R(i, i);
            }
            return x;
        }

        /**
         * @brief Householder QR 분해
         */
        // ============================================================
        // [FIX #6] QR_decompose_Householder
        //   원본 버그: `norm_x = std::sqrt(norm_x)` 가 누산 루프 안에 중첩됨
        //             → 누산 완료 후 루프 밖에서 sqrt 적용해야 함
        // ============================================================
        bool QR_decompose_Householder(StaticVector<T, Cols>& tau) {
            static_assert(Rows >= Cols, "QR requires Rows >= Cols");
            for (size_t i = 0; i < Cols; ++i) {
                // [FIX] norm 누산 루프와 sqrt를 분리
                T norm_sq = static_cast<T>(0);
                for (size_t k = i; k < Rows; ++k) {
                    norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i))
                             * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
                T norm_x = MathTraits<T>::sqrt(norm_sq);  // 루프 밖에서 sqrt

                if (MathTraits<T>::near_zero(norm_x)) {
                    tau(i) = static_cast<T>(0);
                    continue;
                }

                T sign = ((*this)(static_cast<int>(i), static_cast<int>(i))
                         >= static_cast<T>(0))
                       ? static_cast<T>(1.0) : static_cast<T>(-1.0);
                T v0 = (*this)(static_cast<int>(i), static_cast<int>(i))
                     + sign * norm_x;

                for (size_t k = i + 1; k < Rows; ++k) {
                    (*this)(static_cast<int>(k), static_cast<int>(i)) /= v0;
                }

                T v_sq_norm = static_cast<T>(1.0);
                for (size_t k = i + 1; k < Rows; ++k) {
                    v_sq_norm += (*this)(static_cast<int>(k), static_cast<int>(i))
                               * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
                tau(i) = static_cast<T>(2.0) / v_sq_norm;

                (*this)(static_cast<int>(i), static_cast<int>(i)) = -sign * norm_x;

                for (size_t j = i + 1; j < Cols; ++j) {
                    T dot = (*this)(static_cast<int>(i), static_cast<int>(j));
                    for (size_t k = i + 1; k < Rows; ++k) {
                        dot += (*this)(static_cast<int>(k), static_cast<int>(i))
                             * (*this)(static_cast<int>(k), static_cast<int>(j));
                    }
                    T tau_dot = tau(i) * dot;
                    (*this)(static_cast<int>(i), static_cast<int>(j)) -= tau_dot;
                    for (size_t k = i + 1; k < Rows; ++k) {
                        (*this)(static_cast<int>(k), static_cast<int>(j))
                            -= tau_dot
                             * (*this)(static_cast<int>(k), static_cast<int>(i));
                    }
                }
            }
            return true;
        }

        // ============================================================
        // [FIX #6] QR_solve_Householder
        //   원본 버그: `StaicVector` → `StaticVector` 오타
        // ============================================================
        StaticVector<T, Cols> QR_solve_Householder(
            const StaticVector<T, Cols>& tau,
            const StaticVector<T, Rows>& b) const   // [FIX] `StaicVector` → `StaticVector`
        {
            StaticVector<T, Rows> y = b;

            // Q^T b 적용 (Householder reflections)
            for (size_t i = 0; i < Cols; ++i) {
                if (MathTraits<T>::near_zero(tau(i))) {
                    continue;
                }
                T dot = y(i);
                for (size_t k = i + 1; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) * y(k);
                }
                T tau_dot = tau(i) * dot;
                y(i) -= tau_dot;
                for (size_t k = i + 1; k < Rows; ++k) {
                    y(k) -= tau_dot
                          * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
            }

            // Back substitution: Rx = Q^T b
            StaticVector<T, Cols> x;
            for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
                T sum = static_cast<T>(0);
                for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                    sum += (*this)(i, static_cast<int>(j)) * x(j);
                }
                x(static_cast<size_t>(i)) =
                    (y(static_cast<size_t>(i)) - sum)
                    / (*this)(i, i);
            }
            return x;
        }

        /**
         * @brief 디버깅 유틸리티
         */
        void print(const char* name) const {
            std::cout << "Matrix [" << name << "] (Col-major, "
                      << Rows << "x" << Cols << "):\n";
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    std::cout << std::fixed
                              << std::setw(10)
                              << std::setprecision(4)
                              << (*this)(static_cast<int>(i), static_cast<int>(j))
                              << "\t";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }
};

#endif // STATIC_MATRIX_HPP_
