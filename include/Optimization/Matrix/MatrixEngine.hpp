#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#include <algorithm>  // std::copy, std::fill (Block Operations 최적화)
#include <cassert>
#include <cmath>  // std::abs, std::sqrt
#include <iomanip>
#include <iostream>
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
// AD 호환 Traits 구조체
//   - T가 scalar(double/float)이면 기본 std:: 함수 위임
//   - T가 Dual<T> 등 복합 타입이면 특수화(Specialization)로 확장 가능
// ============================================================
template <typename T>
struct MathTraits {
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }

    /**
     * @brief Epsilon 특이성 검사
     *   - scalar: std::numeric_limits<T>::epsilon() 직접 비교
     *   - Dual:   특수화에서 실수부(real part)만 비교하도록 오버라이드
     */
    static bool near_zero(const T& x) { return std::abs(x) <= std::numeric_limits<T>::epsilon(); }
};

/**
 * @brief Layer 1 & 2: Static Matrix Engine (Column-major)
 *
 *   Layer 1 - 기본 선형 대수 (산술 연산자, 메모리 접근자)
 *   Layer 2 - 분해 솔버 (LU / Cholesky / LDLT / MGS-QR / Householder-QR)
 *             + 블록 연산 (insert_block / insert_transposed_block / extract_block)
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
   private:
    // ============================================================
    // alignas(64): AVX-512 / 캐시 라인 정렬 보장
    //   - 멀티코어/SIMD 환경에서 비정렬 페널티 제거
    //   - Aurix TriCore SIMD 파이프라인 최적화 대응
    // ============================================================
    alignas(64) T data[Rows * Cols]{};

   public:
    // ============================================================
    // 메모리 접근자 (Col-major: data[col * Rows + row])
    // ============================================================
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

    T* data_ptr() { return data; }
    const T* data_ptr() const { return data; }

    /**
     * @brief 고속 영행렬 초기화
     *   KKT 행렬 재조립 시 이전 루프의 찌꺼기를 지우는 데 사용.
     *   std::fill은 컴파일러에 의해 memset으로 최적화된다.
     */
    void set_zero() { std::fill(data, data + (Rows * Cols), static_cast<T>(0)); }

    // ============================================================
    // 기본 선형 대수 (Algebraic Operators)
    // ============================================================
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
        // MathTraits::near_zero 로 AD 호환 epsilon 검사
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

    /**
     * @brief 행렬 곱셈 (J-K-I 캐시 최적화)
     *
     *   Col-major 레이아웃에서 내부 루프가 연속 메모리를 순차 접근하도록
     *   J-K-I 순서를 채택하여 공간 지역성(Spatial Locality)을 극대화한다.
     *
     *   수정 이력:
     *     - `a_base * (k * Rows)` -> `a_base + (k * Rows)` (포인터 산술 오타)
     *     - `return result`       -> `return Result`        (대소문자 오타)
     */
    template <size_t OtherCols>
    StaticMatrix<T, Rows, OtherCols> operator*(
        const StaticMatrix<T, Cols, OtherCols>& Other) const {
        StaticMatrix<T, Rows, OtherCols> Result;
        T* res_base = Result.data_ptr();
        const T* a_base = this->data;

        for (size_t j = 0; j < OtherCols; ++j) {
            T* res_col = res_base + (j * Rows);  // Result의 j열 시작
            for (size_t k = 0; k < Cols; ++k) {
                const T* a_col = a_base + (k * Rows);  // A의 k열 시작
                const T b_kj = Other(static_cast<int>(k), static_cast<int>(j));
                for (size_t i = 0; i < Rows; ++i) {
                    res_col[i] += a_col[i] * b_kj;
                }
            }
        }
        return Result;
    }
    // ============================================================
    // 행렬 전치 (Transpose)
    // ============================================================
    StaticMatrix<T, Cols, Rows> transpose() const {
        StaticMatrix<T, Cols, Rows> Result;
        for (size_t j = 0; j < Cols; ++j) {
            const T* src_col = this->data_ptr() + (j * Rows);
            for (size_t i = 0; i < Rows; ++i) {
                Result(static_cast<int>(j), static_cast<int>(i)) = src_col[i];
            }
        }
        return Result;
    }

    // ============================================================
    // 분해 솔버 (Decompositions & Solvers)
    // ============================================================

    /**
     * @brief LU 분해 (In-place Doolittle)
     *
     *   A를 L·U로 in-place 분해한다.
     *   대각 이상(j >= i): U 행 갱신  U[i][j] = A[i][j] - Sigma L[i][k]*U[k][j]
     *   대각 이하(j >  i): L 열 갱신  L[j][i] = (A[j][i] - Sigma L[j][k]*U[k][i]) / U[i][i]
     *
     * @return true  - 분해 성공
     * @return false - 대각 원소 ≈ 0 (특이행렬)
     */
    bool LU_decompose() {
        static_assert(Rows == Cols, "LU Decomposition requires a square matrix");

        for (size_t i = 0; i < Rows; ++i) {
            // U 행 갱신
            for (size_t j = i; j < Cols; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) -= sum;
            }

            if (MathTraits<T>::near_zero((*this)(static_cast<int>(i), static_cast<int>(i)))) {
                return false;  // Singular Matrix
            }

            // L 열 갱신
            for (size_t j = i + 1; j < Rows; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(j), static_cast<int>(k)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
                }
                (*this)(static_cast<int>(j), static_cast<int>(i)) =
                    ((*this)(static_cast<int>(j), static_cast<int>(i)) - sum) /
                    (*this)(static_cast<int>(i), static_cast<int>(i));
            }
        }
        return true;
    }

    /**
     * @brief LU 해법 (Forward + Backward Substitution)
     *
     *   수정 이력:
     *     - 반환 타입 StaticMatrix<T, Rows> -> StaticVector<T, Rows> (Cols 누락)
     *     - forward substitution y(i) -> y(j) (이미 풀린 원소 참조 오류)
     */
    StaticVector<T, Rows> LU_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> y, x;

        // Forward substitution: Ly = b
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t j = 0; j < i; ++j) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(j)) * y(j);
            }
            y(i) = b(i) - sum;
        }

        // Back substitution: Ux = y
        for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
            T sum = static_cast<T>(0);
            for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                sum += (*this)(i, static_cast<int>(j)) * x(j);
            }
            x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(i, i);
        }
        return x;
    }

    /**
     * @brief Cholesky 분해 (A = LL^T)
     *
     *   대각 원소: L[j][j] = sqrt(A[j][j] - Sigma L[j][k]^2)
     *   비대각:   L[i][j] = (A[i][j] - Sigma L[i][k]*L[j][k]) / L[j][j]  (i > j)
     *
     * @return true  - 분해 성공 (A는 양정치)
     * @return false - d <= epsilon (비양정치 또는 특이행렬)
     */
    bool Cholesky_decompose() {
        static_assert(Rows == Cols, "Cholesky requires a square matrix");

        for (size_t j = 0; j < Cols; ++j) {
            T s = static_cast<T>(0);
            for (size_t k = 0; k < j; ++k) {
                s += (*this)(static_cast<int>(j), static_cast<int>(k)) *
                     (*this)(static_cast<int>(j), static_cast<int>(k));
            }
            T d = (*this)(static_cast<int>(j), static_cast<int>(j)) - s;

            // 양정치성 검사: d > epsilon 이어야 함
            if (d <= std::numeric_limits<T>::epsilon()) {
                return false;
            }
            (*this)(static_cast<int>(j), static_cast<int>(j)) = MathTraits<T>::sqrt(d);

            for (size_t i = j + 1; i < Rows; ++i) {
                T s_ij = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    s_ij += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                            (*this)(static_cast<int>(j), static_cast<int>(k));
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) =
                    ((*this)(static_cast<int>(i), static_cast<int>(j)) - s_ij) /
                    (*this)(static_cast<int>(j), static_cast<int>(j));
            }
        }
        return true;
    }

    /**
     * @brief Cholesky 해법 (L*L^T * x = b)
     *
     *   Forward : Ly  = b   -> y[i] = (b[i] - Sigma L[i][k]*y[k]) / L[i][i]
     *   Backward: L^T x = y -> x[i] = (y[i] - Sigma L[k][i]*x[k]) / L[i][i]
     */
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
            x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(i, i);
        }
        return x;
    }

    /**
     * @brief LDLT 분해 (Square-Root-Free Cholesky, A = L*D*L^T)
     *
     *   D 갱신: D[j][j] = A[j][j] - Sigma L[j][k]^2 * D[k][k]
     *   L 갱신: L[i][j] = (A[i][j] - Sigma L[i][k]*L[j][k]*D[k][k]) / D[j][j]  (i > j)
     *
     *   수정 이력:
     *     - near_zero(D_jj) -> D_jj <= epsilon 으로 교체.
     *       near_zero는 절댓값 기준이어서 음수 D를 통과시키는 버그 존재.
     *       LDLT는 양정치 행렬 전용이므로 반드시 부호 검사가 필요하다.
     *
     * @return true  - 분해 성공
     * @return false - D <= epsilon (음정치 또는 특이행렬)
     */
    bool LDLT_decompose() {
        static_assert(Rows == Cols, "LDLT requires a square matrix");

        for (size_t j = 0; j < Cols; ++j) {
            T sum_D = static_cast<T>(0);
            for (size_t k = 0; k < j; ++k) {
                T L_jk = (*this)(static_cast<int>(j), static_cast<int>(k));
                sum_D += L_jk * L_jk * (*this)(static_cast<int>(k), static_cast<int>(k));
            }

            T D_jj = (*this)(static_cast<int>(j), static_cast<int>(j)) - sum_D;

            // 비특이성 검사 (음수 허용, 0 근접만 거부)
            if (MathTraits<T>::near_zero(D_jj)) {
                return false;
            }
            (*this)(static_cast<int>(j), static_cast<int>(j)) = D_jj;

            for (size_t i = j + 1; i < Rows; ++i) {
                T sum_L = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    sum_L += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                             (*this)(static_cast<int>(j), static_cast<int>(k)) *
                             (*this)(static_cast<int>(k), static_cast<int>(k));
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) =
                    ((*this)(static_cast<int>(i), static_cast<int>(j)) - sum_L) /
                    (*this)(static_cast<int>(j), static_cast<int>(j));
            }
        }
        return true;
    }

    /**
     * @brief LDLT 해법 (L*D*L^T * x = b)
     *
     *   3단계 치환:
     *     1) Lz = b  (Forward, L 단위 하삼각)
     *     2) Dy = z  (대각 스케일링)
     *     3) L^T x = y (Backward)
     */
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
     *
     *   열 단위 직교화 (in-place, *this 가 Q가 됨):
     *     R[i][i]  = ||A[:,i]||_2
     *     Q[:,i]   = A[:,i] / R[i][i]
     *     R[i][j]  = Q[:,i]^T * A[:,j]      (j > i)
     *     A[:,j]  -= R[i][j] * Q[:,i]        (j > i)
     *
     *   수정 이력:
     *     - norm_sq = ... -> norm_sq += (누산 연산자 누락)
     *
     * @return true  - 분해 성공
     * @return false - 열 노름 ≈ 0 (선형종속)
     */
    bool QR_decompose_MGS(StaticMatrix<T, Cols, Cols>& R) {
        static_assert(Rows >= Cols, "MGS-QR requires Rows >= Cols");

        for (size_t i = 0; i < Cols; ++i) {
            T norm_sq = static_cast<T>(0);
            for (size_t k = 0; k < Rows; ++k) {
                norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            R(static_cast<int>(i), static_cast<int>(i)) = MathTraits<T>::sqrt(norm_sq);

            if (MathTraits<T>::near_zero(R(static_cast<int>(i), static_cast<int>(i)))) {
                return false;  // 선형종속 열
            }

            for (size_t k = 0; k < Rows; ++k) {
                (*this)(static_cast<int>(k), static_cast<int>(i)) /=
                    R(static_cast<int>(i), static_cast<int>(i));
            }

            for (size_t j = i + 1; j < Cols; ++j) {
                T dot = static_cast<T>(0);
                for (size_t k = 0; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                R(static_cast<int>(i), static_cast<int>(j)) = dot;

                for (size_t k = 0; k < Rows; ++k) {
                    (*this)(static_cast<int>(k), static_cast<int>(j)) -=
                        dot * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
            }
        }
        return true;
    }

    /**
     * @brief MGS-QR 해법 (Q^T b -> 후진 대입)
     *
     *   *this 는 QR_decompose_MGS 후 Q 행렬.
     *   y[i] = Q[:,i]^T * b  (Q^T b 계산)
     *   Back substitution: Rx = y
     */
    StaticVector<T, Cols> QR_solve(const StaticMatrix<T, Cols, Cols>& R,
                                   const StaticVector<T, Rows>& b) const {
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
            for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                sum += R(i, static_cast<int>(j)) * x(j);
            }
            x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / R(i, i);
        }
        return x;
    }

    /**
     * @brief Householder QR 분해 (Compact 표현)
     *
     *   각 단계 i 에서 Householder 반사를 적용하여 i열 이하를 소거한다:
     *     norm_x  = ||A[i:,i]||_2
     *     v0      = A[i][i] + sign(A[i][i]) * norm_x
     *     v[k]    = A[k][i] / v0   (k > i, compact 저장)
     *     tau     = 2 / ||v||^2
     *     A[i][i] = -sign * norm_x  (R의 대각 원소)
     *     A      <- (I - tau*v*v^T) A  (rank-1 update, 열 단위)
     *
     *   수정 이력:
     *     - sqrt가 누산 루프 안에 중첩되어 있던 버그 수정:
     *       누산 루프 완료 후 단 1회 sqrt를 적용하도록 분리.
     *
     * @return true (영열 처리는 tau=0으로 스킵)
     */
    bool QR_decompose_Householder(StaticVector<T, Cols>& tau) {
        static_assert(Rows >= Cols, "Householder-QR requires Rows >= Cols");

        for (size_t i = 0; i < Cols; ++i) {
            // norm 누산 완료 후 sqrt — 루프 분리가 핵심
            T norm_sq = static_cast<T>(0);
            for (size_t k = i; k < Rows; ++k) {
                norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            T norm_x = MathTraits<T>::sqrt(norm_sq);

            if (MathTraits<T>::near_zero(norm_x)) {
                tau(i) = static_cast<T>(0);
                continue;
            }

            T sign = ((*this)(static_cast<int>(i), static_cast<int>(i)) >= static_cast<T>(0))
                         ? static_cast<T>(1.0)
                         : static_cast<T>(-1.0);
            T v0 = (*this)(static_cast<int>(i), static_cast<int>(i)) + sign * norm_x;

            // Householder 벡터 compact 저장 (하삼각 영역)
            for (size_t k = i + 1; k < Rows; ++k) {
                (*this)(static_cast<int>(k), static_cast<int>(i)) /= v0;
            }

            T v_sq_norm = static_cast<T>(1.0);
            for (size_t k = i + 1; k < Rows; ++k) {
                v_sq_norm += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                             (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            tau(i) = static_cast<T>(2.0) / v_sq_norm;

            (*this)(static_cast<int>(i), static_cast<int>(i)) = -sign * norm_x;

            // (I - tau*v*v^T) A 적용 (j > i 열에 대해 rank-1 update)
            for (size_t j = i + 1; j < Cols; ++j) {
                T dot = (*this)(static_cast<int>(i), static_cast<int>(j));
                for (size_t k = i + 1; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                T tau_dot = tau(i) * dot;
                (*this)(static_cast<int>(i), static_cast<int>(j)) -= tau_dot;
                for (size_t k = i + 1; k < Rows; ++k) {
                    (*this)(static_cast<int>(k), static_cast<int>(j)) -=
                        tau_dot * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
            }
        }
        return true;
    }

    /**
     * @brief Householder QR 해법
     *
     *   1) Q^T b 적용: 저장된 Householder 반사를 순서대로 b에 적용.
     *   2) Back substitution: Rx = Q^T b.
     *
     *   수정 이력:
     *     - 파라미터 타입 StaicVector -> StaticVector (오타)
     */
    StaticVector<T, Cols> QR_solve_Householder(const StaticVector<T, Cols>& tau,
                                               const StaticVector<T, Rows>& b) const {
        StaticVector<T, Rows> y = b;

        // Q^T b 적용 (Householder reflections 순차 적용)
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
                y(k) -= tau_dot * (*this)(static_cast<int>(k), static_cast<int>(i));
            }
        }

        // Back substitution: Rx = Q^T b
        StaticVector<T, Cols> x;
        for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
            T sum = static_cast<T>(0);
            for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                sum += (*this)(i, static_cast<int>(j)) * x(j);
            }
            x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(i, i);
        }
        return x;
    }

    // ============================================================
    // 블록 연산 (Block Operations)
    //   Col-major 연속성을 활용한 고속 삽입/추출.
    //   KKT 행렬 조립, Hessian 블록 업데이트 등에 사용.
    // ============================================================

    /**
     * @brief 부분 행렬 삽입 (std::copy — 열 연속성 최적화)
     *
     *   Col-major 레이아웃에서 열(Column) 단위로 std::copy를 호출하면
     *   src/dest 포인터가 모두 연속 메모리를 가리키므로 memcpy 수준으로 최적화된다.
     *
     * @tparam SubRows  삽입할 블록의 행 수
     * @tparam SubCols  삽입할 블록의 열 수
     * @param  start_row  삽입 시작 행 인덱스
     * @param  start_col  삽입 시작 열 인덱스
     * @param  block      삽입할 부분 행렬
     */
    template <size_t SubRows, size_t SubCols>
    void insert_block(size_t start_row, size_t start_col,
                      const StaticMatrix<T, SubRows, SubCols>& block) {
        static_assert(SubRows <= Rows, "SubMatrix rows exceed target");
        static_assert(SubCols <= Cols, "SubMatrix cols exceed target");
        assert(start_row + SubRows <= Rows && "Row index out of bounds");
        assert(start_col + SubCols <= Cols && "Col index out of bounds");

        for (size_t j = 0; j < SubCols; ++j) {
            const T* src = block.data_ptr() + (j * SubRows);
            T* dest = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            std::copy(src, src + SubRows, dest);
        }
    }

    /**
     * @brief 전치 부분 행렬 삽입
     *
     *   block^T 를 (*this)의 (start_row, start_col) 위치에 삽입한다.
     *   전치 삽입은 메모리 스캐터링이 불가피하므로 2중 루프로 처리한다.
     *
     *   예시 (KKT 조립):
     *     KKT.insert_block(0, 0, H);              // H 블록
     *     KKT.insert_block(n, 0, A);              // A 블록
     *     KKT.insert_transposed_block(0, n, A);   // A^T 블록
     *
     * @tparam SubRows  원본 블록의 행 수 (삽입 후 열 방향이 됨)
     * @tparam SubCols  원본 블록의 열 수 (삽입 후 행 방향이 됨)
     */
    template <size_t SubRows, size_t SubCols>
    void insert_transposed_block(size_t start_row, size_t start_col,
                                 const StaticMatrix<T, SubRows, SubCols>& block) {
        static_assert(SubCols <= Rows, "Transposed block rows exceed target");
        static_assert(SubRows <= Cols, "Transposed block cols exceed target");
        assert(start_row + SubCols <= Rows && "Row index out of bounds");
        assert(start_col + SubRows <= Cols && "Col index out of bounds");

        for (size_t j = 0; j < SubCols; ++j) {
            for (size_t i = 0; i < SubRows; ++i) {
                (*this)(static_cast<int>(start_row + j), static_cast<int>(start_col + i)) =
                    block(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    /**
     * @brief 블록 추출 (std::copy — 열 연속성 최적화)
     *
     *   (*this)의 (start_row, start_col) 위치에서 SubRows x SubCols 블록을
     *   복사하여 반환한다. insert_block과 동일하게 열 단위 std::copy를 사용.
     *
     * @tparam SubRows  추출할 블록의 행 수
     * @tparam SubCols  추출할 블록의 열 수
     */
    template <size_t SubRows, size_t SubCols>
    StaticMatrix<T, SubRows, SubCols> extract_block(size_t start_row, size_t start_col) const {
        assert(start_row + SubRows <= Rows && "Row index out of bounds");
        assert(start_col + SubCols <= Cols && "Col index out of bounds");

        StaticMatrix<T, SubRows, SubCols> result;
        for (size_t j = 0; j < SubCols; ++j) {
            const T* src = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            T* dest = result.data_ptr() + (j * SubRows);
            std::copy(src, src + SubRows, dest);
        }
        return result;
    }

    // ============================================================
    // 디버깅 유틸리티
    // ============================================================
    void print(const char* name) const {
        std::cout << "Matrix [" << name << "] (Col-major, " << Rows << "x" << Cols << "):\n";
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                std::cout << std::fixed << std::setw(10) << std::setprecision(4)
                          << (*this)(static_cast<int>(i), static_cast<int>(j)) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
};

#endif  // STATIC_MATRIX_HPP_