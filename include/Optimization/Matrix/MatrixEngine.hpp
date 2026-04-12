#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#include <algorithm>  // std::copy, std::fill (연속 메모리 블록 단위의 고속 복사 및 초기화)
#include <cassert>  // 런타임 인덱스 경계 검사 (디버깅 용도, 릴리즈 시 NDEBUG로 비활성화 가능)
#include <cmath>     // std::abs, std::sqrt (기본 스칼라 수학 연산)
#include <iomanip>   // 출력 포맷팅
#include <iostream>  // 디버깅 콘솔 출력
#include <limits>  // std::numeric_limits (타입에 따른 머신 입실론 활용을 위해 필요)
#include <stdexcept>  // std::invalid_argument (0으로 나누기 등 치명적 수학 오류 예외 처리)
#include <type_traits>  // std::is_floating_point (AD Traits 템플릿 메타 프로그래밍 확장 대비)

#include "Optimization/Dual.hpp"  // Auto Differentiation(자동 미분)을 위한 Dual Number 구조체

/**
 * @brief 전방 선언 (Forward Declaration)
 * 클래스 내에서 자기 자신 또는 연관된 타입(StaticVector)을 참조하기 위해 선언.
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix;

/**
 * @brief Alias Template (상태 벡터)
 * 수학적 벡터(Column Vector)를 N x 1 행렬로 취급하여 타입 별칭 지정.
 * 메모리 구조상 행렬과 완벽히 동일하게 동작함.
 */
template <typename T, size_t N>
using StaticVector = StaticMatrix<T, N, 1>;

// ============================================================
// AD(Auto Differentiation) 호환 Traits 구조체
//   최적화 솔버(Layer 4)가 Jacobian/Hessian을 추출할 때, 자료형 T는
//   단순 실수(double)일 수도 있고 미분값을 포함하는 Dual 타입일 수도 있음.
//   제네릭 프로그래밍 환경에서 두 타입의 수학 연산을 통일하기 위한 인터페이스.
// ============================================================
template <typename T>
struct MathTraits {
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }

    /**
     * @brief Epsilon 특이성 검사 (기본 스칼라용)
     * 행렬 분해(LU, Cholesky 등) 시 피벗(Pivot)이 0에 가까운지 판단.
     * std::numeric_limits<T>::epsilon()을 사용하여 하드웨어 아키텍처에 맞는 최소 오차 적용.
     */
    static bool near_zero(const T& x) { return std::abs(x) <= std::numeric_limits<T>::epsilon(); }
};

// Dual 타입에 대한 템플릿 특수화(Specialization)
template <typename T>
struct MathTraits<Optimization::Dual<T>> {
    static Optimization::Dual<T> abs(const Optimization::Dual<T>& x) {
        return Optimization::ad::abs(x);  // Dual 내부의 오버로딩된 abs 호출
    }
    static Optimization::Dual<T> sqrt(const Optimization::Dual<T>& x) {
        return Optimization::ad::sqrt(x);  // Dual 내부의 오버로딩된 sqrt 호출
    }

    /**
     * @brief Epsilon 특이성 검사 (Dual 타입용)
     * 특이성(Singularity)은 미분값(gradient/hessian)이 아닌 순수 함수값(Value)에
     * 의해서만 결정되므로, 구조체 내부의 실수부(v)만 추출하여 입실론과 비교함.
     */
    static bool near_zero(const Optimization::Dual<T>& x) {
        return std::abs(x.v) <= std::numeric_limits<T>::epsilon();
    }
};

/**
 * @brief Layer 1 & 2: Static Matrix Engine (Column-major 레이아웃)
 *
 * 동적 할당(new/malloc)을 철저히 배제한 정적 메모리(Stack 또는 BSS) 기반 행렬 클래스.
 * NMPC의 실시간성(Hard Real-time)을 보장하고, Heap 단편화(Fragmentation)를 방지.
 *
 * Layer 1 - 기본 선형 대수 (산술 연산자, J-K-I 캐시 최적화 곱셈)
 * Layer 2 - 분해 솔버 (LU / Cholesky / LDLT / MGS-QR / Householder-QR)
 * + 블록 연산 (KKT 행렬과 같은 대형 희소/블록 행렬 조립용)
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
   private:
    // ============================================================
    // alignas(64): AVX-512 / 캐시 라인(Cache Line) 정렬 보장
    //   최신 CPU(64바이트 캐시 라인) 및 GPU/SIMD 환경에서 메모리 접근 패널티를 제거.
    //   SIMD 벡터화 연산 시 메모리가 정렬되어 있지 않으면 심각한 병목이 발생함.
    //   Column-major 배열이므로 data 배열 전체가 하나의 연속된 메모리 블록을 형성.
    // ============================================================
    alignas(64) T data[Rows * Cols]{};

   public:
    // ============================================================
    // 메모리 접근자 (Col-major: data[col * Rows + row])
    //   BLAS/LAPACK 표준과 동일한 Column-major 방식을 사용하여,
    //   열(Column) 방향 접근 시 공간 지역성(Spatial Locality)을 극대화.
    // ============================================================
    T& operator()(int r, int c) {
        // 인덱스 검증. 오버헤드를 막기 위해 릴리즈 빌드에서는 무시됨.
        assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
        return data[c * Rows + r];
    }
    const T& operator()(int r, int c) const {
        assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
        return data[c * Rows + r];
    }

    // 1차원 선형 메모리 직접 접근 (블록 복사 등에서 포인터 산술 연산용)
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
     * 매 제어 주기(Control Loop)마다 KKT 행렬을 재조립할 때 이전 주기의
     * 데이터(찌꺼기)를 초기화해야 함. 루프를 돌며 0을 넣는 것보다
     * std::fill을 사용하면 최적화 컴파일러가 이를 고속 memset 명령어로 치환함.
     */
    void set_zero() { std::fill(data, data + (Rows * Cols), static_cast<T>(0)); }

    // ============================================================
    // 기본 선형 대수 (Algebraic Operators)
    //   두 행렬의 덧셈/뺄셈/스칼라배는 요소별(Element-wise) 연산이므로
    //   2D 인덱스 대신 1D 선형 배열을 직접 순회하여 루프 오버헤드를 최소화.
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
        // AD 호환 Epsilon 검사를 통해 스칼라가 0에 가까운지 확인 (안전성 보장)
        if (MathTraits<T>::near_zero(scalar)) {
            throw std::invalid_argument("Division by near-zero scalar in StaticMatrix::operator/");
        }
        StaticMatrix<T, Rows, Cols> Result;
        const size_t total = Rows * Cols;
        // 나눗셈을 곱셈으로 변환. (CPU 사이클 측면에서 나눗셈은 곱셈보다 훨씬 느림)
        T inv_scalar = static_cast<T>(1.0) / scalar;
        for (size_t i = 0; i < total; ++i) {
            Result.data[i] = this->data[i] * inv_scalar;
        }
        return Result;
    }

    /**
     * @brief 행렬 곱셈 (J-K-I 루프 캐시 최적화)
     *
     * 표준적인 I-J-K 루프는 행렬 B의 열은 연속적으로 읽지만, 행렬 A의 행은
     * 건너뛰며 읽게 되어(Col-major 특성상) 심각한 캐시 미스(Cache Miss)를 유발함.
     * 이를 방지하기 위해 J-K-I 순서로 루프를 재배치함.
     * 이 구조에서는 가장 안쪽 루프(i)가 Result와 A 행렬의 메모리를
     * 완벽하게 순차적(Sequential)으로 접근하도록 설계됨.
     */
    template <size_t OtherCols>
    StaticMatrix<T, Rows, OtherCols> operator*(
        const StaticMatrix<T, Cols, OtherCols>& Other) const {
        StaticMatrix<T, Rows, OtherCols> Result;
        T* res_base = Result.data_ptr();
        const T* a_base = this->data;

        // J루프: Other 행렬의 열 이동
        for (size_t j = 0; j < OtherCols; ++j) {
            T* res_col = res_base + (j * Rows);  // Result의 j번째 열 시작 주소 캐싱
            // K루프: A 행렬의 열 이동, Other 행렬의 행 이동
            for (size_t k = 0; k < Cols; ++k) {
                const T* a_col = a_base + (k * Rows);  // A의 k번째 열 시작 주소 캐싱
                const T b_kj = Other(static_cast<int>(k), static_cast<int>(j));  // 스칼라 값 고정
                // I루프: Result와 A의 열 단위 연속 메모리 접근 (SIMD 벡터화가 발생하기 가장 좋은
                // 형태)
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
        // 원본 행렬의 열(j)을 순차적으로 읽어 대상 행렬의 행(j)에 흩뿌리는(Scatter) 구조.
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
    //   제어 아키텍처에서 시스템 방정식(Ax=b)을 풀기 위한 핵심 엔진.
    //   역행렬(Inverse)을 직접 구하는 것은 수치적으로 불안정하고 연산량이 많으므로
    //   반드시 분해(Decomposition) 후 전진/후진 대입(Substitution)을 수행함.
    // ============================================================

    /**
     * @brief LU 분해 (In-place Doolittle 알고리즘)
     *
     * 행렬 A를 하삼각행렬 L과 상삼각행렬 U로 제자리(in-place) 분해.
     * 메모리를 추가로 할당하지 않고 원본 행렬 공간에 L과 U를 덮어씀.
     * (L의 대각 성분은 1로 가정하여 저장하지 않음)
     *
     * @return true  - 분해 성공
     * @return false - Pivot(대각 원소)이 0에 가까워 분해 불가 (특이행렬, Singular Matrix)
     */
    bool LU_decompose() {
        static_assert(Rows == Cols, "LU Decomposition requires a square matrix");

        for (size_t i = 0; i < Rows; ++i) {
            // 1. U 행렬 갱신 (i번째 행의 j번째 열 원소들 계산)
            for (size_t j = i; j < Cols; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) -= sum;
            }

            // Pivot 안전성 검사. Pivot이 0이면 이후 나눗셈에서 무한대가 발생함.
            if (MathTraits<T>::near_zero((*this)(static_cast<int>(i), static_cast<int>(i)))) {
                return false;
            }

            // 2. L 행렬 갱신 (i번째 열의 j번째 행 원소들 계산)
            for (size_t j = i + 1; j < Rows; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(j), static_cast<int>(k)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
                }
                (*this)(static_cast<int>(j), static_cast<int>(i)) =
                    ((*this)(static_cast<int>(j), static_cast<int>(i)) - sum) /
                    (*this)(static_cast<int>(i), static_cast<int>(i));  // Pivot으로 나눔
            }
        }
        return true;
    }

    /**
     * @brief LU 분해 해법 (Forward + Backward Substitution)
     * Ax = LUx = b 시스템의 해를 구함.
     * 과정: 1) Ly = b 를 통해 y를 구함 (전진 대입)
     * 2) Ux = y 를 통해 x를 구함 (후진 대입)
     */
    StaticVector<T, Rows> LU_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> y, x;

        // 1단계: Forward substitution (Ly = b)
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t j = 0; j < i; ++j) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(j)) * y(j);
            }
            y(i) = b(i) - sum;  // L의 대각 성분은 1이므로 나눌 필요 없음
        }

        // 2단계: Back substitution (Ux = y)
        // 상삼각행렬이므로 맨 아래(Rows-1)부터 위로 올라가며 해를 계산
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
     * 행렬 A가 대칭(Symmetric)이고 양의 정부호(Positive Definite)일 때만 사용 가능.
     * LU 분해에 비해 연산량이 절반(O(N^3/3))이며 수치적으로 매우 안정적임.
     * Gauss-Newton(J^T J) 구조의 Hessian을 풀 때 가장 강력한 솔버.
     *
     * @return true  - A가 양의 정부호 행렬임 (분해 성공)
     * @return false - A가 비양정치 행렬이거나 음수 발생 (루트 내부가 음수)
     */
    bool Cholesky_decompose() {
        static_assert(Rows == Cols, "Cholesky requires a square matrix");

        for (size_t j = 0; j < Cols; ++j) {
            T s = static_cast<T>(0);
            // 대각 성분 계산을 위한 누산
            for (size_t k = 0; k < j; ++k) {
                s += (*this)(static_cast<int>(j), static_cast<int>(k)) *
                     (*this)(static_cast<int>(j), static_cast<int>(k));
            }
            T d = (*this)(static_cast<int>(j), static_cast<int>(j)) - s;

            // 양정치성(Positive Definiteness) 엄격 검사.
            // 0 이하이면 무조건 실패 반환. (행렬이 볼록성(Convexity)을 잃었음을 의미)
            if (d <= std::numeric_limits<T>::epsilon()) {
                return false;
            }
            (*this)(static_cast<int>(j), static_cast<int>(j)) = MathTraits<T>::sqrt(d);

            // 비대각 성분 계산
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
     * @brief Cholesky 분해 해법 (L*L^T * x = b)
     */
    StaticVector<T, Rows> Cholesky_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> x, y;

        // 1. Forward substitution: Ly = b
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < i; ++k) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(k)) * y(k);
            }
            y(i) = (b(i) - sum) / (*this)(static_cast<int>(i), static_cast<int>(i));
        }

        // 2. Back substitution: L^T x = y
        // 주의: L^T를 다루므로 원본 L(하삼각)의 인덱스 접근 순서가 행/열이 뒤집힘.
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
     * Cholesky와 유사하지만 루트(sqrt) 연산을 피하기 위해 설계됨.
     * 임베디드 환경에서 루트 연산은 매우 무거울 수 있으므로 대안으로 사용.
     * 이론적으로는 준정부호(Semi-definite) 행렬도 처리할 수 있으나,
     * 본 구현은 수치적 강건성을 위해 D_jj가 epsilon 이하일 경우 실패로 처리함.
     *
     * @return true  - 분해 성공
     * @return false - D가 0 이하 (음정치 또는 특이행렬)
     */
    bool LDLT_decompose() {
        static_assert(Rows == Cols, "LDLT requires a square matrix");

        for (size_t j = 0; j < Cols; ++j) {
            T sum_D = static_cast<T>(0);
            for (size_t k = 0; k < j; ++k) {
                T L_jk = (*this)(static_cast<int>(j), static_cast<int>(k));
                // L_jk^2 * D_kk 누산
                sum_D += L_jk * L_jk * (*this)(static_cast<int>(k), static_cast<int>(k));
            }

            T D_jj = (*this)(static_cast<int>(j), static_cast<int>(j)) - sum_D;

            // 비특이성 검사 (0 근접 또는 음수 거부)
            // LDLT도 NMPC의 KKT 매트릭스 특성상 강한 Positive Definite를 요구함.
            if (MathTraits<T>::near_zero(D_jj)) {
                return false;
            }
            (*this)(static_cast<int>(j), static_cast<int>(j)) = D_jj;  // 대각 원소 갱신 (D 저장)

            // L 행렬 갱신
            for (size_t i = j + 1; i < Rows; ++i) {
                T sum_L = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    sum_L +=
                        (*this)(static_cast<int>(i), static_cast<int>(k)) *
                        (*this)(static_cast<int>(j), static_cast<int>(k)) *
                        (*this)(static_cast<int>(k), static_cast<int>(k));  // L_ik * L_jk * D_kk
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) =
                    ((*this)(static_cast<int>(i), static_cast<int>(j)) - sum_L) /
                    (*this)(static_cast<int>(j), static_cast<int>(j));  // D_jj로 나눔
            }
        }
        return true;
    }

    /**
     * @brief LDLT 분해 해법 (L*D*L^T * x = b)
     */
    StaticVector<T, Rows> LDLT_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> z, y, x;

        // 1. Forward substitution: Lz = b (L의 대각은 1)
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < i; ++k) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(k)) * z(k);
            }
            z(i) = b(i) - sum;
        }

        // 2. 대각 스케일링: Dy = z
        // 이 과정이 Cholesky 분해의 sqrt 연산을 대체하는 역할을 함.
        for (size_t i = 0; i < Rows; ++i) {
            y(i) = z(i) / (*this)(static_cast<int>(i), static_cast<int>(i));
        }

        // 3. Back substitution: L^T x = y
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
     * @brief MGS-QR 분해 (Modified Gram-Schmidt 알고리즘)
     *
     * A = QR 로 분해. 직교 행렬 Q는 원본 위치(*this)에 저장되고,
     * 상삼각행렬 R은 인자로 받아 채워넣음.
     * 일반 CGS(Classical Gram-Schmidt)보다 수치적으로 훨씬 안정적임.
     * Gauss-Newton(J^T J) 구성 시 조건수(Condition number) 폭발을 막기 위해
     * Jacobian을 직접 QR 분해하는 아키텍처에서 활용 가능.
     *
     * @return true  - 분해 성공
     * @return false - 특정 열이 이전 열들과 선형 종속(Linear Dependent) 관계임
     */
    bool QR_decompose_MGS(StaticMatrix<T, Cols, Cols>& R) {
        static_assert(Rows >= Cols, "MGS-QR requires Rows >= Cols (Over-determined system)");

        for (size_t i = 0; i < Cols; ++i) {
            // 1. 현재 열 벡터의 Norm(크기) 계산
            T norm_sq = static_cast<T>(0);
            for (size_t k = 0; k < Rows; ++k) {
                norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            R(static_cast<int>(i), static_cast<int>(i)) = MathTraits<T>::sqrt(norm_sq);

            // 열의 크기가 0에 가까우면 직교 기반(Orthogonal basis)을 만들 수 없음
            if (MathTraits<T>::near_zero(R(static_cast<int>(i), static_cast<int>(i)))) {
                return false;
            }

            // 2. 현재 열 벡터를 정규화하여 Q 벡터로 변환 (원본 공간 덮어쓰기)
            for (size_t k = 0; k < Rows; ++k) {
                (*this)(static_cast<int>(k), static_cast<int>(i)) /=
                    R(static_cast<int>(i), static_cast<int>(i));
            }

            // 3. 현재 구해진 Q 벡터의 성분을 나머지 열들에서 모두 빼냄 (투영 소거)
            // 이것이 MGS가 일반 CGS보다 직교성을 오래 유지하는 비결.
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
     * @brief MGS-QR 해법 (Ax=b -> QRx=b -> Rx = Q^T b)
     */
    StaticVector<T, Cols> QR_solve(const StaticMatrix<T, Cols, Cols>& R,
                                   const StaticVector<T, Rows>& b) const {
        StaticVector<T, Cols> y, x;

        // 1. Q^T b 계산 (Q행렬은 *this에 저장되어 있음)
        for (size_t i = 0; i < Cols; ++i) {
            T dot = static_cast<T>(0);
            for (size_t k = 0; k < Rows; ++k) {
                dot += (*this)(static_cast<int>(k), static_cast<int>(i)) * b(k);
            }
            y(i) = dot;
        }

        // 2. Back substitution: Rx = y
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
     * @brief Householder QR 분해 (메모리 절약형 Compact 표현)
     *
     * MGS보다 직교성이 훨씬 보장되는 강력한 분해 기법.
     * 명시적인 Q 행렬을 만들지 않고, Householder 반사 벡터 v를
     * 하삼각 공간에 쑤셔넣어 저장(Compact Storage)하는 고도의 최적화 기법.
     * NMPC 등 제약조건(Constraints)이 심하게 걸린 Jacobian의 Null-space
     * 계산이 필요할 때 가장 완벽한 솔버 역할을 수행함.
     *
     * @param tau  Householder 반사 스케일링 팩터를 저장할 벡터
     * @return true (특이 열이 발생해도 tau=0 처리 후 계속 진행)
     */
    bool QR_decompose_Householder(StaticVector<T, Cols>& tau) {
        static_assert(Rows >= Cols, "Householder-QR requires Rows >= Cols");

        for (size_t i = 0; i < Cols; ++i) {
            // 현재 열의 서브벡터(i~Rows) Norm 누산.
            T norm_sq = static_cast<T>(0);
            for (size_t k = i; k < Rows; ++k) {
                norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            T norm_x = MathTraits<T>::sqrt(norm_sq);  // 누산 후 단 1회 루트 연산

            // 벡터가 완전히 0인 경우 반사를 스킵함
            if (MathTraits<T>::near_zero(norm_x)) {
                tau(i) = static_cast<T>(0);
                continue;
            }

            // 부호 처리 (수치적 안정성을 위해 -sign을 취하여 Cancellation 오차 방지)
            T sign = ((*this)(static_cast<int>(i), static_cast<int>(i)) >= static_cast<T>(0))
                         ? static_cast<T>(1.0)
                         : static_cast<T>(-1.0);
            T v0 = (*this)(static_cast<int>(i), static_cast<int>(i)) + sign * norm_x;

            // Householder 벡터 v의 나머지 성분을 행렬 하삼각 영역에 Compact하게 저장
            for (size_t k = i + 1; k < Rows; ++k) {
                (*this)(static_cast<int>(k), static_cast<int>(i)) /= v0;
            }

            // Tau(스케일 팩터) 계산
            T v_sq_norm = static_cast<T>(1.0);  // v0는 정규화되어 1로 취급
            for (size_t k = i + 1; k < Rows; ++k) {
                v_sq_norm += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                             (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            tau(i) = static_cast<T>(2.0) / v_sq_norm;

            // R 행렬의 대각 성분 확정
            (*this)(static_cast<int>(i), static_cast<int>(i)) = -sign * norm_x;

            // Rank-1 Update 적용 (A <- (I - tau*v*v^T) A)
            // 현재 열 이후의 모든 열(j > i)에 대해 직교 반사를 투영함
            for (size_t j = i + 1; j < Cols; ++j) {
                T dot = (*this)(static_cast<int>(i), static_cast<int>(j));  // v_0 = 1 성분
                for (size_t k = i + 1; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                T tau_dot = tau(i) * dot;

                // 원본 행렬 업데이트
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
     * @brief Householder QR 해법 (명시적인 Q 행렬 없이 풀기)
     */
    StaticVector<T, Cols> QR_solve_Householder(const StaticVector<T, Cols>& tau,
                                               const StaticVector<T, Rows>& b) const {
        StaticVector<T, Rows> y = b;

        // 1. Q^T b 적용: 저장된 Householder 벡터와 tau를 이용하여
        //    b 벡터를 연속적으로 직교 반사(Reflection)시킴.
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

        // 2. Back substitution: Rx = (Q^T b)
        // R 행렬은 상삼각 영역에 보존되어 있음.
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
    //   SQP 등에서 Hessian 블록, 제약조건(A) 블록 등을 모아 거대한
    //   KKT 시스템 행렬을 조립할 때 필수적인 유틸리티.
    //   Col-major의 특성인 '열 단위 연속성'을 극한으로 활용함.
    // ============================================================

    /**
     * @brief 부분 행렬 삽입 (std::copy를 통한 열 단위 블록 카피)
     *
     * 이중 루프(for i, for j)로 원소를 하나씩 할당하는 대신,
     * 한 열(Column) 전체가 메모리 상에 일렬로 늘어서 있다는 점을 활용.
     * std::copy를 한 번 호출할 때마다 하나의 열 블록 전체가 memcpy 속도로 복사됨.
     */
    template <size_t SubRows, size_t SubCols>
    void insert_block(size_t start_row, size_t start_col,
                      const StaticMatrix<T, SubRows, SubCols>& block) {
        // 타겟 행렬의 범위를 넘지 않는지 컴파일 타임 검사
        static_assert(SubRows <= Rows, "SubMatrix rows exceed target");
        static_assert(SubCols <= Cols, "SubMatrix cols exceed target");

        // 삽입 위치가 올바른지 런타임 검사
        assert(start_row + SubRows <= Rows && "Row index out of bounds");
        assert(start_col + SubCols <= Cols && "Col index out of bounds");

        for (size_t j = 0; j < SubCols; ++j) {
            // 원본 블록의 j번째 열 시작 주소
            const T* src = block.data_ptr() + (j * SubRows);
            // 타겟 행렬의 삽입 위치 시작 주소
            T* dest = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            // 한 열 만큼의 데이터를 통째로 복사
            std::copy(src, src + SubRows, dest);
        }
    }

    /**
     * @brief 전치(Transposed) 부분 행렬 삽입
     *
     * KKT 행렬의 좌하단에 제약조건 행렬 A의 전치 행렬(A^T)을 집어넣을 때 사용.
     * 행과 열이 뒤집히므로 std::copy와 같은 선형 복사 최적화를 쓸 수 없어,
     * 캐시를 흩뿌리는(Scattering) 2중 루프가 불가피함.
     */
    template <size_t SubRows, size_t SubCols>
    void insert_transposed_block(size_t start_row, size_t start_col,
                                 const StaticMatrix<T, SubRows, SubCols>& block) {
        // 전치되므로 검사 조건이 행렬 차원이 서로 바뀜
        static_assert(SubCols <= Rows, "Transposed block rows exceed target");
        static_assert(SubRows <= Cols, "Transposed block cols exceed target");
        assert(start_row + SubCols <= Rows && "Row index out of bounds");
        assert(start_col + SubRows <= Cols && "Col index out of bounds");

        for (size_t j = 0; j < SubCols; ++j) {
            for (size_t i = 0; i < SubRows; ++i) {
                // 타겟 행렬(this)에는 (j, i) 순서로 접근하여 전치 효과 부여
                (*this)(static_cast<int>(start_row + j), static_cast<int>(start_col + i)) =
                    block(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    /**
     * @brief 부분 행렬 추출 (std::copy를 통한 열 단위 블록 카피)
     *
     * 계산이 완료된 거대 KKT 벡터(상태 + 슬랙 + 듀얼)에서
     * 필요한 부분의 해(예: 차량 제어 입력 u)만 분리해 낼 때 사용.
     * insert_block과 역순으로 동일한 메모리 최적화 기법이 적용됨.
     */
    template <size_t SubRows, size_t SubCols>
    StaticMatrix<T, SubRows, SubCols> extract_block(size_t start_row, size_t start_col) const {
        assert(start_row + SubRows <= Rows && "Row index out of bounds");
        assert(start_col + SubCols <= Cols && "Col index out of bounds");

        StaticMatrix<T, SubRows, SubCols> result;
        for (size_t j = 0; j < SubCols; ++j) {
            // 원본 행렬의 추출 위치 주소
            const T* src = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            // 타겟(추출될 부분 행렬)의 j번째 열 시작 주소
            T* dest = result.data_ptr() + (j * SubRows);
            std::copy(src, src + SubRows, dest);
        }
        return result;
    }

    // ============================================================
    // 디버깅 유틸리티
    //   Solver 실패 시 Jitter 분석 및 KKT 매트릭스 시각 검증 용도.
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