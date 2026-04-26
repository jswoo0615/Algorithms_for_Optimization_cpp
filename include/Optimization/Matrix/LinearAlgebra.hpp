#ifndef OPTIMIZATION_LINEAR_ALGEBRA_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"

namespace Optimization {
// =======================================================================================
// AD (Auto Differentiation, 자동 미분) 호환 Traits 구조체
//
//  [배경 설명]
//      최적화 솔버 (Layer 4)가 Jacobian 행렬이나 Hessian 행렬을 추출하는 과정에서,
//      행렬의 원소 자료형 (Type) `T`는 단순 실수형 (`double`)일 수도 있고,
//      미분값 (Gradient)을 함께 추적하는 `Optimization::Dual` 타입일 수도 있습니다.
//      C++ 제네릭 프로그래밍 환경에서는 이 두 가지 완전히 다른 타입에 대해
//      단일화된 수학 연산 인터페이스를 제공해야 컴파일 오류가 발생하지 않습니다.
//      이를 위해 `MathTraits` 라는 템플릿 구조체를 도입하여 타입별 연산 방식을 분기 처리합니다.
// =======================================================================================
template <typename T>
struct MathTraits {
    // 일반 스칼라 타입 (예 : double, float)에 대한 수학 함수 매핑
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }

    /**
     * @brief Epsilon 특이성 검사 (기본 스칼라 타입용)
     *
     * [수치적 안정성 판단]
     * 선형 대수 분해 (LU, Cholesky, LDLT, QR 등) 과정에서 (Pivot, 대각 원소)으로 나누는 연산이
     * 빈번합니다. 이때 피벗이 수학적 0은 아니더라도 0에 극도로 가까운 수학적 노이즈라면 (특이 행렬,
     * Singular), 나눗셈 결과가 무한대로 발산하여 시스템이 붕괴합니다. 따라서
     * `std::numeric_limits<T>::epsilon()`을 사용하여 하드웨어 부동소숫점 아키텍처에 맞는 최소 허용
     * 오차 범위 내에 값이 존재하는지 (사실상 0인지) 안전하게 판단합니다.
     */
    static bool near_zero(const T& x) { return std::abs(x) <= std::numeric_limits<T>::epsilon(); }
};

/**
 * @brief Optimization::Dual 타입에 대한 템플릿 특수화
 * @note
 * Dual 타입은 내부에 실수부 (v)와 미분부 (d)를 모두 가지므로,
 * Dual 헤더에 오버로딩된 특수 수학 함수를 명시적으로 호출합니다.
 */
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
     *
     * [논리적 주의점]
     * 분해 과정에서 행렬이 특이행렬 (Singular Matrix)인지 판별하는 기준은,
     * 행렬 원소의 `미분값 (gradient/hessian)`이 아니라 오직 순수 `함수값 (Value, 실수부)`에
     * 의해서만 결정되어야 합니다 따라서 Dual 구조체 전체의 크기를 비교하는 것이 아니라, 내부의
     * 실수부 멤버인 `v`만 추출하여 입실론과 비교합니다.
     */
    static bool near_zero(const Optimization::Dual<T>& x) {
        return std::abs(x.v) <= std::numeric_limits<T>::epsilon();
    }
};

// ============================================================
// 기본 선형 대수 연산 (Algebraic Operators)
// ============================================================
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
        throw std::invalid_argument("Division by near-zero scalar in StaticMatrix::operator/()");
    }
    StaticMatrix<T, Rows, Cols> Result;
    const size_t total = Rows * Cols;
    /**
     * @brief 연산 최적화 팁
     * @note
     * CPU 명령어 사이클 측면에서 부동소숫점 나눗셈 (FDIV)은 곱셈 (FMUL)에 비해 약 10 ~ 20배 정도
     * 압도적으로 느립니다. 루프 내에서 N*M 번 나누는 대신, 먼저 스칼라의 역수 (Inverse)를 단 1번만
     * 구하고, 루프 안에서는 역수를 곱해주는 방식으로 변환하여 엄청난 속도 이득을 얻습니다.
     */
    T inv_scalar = static_cast<T>(1.0) / scalar;
    for (size_t i = 0; i < total; ++i) {
        Result(i) = A(i) * inv_scalar;
    }
    return Result;
}

/**
 * @brief 행렬 곱셈 연산자 오버로딩 (J-K-I 루프 캐시 최적화 적용)
 *
 * [이론적 배경 및 최적화 원리]
 * 일반적인 학부 수준의 행렬 곱셈 (C = A * B) 루프 순서는 행 (i) -> 열 (j) -> 내적 (k)
 * 즉, "I-J-K" 순서입니다.
 * 그러나 이 클래스는 Column-major (열 우선) 메모리 레이아웃을 취하고 있습니다.
 * I-J-K 순서로 연산할 경우, 행렬 B의 요소는 열 (j)을 따라가므로 메모리를 연속적으로 예쁘게 읽지만,
 * 행렬 A의 요소는 같은 행 (i) 의 다음 열 (k) 요소로 넘어가기 위해 메모리 주소를
 * 매번 건너뛰면서 (Stride가 Row 크기) 읽게 됩니다. 이는 CPU 캐시에 담겨있지 않은 엉뚱한 주소를
 * 계속 요구하므로 막대한 **캐시 미스 (Cache Miss)** 패널티를 유발하여 성능을 떨어뜨립니다.
 *
 * 이를 완벽히 방지하기 위해, 루프의 순서를 "J-K-I" 순서로 재배치 (Loop Reordering) 했습니다
 * 이 J-K-I 구조에서는 가장 빈번하게 도는 최하단 (가장 안쪽) I-루프가
 * Result 행렬의 특정 열 (res_col)과, 행렬 A의 특정 열 (a_col)을 위에서 아래로 (행 i를 증가시키며)
 * 순차적으로 쭉 훑고 지나갑니다. 이 형태는 메모리에 완전히 일렬로 순차 접근 (Sequential Access)하는
 * 형태이므로 캐시 적중률이 100%에 가까워지며, 컴파일러가 SIMD 벡터화 (AVX, SSE) 인스트럭션을
 * 적용하여 루프를 병렬로 묶어 처리하기 가장 이상적인 형태가 됩니다.
 */
template <typename T, size_t Rows, size_t Cols, size_t OtherCols>
StaticMatrix<T, Rows, OtherCols> operator*(const StaticMatrix<T, Rows, Cols>& A, const StaticMatrix<T, Cols, OtherCols>& B) {
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
// ============================================================
// 행렬 전치 (Matrix Transpose)
// ============================================================
template <typename T, size_t Rows, size_t Cols>
StaticMatrix<T, Cols, Rows> transpose(const StaticMatrix<T, Rows, Cols>& mat) {
    StaticMatrix<T, Cols, Rows> Result;
    /**
     * @note
     * 원본 행렬의 열 (j)을 순차적 (Sequential)으로 쭉 ㅇ릭어서,
     * 대상 행렬의 행 (j) 위치에 여기저기 흩뿌려 (Scatter) 할당하는 방식입니다.
     * 읽는 쪽이라도 캐시 연속성을 확보하기 위해 외부 루프를 j로 잡습니다
     */
    for (size_t j = 0; j < Cols; ++j) {
        const T* src_col = mat.data_ptr() + (j * Rows);
        for (size_t i = 0; i < Rows; ++i) {
            // 대상 Result 행렬은 (j, i) 순으로 접근되어 메모리 점프가 발생하지만,
            // 전치의 특성상 불가피합니다.
            Result(static_cast<int>(j), static_cast<int>(i)) = src_col[i];
        }
    }
    return Result;
}

/**
 * @brief 선형 시스템 분해 솔버 (Decompositions & Solvers)
 *
 * [솔버 설계의 필요성]
 * 모델 예측 제어 (MPC)의 핵심은 매 주기마다 거대한 KKT 형태의 선형 연립 방정식 시스템 (Ax = b)을
 * 푸는 것입니다. 수학적으로 x = A^{-1} * b 이지만, 컴퓨터로 직접 A의 역행렬 (Inverse)을 통ㅎ째로
 * 구하는 것은 O(N^3)의 막대한 연산량이 소모되며 무엇보다 부동소숫점 오차가 기하급수적으로 누적되어
 * 시스템이 불안정해집니다 (수치적 불안정성) 따라서 현대 수치해석에서는 반드시 행렬 A를 다루기 쉬운
 * 두세 개의 삼각 행렬 조각으로 "분해 (Decomposition)"한 뒤, 전진 대입 (Forward Substitution)과 후진
 * 대입 (Backward Substitution)이라는 아주 싼 연산 (O(N^2))을 통해 빠르고 정확하게 해 (x)를
 * 구합니다.
 */
// ============================================================
    // 1. LUP 분해 및 솔버 (Partial Pivoting 도입으로 수치적 안정성 극대화)
    // ============================================================
    template <typename T, size_t Rows, size_t Cols>
    bool LU_decompose(StaticMatrix<T, Rows, Cols>& mat, StaticVector<int, Rows>& P) {
        static_assert(Rows == Cols, "LUP Decomposition requires a square matrix");

        // 피벗 배열 초기화 (P[i] = i)
        for (size_t i = 0; i < Rows; ++i) P(i) = static_cast<int>(i);

        for (size_t i = 0; i < Rows; ++i) {
            // 1. 부분 피보팅 (Partial Pivoting): 현재 열에서 절대값이 가장 큰 행 찾기
            T max_val = static_cast<T>(0);
            size_t pivot_row = i;
            for (size_t j = i; j < Rows; ++j) {
                T abs_val = MathTraits<T>::abs(mat(static_cast<int>(j), static_cast<int>(i)));
                if (abs_val > max_val) {
                    max_val = abs_val;
                    pivot_row = j;
                }
            }

            // 행렬 특이성(Singularity) 검사
            if (MathTraits<T>::near_zero(max_val)) return false;

            // 2. 행 교환 (Row Swapping)
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

            // 3. 분해 (Schur Complement 업데이트 기반)
            for (size_t j = i + 1; j < Rows; ++j) {
                mat(static_cast<int>(j), static_cast<int>(i)) /= mat(static_cast<int>(i), static_cast<int>(i)); // L 성분
                for (size_t k = i + 1; k < Cols; ++k) {
                    mat(static_cast<int>(j), static_cast<int>(k)) -= mat(static_cast<int>(j), static_cast<int>(i)) * mat(static_cast<int>(i), static_cast<int>(k)); // U 성분
                }
            }
        }
        return true;
    }

    template <typename T, size_t Rows, size_t Cols>
    StaticVector<T, Rows> LU_solve(const StaticMatrix<T, Rows, Cols>& mat, const StaticVector<int, Rows>& P, const StaticVector<T, Rows>& b) {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> x;

        // 1. 피벗 배열을 바탕으로 우항 벡터 b를 섞음 (Permutation)
        for (size_t i = 0; i < Rows; ++i) {
            x(i) = b(P(i));
        }

        // 2. 전진 대입 (Forward substitution: L * y = Pb) - x 배열을 임시 y로 활용
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < i; ++j) {
                x(i) -= mat(static_cast<int>(i), static_cast<int>(j)) * x(j);
            }
        }

        // 3. 후진 대입 (Backward substitution: U * x = y)
        for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
            for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                x(static_cast<size_t>(i)) -= mat(i, static_cast<int>(j)) * x(j);
            }
            x(static_cast<size_t>(i)) /= mat(i, i);
        }
        return x;
    }

/**
 * @brief 촐레스키 분해 (Cholesky Decomposition, A = L * L^{T})
 *
 * [제약 조건 및 강점]
 * 대상 행렬 A가 반드시 대칭 행렬 (Symmetric, A = A^T)이고, 동시에 양의 정부호 (Positive Definite,
 * 모든 고윳값이 양수) 일 때만 수학적으로 성립하는 특수 분해 기법입니다. 일반 LU 분해에 비해 필요한
 * 반복 연산량이 정확히 절반 (O(N^3/3))으로 줄어들며, 부동소숫점 오차에 대한 수치적 안정성
 * (Numerical Stability)이 압도적으로 좋습니다. SQP 최적화나 Gauss-Newton (J^T J 구조) 형태의
 * 역삭/제어 문제를 풀 때, Hessian 행렬은 기본적으로 양의 정부호 대칭성을 띠므로 이때 가장 강력한
 * 최우선 솔버로 활용됩니다. 분해 결과인 L 행렬은 하삼각 공간에 덮어써서 저장되며 상삼각 영역은
 * 버려집니다
 *
 * @return true - A가 양의 정부호 행렬임이 증명되어 분해가 정상 성공함
 * @return false - A가 비양정치 행렬 (Non-Positive Definite)이거나 음수 고윳값이 발생함.
 *          (수식 내부에서 루트 안에 음수가 들어가려 할 때 즉시 발각되어 차단됨)
 */
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

        for (size_t i = j + 1; i < Rows; ++i) {
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
        for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
            sum += mat(static_cast<int>(k), i) * x(k);
        }
        x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / mat(i, i);
    }
    return x;
}

/**
 * @brief LDLT 분해 (Square-Root-Free Cholesky, A = L * D * L^T)
 *
 * [등장 배경]
 * Cholesky 분해는 매 열마다 무거운 무리수 루트 (sqrt) 연산을 강제합니다.
 * CPU성능이 매우 제한된 구형 임베디드 마이크로컨트롤러에서는 이는 병목 현상이 될 수 있습니다.
 * 루트 연산을 수학적으로 우회하여 제거하기 위해 대각 행렬 D를 중앙에 끼워넣는 방식으로 변형한 것이
 * LDLT 분해입니다.
 *
 * 이론상 행렬이 0의 고윳값을 갖는 준정부호 (Positive Semi-Definite) 행렬이더라도 분해가 가능하도록
 * 설계되었으나, 본 엔진은 제어 솔버 (NMPC, SQP) 내부망에 통합되어 있으므로 알고리즘의 붕괴를 사전에
 * 막기 위해 D의 원소가 엄격한 양수 (Positive Definite)인지 입실론 수준에서 강하게 검열합니다.
 * 하삼각 영역에 L을, 대각선 영역에 D 원소를 동시에 덮어써서 콤팩트하게 저장합니다.
 * (이때  L의 대각은 암묵적 1로 처리합니다)
 *
 * @return true - 분해 성공
 * @return false - D 성분이 0 이하 (행렬이 음정치 또는 특이행렬 결함이 발생함)
 */
template <typename T, size_t Rows, size_t Cols>
bool LDLT_decompose(StaticMatrix<T, Rows, Cols>& mat) {
    static_assert(Rows == Cols, "LDLT requires a square matrix");
    for (size_t j = 0; j < Cols; ++j) {
        T sum_D = static_cast<T>(0);
        for (size_t k = 0; k < j; ++k) {
            T L_jk = mat(static_cast<int>(j), static_cast<int>(k));
            sum_D += L_jk * L_jk * mat(static_cast<int>(k), static_cast<int>(k));
        }
        T D_jj = mat(static_cast<int>(j), static_cast<int>(j)) - sum_D;
        if (MathTraits<T>::near_zero(D_jj)) {
            return false;
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
    return true;
}

/**
 * @brief LDLT 분해 해법 (L * D * L^T * x = b)
 *
 * 과정이 총 3단계 (전진 대입 -> 대각 스케일링 조정 -> 후진 대입)로 진행됩니다
 */

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Rows> LDLT_solve(const StaticMatrix<T, Rows, Cols>& mat,
                                 const StaticVector<T, Rows>& b) {
    static_assert(Rows == Cols, "Solve requires a square matrix");
    StaticVector<T, Rows> z, y, x;

    /**
     * @brief 1단계 : 전진 대입 Lz = b
     * @note
     *  LDLT에서 L의 대각 성분은 암묵적으로 항상 1.0으로 고정되어 있으므로,
     *  이 부분에서는 나눗셈이 발생하지 않아 연산이 쾌적합니다.
     */

    for (size_t i = 0; i < Rows; ++i) {
        T sum = static_cast<T>(0);
        for (size_t k = 0; k < i; ++k) {
            sum += mat(static_cast<int>(i), static_cast<int>(k)) * z(k);
        }
        z(i) = b(i) - sum;
    }

    /**
     * @brief 2단계 : 중앙 대각 행렬 (D) 스케일링 : Dy = z
     * 이 한 줄의 나눗셈 과정이 Cholesky 분해의 무거운 무리수 루트 (sqrt) 연산을 대체하는 수학적
     * 역할입니다.
     */

    for (size_t i = 0; i < Rows; ++i) {
        y(i) = z(i) / mat(static_cast<int>(i), static_cast<int>(i));
    }

    /**
     * @brief 3단계 : 후진 대입 L^T x = y
     * 여기서도 마찬가지로 L^T의 대각 성분은 암묵적 1이므로
     * 뺄셈만 하고 L_ii로 나누는 절차는 생략됩니다.
     */

    for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
            sum += mat(static_cast<int>(k), i) * x(k);  // 인덱스 뒤집기 트릭
        }
        x(static_cast<size_t>(i)) = y(static_cast<size_t>(i)) - sum;
    }
    return x;
}

/**
 * @brief MGS-QR 분해 (Modified Gram-Schmidt 기반의 A = Q * R 직교 분해)
 *
 * 정방 행렬뿐만 아니라 행이 더 많은 직사각형 행렬 (Over-determined System)도 처리할 수 있는
 * 직교 (Orthogonal) 기반 분해 솔버입니다. 행렬 A의 열 벡터들을 순차적으로 정규화 (Normalize)하고
 * 직교화 (Orthogonalize)하여 직교 행렬 Q와 상삼각 행렬 R을 만들어냅니다.
 *
 * [MGS (Modified)의 핵심 최적화]
 * 학부 수준의 고전적인 CGS (Classical Gram-Schmidt) 알고리즘은 계산 중 부동소숫점 오차가
 * 기하급수적으로 쌓여 최종 도출된 Q 벡터들이 직교성을 상실하고 한쪽으로 무너지는 끔찍한 단점이
 * 있습니다. MGS 알고리즘은 이를 방지하기 위해, 하나의 직교 Q열을 찾아낼 때마다 그 직교성을 "아직
 * 처리하지 않은 모든 나머지 열들에서" 즉각적으로 깎아내고 빼버리는 (투영 소거, Projection) 전략을
 * 사용하여, 컴퓨터 수치 환경에서 훨씬 강건하고 완벽한 직교 기반을 유지합니다. Q 행렬은 원본 A 공간
 * (*this)에 덮어써지고, R 행렬은 별도의 외부 변수를 통해 반환받습니다.
 *
 * @param R (출력용) 분해된 상삼각행렬 R을 담을 외부 행렬 객체
 * @return true - 분해 성공
 * @return false - 특정 열이 이전 열 벡터들과 선형 종속 (Linear Dependent) 관계여서 직교 기반 생성이
 * 불가능할 경우
 */
template <typename T, size_t Rows, size_t Cols>
bool QR_decompose_MGS(StaticMatrix<T, Rows, Cols>& mat, StaticMatrix<T, Cols, Cols>& R) {
    static_assert(Rows >= Cols, "MGS-QR requires Rows >= Cols (Over-determined system)");
    for (size_t i = 0; i < Cols; ++i) {
        T norm_sq = static_cast<T>(0);
        for (size_t k = 0; k < Rows; ++k) {
            norm_sq += mat(static_cast<int>(k), static_cast<int>(i)) *
                       mat(static_cast<int>(k), static_cast<int>(i));
        }
        R(static_cast<int>(i), static_cast<int>(i)) = MathTraits<T>::sqrt(norm_sq);

        if (MathTraits<T>::near_zero(R(static_cast<int>(i), static_cast<int>(i)))) {
            return false;
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
            R(static_cast<int>(i), static_cast<int>(j)) = dot;

            for (size_t k = 0; k < Rows; ++k) {
                mat(static_cast<int>(k), static_cast<int>(j)) -=
                    dot * mat(static_cast<int>(k), static_cast<int>(i));
            }
        }
    }
    return true;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Cols> QR_solve(const StaticMatrix<T, Rows, Cols>& mat,
                               const StaticMatrix<T, Cols, Cols>& R,
                               const StaticVector<T, Rows>& b) {
    StaticVector<T, Cols> y, x;
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

// 5. Householder QR 분해 및 솔버
template <typename T, size_t Rows, size_t Cols>
bool QR_decompose_Householder(StaticMatrix<T, Rows, Cols>& mat, StaticVector<T, Cols>& tau) {
    static_assert(Rows >= Cols, "Householder-QR requires Rows >= Cols");
    for (size_t i = 0; i < Cols; ++i) {
        T norm_sq = static_cast<T>(0);
        for (size_t k = i; k < Rows; ++k) norm_sq += mat(static_cast<int>(k), static_cast<int>(i)) * mat(static_cast<int>(k), static_cast<int>(i));
        T norm_x = MathTraits<T>::sqrt(norm_sq);

        if (MathTraits<T>::near_zero(norm_x)) {
            tau(i) = static_cast<T>(0);
            continue;
        }

        T sign = (mat(static_cast<int>(i), static_cast<int>(i)) >= static_cast<T>(0)) ? static_cast<T>(1.0) : static_cast<T>(-1.0);
        T v0 = mat(static_cast<int>(i), static_cast<int>(i)) + sign * norm_x;

        for (size_t k = i + 1; k < Rows; ++k) mat(static_cast<int>(k), static_cast<int>(i)) /= v0;

        T v_sq_norm = static_cast<T>(1.0);
        for (size_t k = i + 1; k < Rows; ++k) {
            v_sq_norm += mat(static_cast<int>(k), static_cast<int>(i)) * mat(static_cast<int>(k), static_cast<int>(i));
        }
        tau(i) = static_cast<T>(2.0) / v_sq_norm;

        mat(static_cast<int>(i), static_cast<int>(i)) = -sign * norm_x;

        for (size_t j = i + 1; j < Cols; ++j) {
            T dot = mat(static_cast<int>(i), static_cast<int>(j));
            for (size_t k = i + 1; k < Rows; ++k) dot += mat(static_cast<int>(k), static_cast<int>(i)) * mat(static_cast<int>(k), static_cast<int>(j));
            T tau_dot = tau(i) * dot;

            mat(static_cast<int>(i), static_cast<int>(j)) -= tau_dot;
            for (size_t k = i + 1; k < Rows; ++k) mat(static_cast<int>(k), static_cast<int>(j)) -= tau_dot * mat(static_cast<int>(k), static_cast<int>(i));
        }
    }
    return true;
}

template <typename T, size_t Rows, size_t Cols>
StaticVector<T, Cols> QR_solve_Householder(const StaticMatrix<T, Rows, Cols>& mat, const StaticVector<T, Cols>& tau, const StaticVector<T, Rows>& b) {
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

// ======================================================
// NMPC 구조적 해제 전용 연산 (Free functions)
// ======================================================
template <typename T, size_t Rows, size_t Cols, size_t P_Dim>
StaticMatrix<T, Cols, Cols> quadratic_multiply(const StaticMatrix<T, Rows, Cols>& A, const StaticMatrix<T, P_Dim, P_Dim>& P) {
    StaticMatrix<T, P_Dim, Cols> Temp = P * A;
    return transpose(A) * Temp;
}

template <typename T, size_t Rows, size_t Cols, size_t B_Cols>
StaticMatrix<T, Rows, B_Cols> solve_multiple(const StaticMatrix<T, Rows, Cols>& mat, const StaticMatrix<T, Rows, B_Cols>& B) {
    static_assert(Rows == Cols, "Solver requires a square matrix");
    StaticMatrix<T, Rows, B_Cols> X;

    for (size_t j = 0; j < B_Cols; ++j) {
        StaticVector<T, Rows> b_col;
        for (size_t i = 0; i < Rows; ++i)
            b_col(i) = B(static_cast<int>(i), static_cast<int>(j));

        // 미리 분해되었다고 가정한 후 솔버 직접 호출
        StaticVector<T, Rows> x_col = LDLT_solve(mat, b_col);

        for (size_t i = 0; i < Rows; ++i)
            X(static_cast<int>(i), static_cast<int>(j)) = x_col(i);
    }
    return X;
}

}  // namespace linalg

}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_HPP_