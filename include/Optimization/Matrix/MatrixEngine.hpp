#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#include <algorithm>  // std::copy, std::fill (연속 메모리 블록 단위의 고속 복사 및 초기화 기능 제공)
#include <cassert>  // 런타임 인덱스 경계 검사 (디버깅 용도로 쓰이며, 릴리즈 시 NDEBUG 매크로로 오버헤드 없이 비활성화 가능)
#include <cmath>    // std::abs, std::sqrt (기본 스칼라 수학 연산)
#include <iomanip>  // 콘솔 출력 시 열을 맞추기 위한 포맷팅 조절용 (디버깅 목적)
#include <iostream>  // 행렬 상태를 확인하기 위한 디버깅 콘솔 입출력
#include <limits>  // std::numeric_limits (템플릿 타입 T에 따른 머신 입실론(Machine Epsilon)을 동적으로 가져오기 위해 필요)
#include <stdexcept>  // std::invalid_argument (0으로 나누기, 특이 행렬 등 치명적인 수치적 오류가 발생했을 때 예외를 던지기 위함)
#include <type_traits>  // std::is_floating_point (향후 AD Traits나 템플릿 메타 프로그래밍 확장 시 타입 검사를 위해 대비)

#include "Optimization/Dual.hpp"  // Auto Differentiation(자동 미분)을 위한 Dual Number(이원수) 구조체 포함

/**
 * @brief 전방 선언 (Forward Declaration)
 *
 * 컴파일러에게 `StaticMatrix` 템플릿 클래스의 존재를 미리 알려줍니다.
 * 이를 통해 클래스 본문이 정의되기 전에도, 자기 자신을 참조하는 멤버 함수나
 * `StaticVector`와 같은 연관된 별칭(Alias) 타입에서 해당 이름을 자유롭게 사용할 수 있습니다.
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix;

/**
 * @brief Alias Template (상태 벡터를 위한 별칭)
 *
 * 선형 대수학에서 수학적 벡터(Column Vector, 열 벡터)는 본질적으로 N행 1열(N x 1) 형태의
 * 행렬입니다. 따라서 벡터를 위한 별도의 클래스를 구현하지 않고, `StaticMatrix<T, N, 1>`을
 * `StaticVector<T, N>`으로 이름 붙여(Alias) 재사용합니다.
 * 이를 통해 메모리 구조상 행렬과 완벽히 동일하게 동작하며, 행렬-벡터 연산 간의 코드 일관성을 100%
 * 유지합니다.
 */
template <typename T, size_t N>
using StaticVector = StaticMatrix<T, N, 1>;

// ============================================================
// AD(Auto Differentiation, 자동 미분) 호환 Traits 구조체
//
// [배경 설명]
//   최적화 솔버(Layer 4)가 Jacobian 행렬이나 Hessian 행렬을 추출하는 과정에서,
//   행렬의 원소 자료형(Type) `T`는 단순 실수형(`double`)일 수도 있고,
//   미분값(Gradient)을 함께 추적하는 `Optimization::Dual` 타입일 수도 있습니다.
//   C++ 제네릭 프로그래밍 환경에서는 이 두 가지 완전히 다른 타입에 대해
//   단일화된 수학 연산 인터페이스를 제공해야 컴파일 오류가 발생하지 않습니다.
//   이를 위해 `MathTraits`라는 템플릿 구조체를 도입하여 타입별 연산 방식을 분기 처리합니다.
// ============================================================
template <typename T>
struct MathTraits {
    // 일반 스칼라 타입(예: double, float)에 대한 수학 함수 매핑
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }

    /**
     * @brief Epsilon 특이성 검사 (기본 스칼라 타입용)
     *
     * [수치적 안정성 판단]
     * 선형 대수 분해(LU, Cholesky, LDLT 등) 과정에서 피벗(Pivot, 대각 원소)으로 나누는 연산이
     * 빈번합니다. 이때 피벗이 수학적 0은 아니더라도 0에 극도로 가까운 수치적 노이즈라면(특이 행렬,
     * Singular), 나눗셈 결과가 무한대(Inf)로 발산하여 시스템이 붕괴합니다. 따라서
     * `std::numeric_limits<T>::epsilon()`을 사용하여 하드웨어 부동소수점 아키텍처에 맞는 최소 허용
     * 오차 범위 내에 값이 존재하는지(사실상 0인지) 안전하게 판단합니다.
     */
    static bool near_zero(const T& x) { return std::abs(x) <= std::numeric_limits<T>::epsilon(); }
};

// Optimization::Dual 타입에 대한 템플릿 특수화(Specialization)
template <typename T>
struct MathTraits<Optimization::Dual<T>> {
    // Dual 타입은 내부에 실수부(v)와 미분부(d)를 모두 가지므로,
    // Dual 헤더에 오버로딩된 특수 수학 함수를 명시적으로 호출합니다.
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
     * 분해 과정에서 행렬이 특이행렬(Singular Matrix)인지 판별하는 기준은, 행렬 원소의
     * '미분값(gradient/hessian)'이 아니라 오직 순수 '함숫값(Value, 실수부)'에 의해서만 결정되어야
     * 합니다. 따라서 Dual 구조체 전체의 크기를 비교하는 것이 아니라, 내부의 실수부 멤버인 `v`만
     * 추출하여 입실론과 비교합니다.
     */
    static bool near_zero(const Optimization::Dual<T>& x) {
        return std::abs(x.v) <= std::numeric_limits<T>::epsilon();
    }
};

/**
 * @brief Layer 1 & 2: Static Matrix Engine (Column-major 레이아웃 기반 정적 행렬)
 *
 * [핵심 설계 철학: Static Memory Allocation]
 * 동적 할당(heap 영역의 `new`/`malloc`, 혹은 `std::vector`)을 철저히 배제하고,
 * `std::array`와 같은 정적 메모리(Stack 영역 또는 전역 BSS 영역)를 기반으로 설계된 행렬
 * 클래스입니다. 이는 메모리 할당/해제에 따른 예측 불가능한 지연(Latency)을 제거하여 NMPC(비선형
 * 모델 예측 제어)와 같은 Hard Real-time(강성 실시간) 시스템의 데드라인을 엄격히 보장하며, 장기 구동
 * 시 발생하는 Heap 단편화(Fragmentation)를 원천 차단합니다.
 *
 * Layer 1 - 기본 선형 대수: 산술 연산자 오버로딩, J-K-I 루프 재배치를 통한 캐시 최적화 행렬 곱셈.
 * Layer 2 - 시스템 방정식 솔버: LU, Cholesky, LDLT, MGS-QR, Householder-QR 분해 알고리즘.
 *           + 블록 조립 연산 (거대한 KKT 최적성 조건 행렬과 같이 크고 희소한/블록 형태의 행렬을
 * 빠르게 조립하기 위한 기능)
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
   private:
    // ============================================================
    // [메모리 최적화: 정렬(Alignment) 및 공간 지역성(Spatial Locality)]
    // alignas(64): 최신 CPU(x86_64, ARM 등)의 L1 캐시 라인(Cache Line) 크기인 64바이트 단위로
    // 메모리 시작 주소를 정렬합니다.
    //   - 캐시 히트율(Cache Hit Rate) 극대화: 프로세서가 메모리를 읽을 때 64바이트 덩어리(청크)
    //   단위로 가져오므로, 정렬이 어긋나서 캐시 라인을 두 번 읽는 패널티(False Sharing 등)를
    //   방지합니다.
    //   - SIMD 가속 호환: AVX-512 등의 벡터화(Vectorization) 명령어는 메모리가 정렬되어 있지 않으면
    //   심각한 성능 저하나 하드웨어 예외를 발생시킵니다.
    //   - 이 배열은 Column-major(열 우선) 방식이므로, `data` 배열 전체가 메모리 상에 완벽하게
    //   일렬로 연속된(Contiguous) 하나의 1차원 블록을 형성합니다.
    // ============================================================
    alignas(64) T data[Rows * Cols]{};

   public:
    // ============================================================
    // 메모리 접근자 (Accessors - Column-major 방식)
    //
    // [Column-major(열 우선) 레이아웃의 이유]
    // Fortran, BLAS, LAPACK, Eigen 등 세계적인 표준 수치해석 라이브러리들은 모두 Column-major를
    // 사용합니다. 메모리 주소 계산 공식: `data[col * Rows + row]` 즉, 같은 열(Column)에 있는
    // 원소들이 메모리 상에 연속적으로 배치되어 있습니다. 열 단위의 연산(예: 행렬 곱셈, 블록
    // 삽입/추출) 시 공간 지역성을 극대화하여 캐시 미스(Cache Miss)를 획기적으로 줄입니다.
    // ============================================================
    T& operator()(int r, int c) {
        // 인덱스 안전성 검증.
        // 릴리즈 빌드(NDEBUG 정의 시)에서는 이 assert 구문이 완전히 컴파일에서 제외되므로
        // 오버헤드(if문 분기) 없이 최고 속도의 메모리 직접 접근이 가능해집니다.
        assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
        return data[c * Rows + r];
    }
    const T& operator()(int r, int c) const {
        assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
        return data[c * Rows + r];
    }

    // 1차원 선형 메모리 직접 접근 (1D Linear Access)
    // 행렬을 마치 1차원 긴 배열처럼 다룰 때 사용합니다. 요소별 덧셈/뺄셈 연산이나 블록 복사 시
    // 포인터 산술 연산을 위해 제공.
    T& operator()(size_t i) {
        assert(i < Rows * Cols);
        return data[i];
    }
    const T& operator()(size_t i) const {
        assert(i < Rows * Cols);
        return data[i];
    }

    // 로우 레벨 포인터 반환 (C API, std::copy, SIMD 인트린직 등과의 호환성)
    T* data_ptr() { return data; }
    const T* data_ptr() const { return data; }

    /**
     * @brief 고속 영행렬 초기화 (Zero Initialization)
     *
     * 매 제어 주기(Control Loop, ex: 10ms 마다 1회)가 시작될 때, 이전 주기의 KKT 행렬
     * 데이터(찌꺼기)를 깨끗하게 0으로 비워야 새로운 최적화 계산을 꼬임 없이 시작할 수 있습니다.
     * 이중 루프(for i, for j)를 돌며 0을 개별 할당하는 것보다 `std::fill`을 1차원적으로 적용하면,
     * 최신 C++ 최적화 컴파일러가 이를 감지하고 단일 어셈블리 명령어인 고속 `memset`(메모리 블록
     * 채우기) 수준으로 치환(Intrinsic)하여 연산 사이클을 극한으로 단축합니다.
     */
    void set_zero() { std::fill(data, data + (Rows * Cols), static_cast<T>(0)); }

    // ============================================================
    // 기본 선형 대수 연산 (Algebraic Operators)
    //
    // 두 행렬의 덧셈(+), 뺄셈(-) 및 스칼라배(*, /)는 각 성분끼리 매칭되는
    // 독립적인 요소별(Element-wise) 연산입니다.
    // 따라서 2차원(행, 열) 인덱스를 계산하며 순회할 필요 없이,
    // 1차원 선형 배열(data)을 0번 인덱스부터 끝까지 한 번에 쭉 순회(Linear Scan)하여 루프
    // 오버헤드를 최소화합니다.
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
        // AD(자동 미분) 호환 Epsilon 검사를 통해 나누려는 스칼라 값이 0에 너무 가깝지는 않은지
        // 확인합니다. 0으로 나누기(Divide by Zero) 시도 시 프로그램이 즉각 종료(Crash)되는 것을
        // 막고 예외 처리(Exception)로 유도.
        if (MathTraits<T>::near_zero(scalar)) {
            throw std::invalid_argument("Division by near-zero scalar in StaticMatrix::operator/");
        }
        StaticMatrix<T, Rows, Cols> Result;
        const size_t total = Rows * Cols;

        // [연산 최적화 팁]
        // CPU 명령어 사이클 측면에서 부동소수점 나눗셈(FDIV)은 곱셈(FMUL)에 비해 약 10~20배 정도
        // 압도적으로 느립니다. 루프 내에서 N*M 번 나누는 대신, 먼저 스칼라의 역수(Inverse)를 단
        // 1번만 구하고, 루프 안에서는 역수를 곱해주는 방식으로 변환하여 엄청난 속도 이득을
        // 얻습니다.
        T inv_scalar = static_cast<T>(1.0) / scalar;
        for (size_t i = 0; i < total; ++i) {
            Result.data[i] = this->data[i] * inv_scalar;
        }
        return Result;
    }

    /**
     * @brief 행렬 곱셈 연산자 오버로딩 (J-K-I 루프 캐시 최적화 적용)
     *
     * [이론적 배경 및 최적화 원리]
     * 일반적인 학부 수준의 행렬 곱셈(C = A * B) 루프 순서는 행(i) -> 열(j) -> 내적(k) 즉, "I-J-K"
     * 순서입니다. 그러나 이 클래스는 Column-major(열 우선) 메모리 레이아웃을 취하고 있습니다. I-J-K
     * 순서로 연산할 경우, 행렬 B의 요소는 열(j)을 따라가므로 메모리를 연속적으로 예쁘게
     * 읽지만(Stride 1), 행렬 A의 요소는 같은 행(i)의 다음 열(k) 요소로 넘어가기 위해 메모리 주소를
     * 매번 건너뛰면서(Stride가 Rows 크기) 읽게 됩니다. 이는 CPU 캐시에 담겨있지 않은 엉뚱한 주소를
     * 계속 요구하므로 막대한 **캐시 미스(Cache Miss)** 패널티를 유발하여 성능을 나락으로
     * 떨어뜨립니다.
     *
     * 이를 완벽히 방지하기 위해 루프의 순서를 "J-K-I" 순서로 재배치(Loop Reordering)했습니다.
     * 이 J-K-I 구조에서는 가장 빈번하게 도는 최하단(가장 안쪽) I-루프가
     * Result 행렬의 특정 열(res_col)과, 행렬 A의 특정 열(a_col)을 위에서 아래로(행 i를 증가시키며)
     * 순차적으로 쭉 훑고 지나갑니다. 이 형태는 메모리에 완전히 일렬로 순차 접근(Sequential
     * Access)하는 형태이므로 캐시 적중률이 100%에 가까워지며, 컴파일러가 SIMD 벡터화(AVX, SSE)
     * 인스트럭션을 적용하여 루프를 병렬로 묶어 처리하기 가장 이상적인 형태가 됩니다.
     *
     * @tparam OtherCols 곱해질 상대방 행렬 B의 열(Column) 개수
     * @param Other 행렬 연산의 우항(B)이 되는 행렬
     * @return 계산된 결과 행렬 C (Rows x OtherCols 차원)
     */
    template <size_t OtherCols>
    StaticMatrix<T, Rows, OtherCols> operator*(
        const StaticMatrix<T, Cols, OtherCols>& Other) const {
        StaticMatrix<T, Rows, OtherCols> Result;
        T* res_base = Result.data_ptr();
        const T* a_base = this->data;

        // J 루프: 행렬 B(Other)의 열(Column)을 이동
        for (size_t j = 0; j < OtherCols; ++j) {
            T* res_col =
                res_base +
                (j * Rows);  // Result 행렬의 j번째 열이 시작되는 메모리 주소를 캐싱 (포인터 산술)

            // K 루프: 행렬 A의 열(Column) 이동 및 행렬 B의 행(Row) 이동 (내적의 덧셈 항)
            for (size_t k = 0; k < Cols; ++k) {
                const T* a_col = a_base + (k * Rows);  // 행렬 A의 k번째 열 시작 주소 캐싱
                const T b_kj = Other(static_cast<int>(k),
                                     static_cast<int>(j));  // I 루프 동안 곱해질 B의 스칼라 값 고정

                // I 루프 (가장 안쪽): Result의 열과 A의 열을 아래 방향으로 순차적으로 연속 접근
                // 최적화 컴파일러는 이 루프를 감지하고 레지스터 FMA(Fused Multiply-Add) 명령어로
                // 치환합니다.
                for (size_t i = 0; i < Rows; ++i) {
                    res_col[i] += a_col[i] * b_kj;
                }
            }
        }
        return Result;
    }

    // ============================================================
    // 행렬 전치 (Matrix Transpose)
    // ============================================================
    StaticMatrix<T, Cols, Rows> transpose() const {
        StaticMatrix<T, Cols, Rows> Result;
        // 원본 행렬의 열(j)을 순차적(Sequential)으로 쭉 읽어서,
        // 대상 행렬의 행(j) 위치에 여기저기 흩뿌려(Scatter) 할당하는 방식입니다.
        // 읽는 쪽이라도 캐시 연속성을 확보하기 위해 외부 루프를 j로 잡습니다.
        for (size_t j = 0; j < Cols; ++j) {
            const T* src_col = this->data_ptr() + (j * Rows);
            for (size_t i = 0; i < Rows; ++i) {
                // 대상 Result 행렬은 (j, i) 순으로 접근되어 메모리 점프가 발생하지만, 전치의 특성상
                // 불가피합니다.
                Result(static_cast<int>(j), static_cast<int>(i)) = src_col[i];
            }
        }
        return Result;
    }

    // ============================================================
    // 선형 시스템 분해 솔버 (Decompositions & Solvers)
    //
    // [솔버 설계의 필요성]
    // 모델 예측 제어(MPC)의 핵심은 매 주기마다 거대한 KKT 형태의 선형 연립 방정식 시스템(Ax = b)을
    // 푸는 것입니다. 수학적으로 x = A^{-1} * b 이지만, 컴퓨터로 직접 A의 역행렬(Inverse)을 통째로
    // 구하는 것은 O(N^3)의 막대한 연산량이 소모되며 무엇보다 부동소수점 오차가 기하급수적으로
    // 누적되어 시스템이 불안정해집니다(수치적 불안정성). 따라서 현대 수치해석에서는 반드시 행렬 A를
    // 다루기 쉬운 두세 개의 삼각 행렬 조각으로 "분해(Decomposition)"한 뒤, 전진 대입(Forward
    // Substitution)과 후진 대입(Backward Substitution)이라는 아주 싼 연산(O(N^2))을 통해 빠르고
    // 정확하게 해(x)를 구합니다.
    // ============================================================

    /**
     * @brief LU 분해 (In-place Doolittle 알고리즘)
     *
     * 모든 정방 행렬(Square Matrix) A를 단위 하삼각행렬 L(Lower)과 상삼각행렬 U(Upper)의 곱 (A = L
     * * U)으로 쪼갭니다. Doolittle 알고리즘을 사용하며, 메모리를 아끼기 위해(제한된 임베디드 자원
     * 고려) 추가적인 L, U 행렬 공간을 만들지 않고 원본 행렬 A의 메모리 공간에 분해된 결과값들을
     * 그대로 덮어씁니다(In-place 방식). (참고: L 행렬의 대각 성분(Pivot)은 무조건 1.0이므로 저장
     * 공간을 생략할 수 있습니다)
     *
     * @return true  - 분해가 완벽히 성공함.
     * @return false - 계산 도중 특정 대각 원소(Pivot)가 0에 너무 가까워 나눗셈을 할 수 없음 (행렬이
     * 특이성(Singularity)을 가짐을 의미).
     */
    bool LU_decompose() {
        // 정방 행렬이 아니면 컴파일 자체를 거부합니다.
        static_assert(Rows == Cols, "LU Decomposition requires a square matrix");

        for (size_t i = 0; i < Rows; ++i) {
            // 1. U 행렬 부분 갱신 (i번째 행을 따라가며 j열 원소들 계산, 상삼각 영역 덮어쓰기)
            for (size_t j = i; j < Cols; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) -= sum;
            }

            // 피벗(Pivot) 안전성 검사. 방금 계산된 U의 대각 성분 U_ii가 피벗이 됩니다.
            // 이 값이 0이면 하삼각행렬 L을 구할 때 분모가 0이 되어 무한대(NaN/Inf) 폭발이
            // 발생합니다.
            if (MathTraits<T>::near_zero((*this)(static_cast<int>(i), static_cast<int>(i)))) {
                return false;  // 특이 행렬 판정으로 즉시 분해 실패 반환
            }

            // 2. L 행렬 부분 갱신 (i번째 열을 따라가며 j행 원소들 계산, 하삼각 영역 덮어쓰기)
            for (size_t j = i + 1; j < Rows; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(static_cast<int>(j), static_cast<int>(k)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
                }
                // 미리 구해둔 피벗 U_ii 로 나누어 스케일링
                (*this)(static_cast<int>(j), static_cast<int>(i)) =
                    ((*this)(static_cast<int>(j), static_cast<int>(i)) - sum) /
                    (*this)(static_cast<int>(i), static_cast<int>(i));
            }
        }
        return true;
    }

    /**
     * @brief LU 분해의 해법 도출 (Forward + Backward Substitution)
     *
     * 위에서 LU_decompose()를 성공적으로 마친 상태(A 공간에 L과 U가 겹쳐 저장됨)에서, 시스템 방정식
     * Ax = b (즉, LUx = b) 의 해 x를 구합니다.
     *
     * [2단계 해법 알고리즘]
     * 1단계: L(하삼각행렬) * y = b 를 풉니다. 위에서부터 아래로 미지수를 순차적으로 찾기 쉬우므로
     * 전진 대입(Forward Substitution)이라 부릅니다. 2단계: U(상삼각행렬) * x = y 를 풉니다.
     * 밑에서부터 위로 거꾸로 올라가며 미지수를 찾아 해 x를 확정하므로 후진 대입(Backward
     * Substitution)이라 부릅니다.
     *
     * @param b 우항(Right-hand side) 열 벡터
     * @return 계산된 해 벡터(x)
     */
    StaticVector<T, Rows> LU_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> y, x;

        // 1단계: 전진 대입 (Forward substitution: Ly = b)
        // 위(i=0)에서 아래 방향으로 내려감
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t j = 0; j < i; ++j) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(j)) *
                       y(j);  // 하삼각 영역(L) 접근
            }
            // Doolittle LU에서 L의 대각 원소는 암묵적으로 항상 1.0 이므로, 나눗셈 처리가 아예
            // 생략되어 연산이 매우 빠릅니다.
            y(i) = b(i) - sum;
        }

        // 2단계: 후진 대입 (Back substitution: Ux = y)
        // 행렬의 가장 아래 행(맨 밑)부터 위 방향으로 거슬러 올라감
        for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
            T sum = static_cast<T>(0);
            for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
                sum += (*this)(i, static_cast<int>(j)) * x(j);  // 상삼각 영역(U) 접근
            }
            // U의 대각 성분은 1이 아니므로, 방금 구한 y값에서 sum을 빼고 U의 대각 원소로 나누어
            // 최종 x를 구합니다.
            x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(i, i);
        }
        return x;
    }

    /**
     * @brief 촐레스키 분해 (Cholesky Decomposition, A = L * L^T)
     *
     * [제약 조건 및 강점]
     * 대상 행렬 A가 반드시 대칭 행렬(Symmetric, A = A^T)이고, 동시에 양의 정부호(Positive Definite,
     * 모든 고윳값이 양수)일 때만 수학적으로 성립하는 특수 분해 기법입니다. 일반 LU 분해에 비해
     * 필요한 반복 연산량이 정확히 절반(O(N^3/3))으로 줄어들며, 부동소수점 오차에 대한 수치적
     * 안정성(Numerical Stability)이 압도적으로 좋습니다. SQP 최적화나 Gauss-Newton (J^T J 구조)
     * 형태의 역학/제어 문제를 풀 때, Hessian 행렬은 기본적으로 양의 정부호 대칭성을 띠므로 이때
     * 가장 강력한 최우선 솔버로 활용됩니다. 분해 결과인 L 행렬은 하삼각 공간에 덮어써서 저장되며
     * 상삼각 영역은 버려집니다.
     *
     * @return true  - A가 양의 정부호 행렬임이 증명되어 분해가 정상 성공함.
     * @return false - A가 비양정치 행렬(Non-Positive Definite)이거나 음수 고윳값이 발생함. (수식
     * 내부에서 루트 안에 음수가 들어가려 할 때 즉각 발각되어 차단됨).
     */
    bool Cholesky_decompose() {
        static_assert(Rows == Cols, "Cholesky requires a square matrix");

        for (size_t j = 0; j < Cols; ++j) {
            T s = static_cast<T>(0);
            // 1. 현재 열 j의 대각 성분(Pivot)이 될 값의 후보를 계산하기 위한 누산
            for (size_t k = 0; k < j; ++k) {
                s += (*this)(static_cast<int>(j), static_cast<int>(k)) *
                     (*this)(static_cast<int>(j), static_cast<int>(k));
            }
            // 원래 대각 원소 값에서 누산된 제곱합을 뺌
            T d = (*this)(static_cast<int>(j), static_cast<int>(j)) - s;

            // [양정치성(Positive Definiteness) 엄격 검증]
            // 행렬이 양의 정부호라면 이 d 값은 무조건 0보다 뚜렷하게 커야 합니다.
            // 0 이하거나 입실론 이하의 수치적 찌꺼기라면, 행렬이 볼록성(Convexity)을 잃었거나
            // 특이행렬이 된 것이므로 치명적인 허수(Imaginary number) 발생을 막기 위해 무조건
            // 실패(false)를 반환하고 즉시 탈출합니다.
            if (d <= std::numeric_limits<T>::epsilon()) {
                return false;
            }
            // 양수임이 확인되었으므로 안심하고 루트(sqrt)를 씌워 L의 대각 성분 확정
            (*this)(static_cast<int>(j), static_cast<int>(j)) = MathTraits<T>::sqrt(d);

            // 2. L 행렬의 비대각 성분들(현재 대각 요소보다 아래에 있는 행들) 갱신
            for (size_t i = j + 1; i < Rows; ++i) {
                T s_ij = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    s_ij += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                            (*this)(static_cast<int>(j), static_cast<int>(k));
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) =
                    ((*this)(static_cast<int>(i), static_cast<int>(j)) - s_ij) /
                    (*this)(static_cast<int>(j), static_cast<int>(j));  // 확정된 L_jj로 나누기
            }
        }
        return true;
    }

    /**
     * @brief 촐레스키 분해 해법 (L * L^T * x = b)
     *
     * LU 해법과 동일하게 전진-후진 대입을 사용하나, 상삼각행렬 U 대신 L의 전치행렬인 L^T를
     * 사용합니다.
     */
    StaticVector<T, Rows> Cholesky_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> x, y;

        // 1단계: 전진 대입 (Forward substitution: Ly = b)
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < i; ++k) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(k)) * y(k);
            }
            // 촐레스키 L의 대각 성분은 1이 아니므로 L_ii로 직접 나눠줘야 합니다.
            y(i) = (b(i) - sum) / (*this)(static_cast<int>(i), static_cast<int>(i));
        }

        // 2단계: 후진 대입 (Back substitution: L^T * x = y)
        // 주의점: 물리적인 상삼각행렬 데이터가 존재하지 않습니다.
        // 대신 하삼각행렬 L을 논리적으로 전치(Transpose)시켰다고 가정하고,
        // 메모리에 접근할 때 원본 L 행렬의 인덱스 접근 순서(행과 열)를 뒤집어서 읽어오는(
        // (*this)(k, i) ) 영리한 트릭을 사용합니다.
        for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
            T sum = static_cast<T>(0);
            for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
                // L^T의 (i, k) 원소는 L의 (k, i) 원소와 같습니다.
                sum += (*this)(static_cast<int>(k), i) * x(k);
            }
            x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(i, i);
        }
        return x;
    }

    /**
     * @brief LDLT 분해 (Square-Root-Free Cholesky, A = L * D * L^T)
     *
     * [등장 배경]
     * Cholesky 분해는 매 열마다 무거운 무리수 루트(sqrt) 연산을 강제합니다. CPU 성능이 매우 제한된
     * 구형 임베디드 마이크로컨트롤러에서는 이는 병목이 될 수 있습니다. 루트 연산을 수학적으로
     * 우회하여 제거하기 위해 대각 행렬 D를 중앙에 끼워넣는 방식으로 변형한 것이 LDLT 분해입니다.
     *
     * 이론상 행렬이 0의 고윳값을 갖는 준정부호(Positive Semi-definite) 행렬이더라도 분해가
     * 가능하도록 설계되었으나, 본 엔진은 제어 솔버(NMPC, SQP) 내부망에 통합되어 있으므로 알고리즘의
     * 붕괴를 사전에 막기 위해 D의 원소가 엄격한 양수(Positive Definite)인지 입실론 수준에서 강하게
     * 검열합니다. 하삼각 영역에 L을, 대각선 영역에 D 원소를 동시에 덮어써서 콤팩트하게 저장합니다.
     * (이때 L의 대각은 암묵적 1로 처리됨)
     *
     * @return true  - 분해 성공
     * @return false - D 성분이 0 이하 (행렬이 음정치 또는 특이행렬 결함이 발생함)
     */
    bool LDLT_decompose() {
        static_assert(Rows == Cols, "LDLT requires a square matrix");

        for (size_t j = 0; j < Cols; ++j) {
            T sum_D = static_cast<T>(0);
            for (size_t k = 0; k < j; ++k) {
                T L_jk = (*this)(static_cast<int>(j), static_cast<int>(k));
                // L의 원소 제곱에 대각 원소 D_kk를 스케일링 팩터로 곱하여 누산 (루트 대신 사용되는
                // 핵심 트릭)
                sum_D += L_jk * L_jk * (*this)(static_cast<int>(k), static_cast<int>(k));
            }

            // D 행렬의 대각 성분 D_jj 결정
            T D_jj = (*this)(static_cast<int>(j), static_cast<int>(j)) - sum_D;

            // 비특이성(Non-singularity) 및 양정치 검사 (0에 근접하거나 음수면 거부)
            if (MathTraits<T>::near_zero(D_jj)) {
                return false;
            }
            (*this)(static_cast<int>(j), static_cast<int>(j)) =
                D_jj;  // 대각 원소 자리에 D 성분 저장

            // L 행렬 비대각 성분 갱신
            for (size_t i = j + 1; i < Rows; ++i) {
                T sum_L = static_cast<T>(0);
                for (size_t k = 0; k < j; ++k) {
                    sum_L += (*this)(static_cast<int>(i), static_cast<int>(k)) *
                             (*this)(static_cast<int>(j), static_cast<int>(k)) *
                             (*this)(static_cast<int>(k),
                                     static_cast<int>(k));  // 공식 전개: L_ik * L_jk * D_kk
                }
                (*this)(static_cast<int>(i), static_cast<int>(j)) =
                    ((*this)(static_cast<int>(i), static_cast<int>(j)) - sum_L) /
                    (*this)(static_cast<int>(j),
                            static_cast<int>(j));  // 확정된 D_jj로 나누어 마무리
            }
        }
        return true;
    }

    /**
     * @brief LDLT 분해 해법 (L * D * L^T * x = b)
     *
     * 과정이 총 3단계(전진 대입 -> 대각 스케일링 조정 -> 후진 대입)로 진행됩니다.
     */
    StaticVector<T, Rows> LDLT_solve(const StaticVector<T, Rows>& b) const {
        static_assert(Rows == Cols, "Solve requires a square matrix");
        StaticVector<T, Rows> z, y, x;

        // 1단계: 전진 대입 Lz = b
        // 주의: LDLT에서 L의 대각 성분은 암묵적으로 항상 1.0으로 고정되어 있으므로,
        // 이 부분에서는 나눗셈이 발생하지 않아 연산이 쾌적합니다.
        for (size_t i = 0; i < Rows; ++i) {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < i; ++k) {
                sum += (*this)(static_cast<int>(i), static_cast<int>(k)) * z(k);
            }
            z(i) = b(i) - sum;
        }

        // 2단계: 중앙 대각 행렬(D) 스케일링: Dy = z
        // 이 한 줄의 나눗셈 과정이 Cholesky 분해의 무거운 무리수 루트(sqrt) 연산을 훌륭하게
        // 대체하는 수학적 역할을 합니다.
        for (size_t i = 0; i < Rows; ++i) {
            y(i) = z(i) / (*this)(static_cast<int>(i), static_cast<int>(i));
        }

        // 3단계: 후진 대입 L^T x = y
        // 여기서도 마찬가지로 L^T의 대각 성분은 암묵적 1이므로,
        // 뺄셈만 하고 L_ii 로 나누는 절차는 생략됩니다.
        for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
            T sum = static_cast<T>(0);
            for (size_t k = static_cast<size_t>(i) + 1; k < Rows; ++k) {
                sum += (*this)(static_cast<int>(k), i) * x(k);  // 인덱스 뒤집기 트릭
            }
            x(static_cast<size_t>(i)) = y(static_cast<size_t>(i)) - sum;
        }
        return x;
    }

    /**
     * @brief MGS-QR 분해 (Modified Gram-Schmidt 기반의 A = Q * R 직교 분해)
     *
     * 정방 행렬뿐만 아니라 행이 더 많은 직사각형 행렬(Over-determined System)도 처리할 수 있는
     * 직교(Orthogonal) 기반 분해 솔버입니다. 행렬 A의 열 벡터들을 순차적으로 정규화(Normalize)하고
     * 직교화(Orthogonalize)하여 직교 행렬 Q와 상삼각 행렬 R을 만들어냅니다.
     *
     * [MGS(Modified)의 핵심 최적화]
     * 학부 수준의 고전적인 CGS(Classical Gram-Schmidt) 알고리즘은 계산 중 부동소수점 오차가
     * 기하급수적으로 쌓여 최종 도출된 Q 벡터들이 직교성을 상실하고 한쪽으로 무너지는 끔찍한 단점이
     * 있습니다. MGS 알고리즘은 이를 방지하기 위해, 하나의 직교 Q 열을 찾아낼 때마다 그 직교성을
     * "아직 처리하지 않은 모든 나머지 열들에서" 즉각적으로 깎아내고 빼버리는(투영 소거, Projection)
     * 전략을 사용하여, 컴퓨터 수치 환경에서 훨씬 강건하고 완벽한 직교 기반을 유지합니다. Q 행렬은
     * 원본 A 공간(*this)에 덮어써지고, R 행렬은 별도의 외부 변수를 통해 반환받습니다.
     *
     * @param R (출력용) 분해된 상삼각행렬 R을 담을 외부 행렬 객체
     * @return true  - 분해 성공
     * @return false - 특정 열이 이전 열 벡터들과 선형 종속(Linear Dependent) 관계여서 직교 기반
     * 생성이 불가능할 경우
     */
    bool QR_decompose_MGS(StaticMatrix<T, Cols, Cols>& R) {
        static_assert(Rows >= Cols, "MGS-QR requires Rows >= Cols (Over-determined system)");

        for (size_t i = 0; i < Cols; ++i) {
            // 1. 현재 열 벡터 a_i 의 Norm(크기, 길이)을 내적을 통해 도출
            T norm_sq = static_cast<T>(0);
            for (size_t k = 0; k < Rows; ++k) {
                norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            // R의 대각 원소는 현재 열 벡터의 Norm 길이가 할당됨
            R(static_cast<int>(i), static_cast<int>(i)) = MathTraits<T>::sqrt(norm_sq);

            // 열 벡터의 길이가 사실상 0 이라면, 선형 종속성(Linear Dependency)으로 인해 직교
            // 기저(Orthogonal basis)를 형성할 수 없으므로 포기합니다.
            if (MathTraits<T>::near_zero(R(static_cast<int>(i), static_cast<int>(i)))) {
                return false;
            }

            // 2. 현재 열 벡터를 앞서 구한 길이 Norm으로 나누어 크기가 1인 단위 벡터(정규화) Q_i로
            // 변환 (원본 행렬 공간 덮어쓰기)
            for (size_t k = 0; k < Rows; ++k) {
                (*this)(static_cast<int>(k), static_cast<int>(i)) /=
                    R(static_cast<int>(i), static_cast<int>(i));
            }

            // 3. 투영 소거 (MGS 알고리즘의 심장부)
            // 이제 막 확정된 정규 단위 벡터 Q_i 를 기준으로, 그보다 오른쪽에 있는 모든 열(j)들에
            // 대해 Q_i 방향으로 겹쳐진 그림자 성분(성분 길이 = 내적 dot)을 계산하고, 그 그림자
            // 성분을 완전히 빼버려 독립적으로 만듭니다.
            for (size_t j = i + 1; j < Cols; ++j) {
                T dot = static_cast<T>(0);
                for (size_t k = 0; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                // 투영된 성분의 길이를 상삼각 R 행렬에 기록
                R(static_cast<int>(i), static_cast<int>(j)) = dot;

                // 나머지 열 벡터 j 에서 해당 투영 벡터 성분을 물리적으로 빼내어 직교화 완료
                for (size_t k = 0; k < Rows; ++k) {
                    (*this)(static_cast<int>(k), static_cast<int>(j)) -=
                        dot * (*this)(static_cast<int>(k), static_cast<int>(i));
                }
            }
        }
        return true;
    }

    /**
     * @brief MGS-QR 분해 기반 시스템 해법 (Ax=b -> QRx=b -> Rx = Q^T b)
     *
     * Q 행렬은 직교 행렬(Orthogonal Matrix)이므로 놀랍게도 Q의 역행렬(Q^-1)은 단순히 그 전치
     * 행렬(Q^T)과 정확히 동일합니다. 따라서 양변에 Q^T 를 곱하면 좌항은 Rx 만 깔끔하게 남게 되어
     * 초고속 역산이 가능합니다.
     */
    StaticVector<T, Cols> QR_solve(const StaticMatrix<T, Cols, Cols>& R,
                                   const StaticVector<T, Rows>& b) const {
        StaticVector<T, Cols> y, x;

        // 1단계: 행렬-벡터 곱셈 (Q^T * b 연산 수행)
        // 메모리에 저장된 *this 자체가 직교 행렬 Q입니다.
        for (size_t i = 0; i < Cols; ++i) {
            T dot = static_cast<T>(0);
            for (size_t k = 0; k < Rows; ++k) {
                dot += (*this)(static_cast<int>(k), static_cast<int>(i)) * b(k);
            }
            y(i) = dot;
        }

        // 2단계: 후진 대입 (Back substitution: R * x = y)
        // R 행렬은 외부에서 전달받은 상삼각 형태이므로 밑에서부터 채워 올라옵니다.
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
     * @brief 하우스홀더 QR 분해 (Householder Reflection QR, 극강의 메모리 압축 기법)
     *
     * [MGS vs Householder 비교 및 작동 원리]
     * MGS 알고리즘은 성능이 뛰어나지만 정밀도가 극단적으로 요구되는 제약조건 KKT 시스템(Jacobian의
     * 널스페이스(Null-space)를 도출할 때 등)에서는 간혹 직교성이 깨지기도 합니다. 반면 하우스홀더
     * QR 알고리즘은 거울처럼 빛을 반사시키는 물리적 기하학(Reflection) 개념을 이용해 벡터를 특정
     * 초평면(Hyperplane)에 접어 대칭 이동시킵니다. 이 방식은 컴퓨터 부동소수점 오차가 발생하더라도
     * 직교성이 구조상 절대 무너지지 않아 가장 완벽한 해상도를 보장합니다.
     *
     * [Compact Storage 최적화 마법]
     * 더 놀라운 점은 행렬 Q 전체를 명시적으로 N*N 크기만큼 메모리에 만들고 저장하지 않는다는
     * 것입니다. 거울의 각도 방향을 결정하는 고유한 Householder 반사 벡터 v 만을 추출하여, 원본 행렬
     * A의 쓸모없어진 하삼각 영역 빈 공간에 차곡차곡 구겨 넣어 콤팩트하게 저장(Compact
     * Storage)합니다. 이 천재적인 최적화로 인해 메모리 사용량이 극적으로 절약되며 거대한 행렬
     * 분해에 최적입니다.
     *
     * @param tau  거울 반사 스케일링 팩터(Scaling factor) 값들을 저장해 둘 별도의 추가 벡터
     * @return true (하우스홀더는 웬만한 특이 열이 나타나도 tau값을 0으로 닫아 스킵하며 알고리즘
     * 자체가 무너지지 않고 무조건 계속 굴러가는 안정성을 자랑함)
     */
    bool QR_decompose_Householder(StaticVector<T, Cols>& tau) {
        static_assert(Rows >= Cols, "Householder-QR requires Rows >= Cols");

        for (size_t i = 0; i < Cols; ++i) {
            // 현재 타겟이 되는 부분 열(서브벡터 i부터 최하단 Rows까지)의 Norm 크기를 누산.
            T norm_sq = static_cast<T>(0);
            for (size_t k = i; k < Rows; ++k) {
                norm_sq += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            T norm_x = MathTraits<T>::sqrt(norm_sq);  // 누산 후 무거운 루트 연산은 단 1회만 실행

            // 부분 벡터 전체가 완벽하게 0벡터에 가깝다면, 반사(Reflection) 액션을 수행할 필요 없이
            // 스킵.
            if (MathTraits<T>::near_zero(norm_x)) {
                tau(i) = static_cast<T>(0);
                continue;
            }

            // 부호 처리 (Sign Handling)
            // 컴퓨터 수학의 고질적 문제인 수치 소거 오류(Cancellation Error, 비슷한 크기의
            // 부동소수점을 뺄 때 유효 숫자가 뭉텅이로 날아가는 치명적 현상)를 방지하기 위한
            // 안전장치입니다. 기존 대각 원소의 부호와 완전히 반대 부호를 취하여 뺄셈이 덧셈처럼
            // 작용하도록 만듭니다.
            T sign = ((*this)(static_cast<int>(i), static_cast<int>(i)) >= static_cast<T>(0))
                         ? static_cast<T>(1.0)
                         : static_cast<T>(-1.0);
            T v0 = (*this)(static_cast<int>(i), static_cast<int>(i)) + sign * norm_x;

            // 반사 벡터 v 구축 및 Compact하게 쑤셔넣기
            // v의 첫 원소인 v0는 기준점이 되므로 스케일 정규화 용도로 나누고,
            // 나머지 성분들은 행렬의 사용되지 않는 하삼각 영역에 차곡차곡 은밀하게 보관합니다.
            for (size_t k = i + 1; k < Rows; ++k) {
                (*this)(static_cast<int>(k), static_cast<int>(i)) /= v0;
            }

            // Tau (반사 크기 조절용 스케일 팩터) 계산
            T v_sq_norm = static_cast<T>(1.0);  // 방금 나눈 v0 부분은 정규화되어 1의 제곱으로 치급
            for (size_t k = i + 1; k < Rows; ++k) {
                v_sq_norm += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                             (*this)(static_cast<int>(k), static_cast<int>(i));
            }
            tau(i) = static_cast<T>(2.0) / v_sq_norm;  // 거울 반사 방정식 계수 (2.0)

            // 분해된 상삼각 R 행렬의 대각 성분이 되는 부분을 최종 확정 지음
            (*this)(static_cast<int>(i), static_cast<int>(i)) = -sign * norm_x;

            // [Rank-1 Update 투영]
            // 방금 만들어낸 거울(Householder Reflection Vector)을 이용해
            // 현재 열보다 우측에 위치한 모든 열 벡터(A 행렬의 뒷부분)들을 전부 이 거울에 통과시켜
            // 각도를 반사시켜 접어버립니다. (A <- (I - tau*v*v^T) A)
            for (size_t j = i + 1; j < Cols; ++j) {
                T dot =
                    (*this)(static_cast<int>(i), static_cast<int>(j));  // v_0 = 1 기준 성분 초기화
                for (size_t k = i + 1; k < Rows; ++k) {
                    dot += (*this)(static_cast<int>(k), static_cast<int>(i)) *
                           (*this)(static_cast<int>(k), static_cast<int>(j));
                }
                T tau_dot = tau(i) * dot;

                // 거울에 반사된 만큼의 각도를 원본 행렬에서 즉각 갱신(수정)
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
     * @brief 하우스홀더 QR 분해 기반 해법 도출 (명시적 Q 행렬 복원 과정 없이 바로 해를 찾는 마법)
     *
     * 일반 MGS 방식처럼 커다란 행렬 Q가 메모리에 명시적으로 존재하지 않습니다.
     * 하삼각 부분에 조각조각 구겨 넣어 숨겨둔 반사 벡터(v) 조각들과 tau 스케일 배열을 차례로
     * 재조립하여, 우항 벡터 b 에 대해 도미노 쓰러뜨리듯 연속적으로 직교 거울 반사를 적용하면
     * 자연스럽게 좌항의 (Q^T * b) 연산 결과 벡터가 튀어나오게 되는 고도로 최적화된 테크닉을
     * 사용합니다.
     */
    StaticVector<T, Cols> QR_solve_Householder(const StaticVector<T, Cols>& tau,
                                               const StaticVector<T, Rows>& b) const {
        StaticVector<T, Rows> y = b;

        // 1단계: b 벡터에 대해 순차적 거울 반사(Sequential Reflection) 시뮬레이션 적용 (Q^T b 계산
        // 효과)
        for (size_t i = 0; i < Cols; ++i) {
            // 반사가 무효화(tau가 0)된 열은 패스
            if (MathTraits<T>::near_zero(tau(i))) {
                continue;
            }
            T dot = y(i);
            // 하삼각에 숨겨둔 벡터 v 성분을 재조립
            for (size_t k = i + 1; k < Rows; ++k) {
                dot += (*this)(static_cast<int>(k), static_cast<int>(i)) * y(k);
            }
            T tau_dot = tau(i) * dot;

            // 벡터 b(현재 임시변수 y)를 거울에 통과시켜 방향을 접음
            y(i) -= tau_dot;
            for (size_t k = i + 1; k < Rows; ++k) {
                y(k) -= tau_dot * (*this)(static_cast<int>(k), static_cast<int>(i));
            }
        }

        // 2단계: 후진 대입 (Back substitution: R * x = y)
        // R 행렬의 데이터는 원본 배열의 상삼각 영역(Upper Triangular)에 고스란히 남아있으므로 이를
        // 활용.
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
    // 블록 연산 모듈 (Block Operations & Assembly Utilities)
    //
    // [사용처]
    // SQP(순차 이차 계획법), 내점법(IPM) 등 복잡한 최적화 알고리즘에서는,
    // 최적성 조건을 나타내는 거대한 KKT 행렬을 매 주기마다 새롭게 퍼즐 조립하듯 끼워 맞춰 생성해야
    // 합니다. (예: 좌상단에 목적 함수 Hessian 덩어리, 좌하단 및 우상단에 제약 조건 A 덩어리 등) 이
    // 유틸리티들은 "Column-major 특성상 같은 열(Column)은 메모리 연속이다" 라는 핵심 사실을
    // 극한으로 뽑아먹도록 최적화 설계되어 있습니다.
    // ============================================================

    /**
     * @brief 부분 행렬 삽입 (std::copy를 통한 C언어 memcpy 수준의 초고속 열 블록 카피)
     *
     * 이중 for 루프(for i, for j)를 돌며 원소를 바보같이 한 땀 한 땀 찍어 복사하는 대신,
     * 한 열(Column) 덩어리 전체가 메모리에 일렬 주소로 연속해 늘어서 있다는 물리적 성질을
     * 악용합니다. C++ 표준 라이브러리인 `std::copy`를 사용하면, 이 함수가 컴파일될 때 내부적으로
     * CPU 블록 카피 명령(memcpy 등)으로 치환되어 한 번 호출 시 1개의 열 블록 전체가 압도적인 속도로
     * 단번에 타겟 영역에 복사되어 박힙니다.
     *
     * @tparam SubRows 삽입할 작은 블록 행렬의 행 크기
     * @tparam SubCols 삽입할 작은 블록 행렬의 열 크기
     * @param start_row 거대(타겟) 행렬의 기준 시작 행 인덱스
     * @param start_col 거대(타겟) 행렬의 기준 시작 열 인덱스
     * @param block 쑤셔넣을 작은 부분 블록 행렬
     */
    template <size_t SubRows, size_t SubCols>
    void insert_block(size_t start_row, size_t start_col,
                      const StaticMatrix<T, SubRows, SubCols>& block) {
        // [컴파일 타임 에러 검출] 템플릿 크기 자체가 타겟 행렬의 허용 크기를 초과하면 애초에 빌드를
        // 차단
        static_assert(SubRows <= Rows, "SubMatrix rows exceed target");
        static_assert(SubCols <= Cols, "SubMatrix cols exceed target");

        // [런타임 인덱스 경계 에러 검출] 삽입 시작 좌표가 밀려서 꼬리가 삐져나가는 현상 방지
        assert(start_row + SubRows <= Rows && "Row index out of bounds");
        assert(start_col + SubCols <= Cols && "Col index out of bounds");

        // 루프는 '열(Column)' 단위로만 돕니다. 행 단위 접근은 통째로 생략됨.
        for (size_t j = 0; j < SubCols; ++j) {
            // 원본 블록 행렬의 j번째 열 데이터 덩어리가 시작되는 첫 번째 메모리 주소(포인터) 계산
            const T* src = block.data_ptr() + (j * SubRows);
            // 삽입 당할 거대 타겟 행렬 내에서, 정확히 박힐 시작 메모리 주소(포인터) 계산
            T* dest = this->data_ptr() + ((start_col + j) * Rows) + start_row;

            // src 포인터부터 1열 크기(SubRows 개수)만큼 통째로 긁어서 dest 에 한 번에 복사
            std::copy(src, src + SubRows, dest);
        }
    }

    /**
     * @brief 전치(Transposed) 형태의 부분 행렬을 조립하여 바로 삽입
     *
     * KKT 시스템 행렬을 조립하다 보면 구조상 좌하단에는 제약 조건 행렬 A를 통째로 넣고,
     * 우상단 대칭점에는 이를 전치시킨 A^T를 집어넣어야 할 때가 빈번하게 발생합니다.
     * 이때 전치 행렬을 미리 만들어 메모리에 저장하는 것은 공간적 낭비이므로,
     * 원본 A 블록을 넘겨주면 삽입과 동시에 행과 열을 뒤틀어서 박아 넣습니다.
     * (이 경우는 메모리 구조가 뒤틀리기 때문에 std::copy 선형 복사 최적화 마법을 쓸 수 없고, 캐시
     * 흩뿌림(Scattering) 방식의 2중 루프가 불가피합니다)
     */
    template <size_t SubRows, size_t SubCols>
    void insert_transposed_block(size_t start_row, size_t start_col,
                                 const StaticMatrix<T, SubRows, SubCols>& block) {
        // 전치 삽입이므로 크기 검사 로직의 행과 열 타겟 조건이 거꾸로 반전됩니다.
        static_assert(SubCols <= Rows, "Transposed block rows exceed target");
        static_assert(SubRows <= Cols, "Transposed block cols exceed target");
        assert(start_row + SubCols <= Rows && "Row index out of bounds");
        assert(start_col + SubRows <= Cols && "Col index out of bounds");

        for (size_t j = 0; j < SubCols; ++j) {
            for (size_t i = 0; i < SubRows; ++i) {
                // 타겟(this)에는 (j, i) 순으로 좌표를 뒤집어서 박아넣어 물리적인 전치(Transpose)
                // 효과를 부여함
                (*this)(static_cast<int>(start_row + j), static_cast<int>(start_col + i)) =
                    block(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    /**
     * @brief 거대 행렬에서 특정 부분 행렬(블록)만 추출하여 복사해 오기
     *
     * KKT 시스템과 같은 초대형 복합 선형 방정식을 풀고 나면 엄청나게 긴 형태의 결과 벡터를 얻게
     * 됩니다. 이 벡터에는 상태 변수(x), 슬랙 변수(s), 듀얼 승수(lambda) 등이 잡탕으로 섞여
     * 있습니다. 제어 로직에서는 이 중에서 로봇/차량에 실질적인 지시를 내릴 '제어 입력 신호(Control
     * Input, u)' 블록만 깔끔하게 도려내야 합니다. `insert_block`과 정반대 순서로, 완벽히 동일한
     * 메모리 고속 복사(std::copy) 최적화 기법을 적용하여 덩어리째 뜯어옵니다.
     */
    template <size_t SubRows, size_t SubCols>
    StaticMatrix<T, SubRows, SubCols> extract_block(size_t start_row, size_t start_col) const {
        assert(start_row + SubRows <= Rows && "Row index out of bounds");
        assert(start_col + SubCols <= Cols && "Col index out of bounds");

        StaticMatrix<T, SubRows, SubCols> result;
        // 행은 건너뛰고 열 단위 덩어리째 접근
        for (size_t j = 0; j < SubCols; ++j) {
            // 거대 원본 행렬에서 도려낼 좌표 메모리 주소(포인터) 계산
            const T* src = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            // 새롭게 뽑아낼 결과 행렬 공간의 j번째 열 시작 메모리 주소
            T* dest = result.data_ptr() + (j * SubRows);
            // 덩어리 단위 통째 초고속 복사
            std::copy(src, src + SubRows, dest);
        }
        return result;
    }

    // ============================================================
    // NMPC 구조적 해제 전용 연산
    // ============================================================
    /**
     * @brief 샌드위치 2차 형식 곱셈 : Result = A^T * P * A
     * Riccati Backward Pass에서 H_xx, H_uu를 조립할 때 캐시 미스를 최소화하여 계산
     */
    template <size_t P_Dim>
    StaticMatrix<T, Cols, Cols> quadratic_multiply(const StaticMatrix<T, P_Dim, P_Dim>& P) const {
        // 1. Temp = P * A (P_Dim x P_Dim * P_Dim x Cols)
        StaticMatrix<T, P_Dim, Cols> Temp = P * (*this);
        // 2. Result = A^T * Temp (Cols x P_Dim * P_Dim x Cols)
        return this->transpose() * Temp;
    }

    /**
     * @brief 다중 열 (Multi-Column) 시스템 풀이 : A * X = B
     * K_k = -H_uu^{-1} * H_ux를 구할 때 역행렬을 직접 구하지 않고
     * LDLT 분해된 상태에서 B의 열 벡터들을 순차적으로 풀어냅니다
     */
    template <size_t B_Cols>
    StaticMatrix<T, Rows, B_Cols> solve_multiple(const StaticMatrix<T, Rows, B_Cols>& B) const {
        static_assert(Rows == Cols, "Solver requires a square matrix");
        StaticMatrix<T, Rows, B_Cols> X;

        // B행렬 각 열 (Column)을 추출하여 벡터로 풀고 다시 X에 조립
        for (size_t j = 0; j < B_Cols; ++j) {
            StaticVector<T, Rows> b_col;
            for (size_t i = 0; i < Rows; ++i) {
                b_col(i) = B(static_cast<int>(i), static_cast<int>(j));
            }
            // LDLT_solve를 호출하여 1개 열에 대한 해를 구함 (미리 LDLT_decompose가 호출되어 있어야
            // 함)
            StaticVector<T, Rows> x_col = this->LDLT_solve(b_col);

            for (size_t i = 0; i < Rows; ++i) {
                X(static_cast<int>(i), static_cast<int>(j)) = x_col(i);
            }
        }
        return X;
    }

    // ============================================================
    // 디버깅 및 시각화 유틸리티
    //
    // 제어 솔버가 최적해를 도출하는 데 실패하거나 차량이 튀는(Jitter) 이상 증상을 보일 때,
    // 원인이 되는 KKT 시스템 매트릭스가 제대로 조립되었는지 콘솔 화면에서 시각적으로 직접 확인하기
    // 위한 필수 함수입니다.
    // ============================================================
    void print(const char* name) const {
        // 행렬 이름, 레이아웃 정보, 차원 출력
        std::cout << "Matrix [" << name << "] (Col-major, " << Rows << "x" << Cols << "):\n";
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                // iomanip 라이브러리를 이용하여 정해진 폭(10칸)과 고정된 소수점 자리수(4자리)를
                // 맞춰 표처럼 깔끔하게 출력
                std::cout << std::fixed << std::setw(10) << std::setprecision(4)
                          << (*this)(static_cast<int>(i), static_cast<int>(j)) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
};

#endif  // STATIC_MATRIX_HPP_