#ifndef OPTIMIZATION_DUAL_HPP_
#define OPTIMIZATION_DUAL_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

namespace Optimization {

/**
 * @brief 전진 모드 자동 미분(Forward-mode Automatic Differentiation)을 위한 스칼라 듀얼(Dual) 수
 * 구조체
 * @details
 * Dual Number(이원수)는 f(x + ε) = f(x) + f'(x)ε (ε^2 = 0) 성질을 이용해
 * 값(Value)과 함께 그 미분값(Derivative)을 동시에 계산하는 자료구조입니다.
 * 이를 통해 함수의 기울기를 수치적 오차(Truncation/Round-off Error) 없이 정확하게 구할 수 있습니다.
 *
 * @tparam T 내부 값 및 미분값을 저장할 데이터 타입 (일반적으로 double)
 */
// =====================================================================
// 1. Scalar Dual (1D Auto Differentiation)
// =====================================================================
template <typename T>
struct Dual {
    T v;  ///< 원본 함수의 평가된 값 (Value)
    T d;  ///< 값에 대한 1계 미분값 (Derivative)

    /**
     * @brief 스칼라 값과 미분값으로 Dual Number를 초기화합니다.
     * @param value 초기 값 (기본값: 0)
     * @param deriv 초기 미분값 (상수일 경우 0, 독립 변수일 경우 1로 설정)
     */
    constexpr Dual(const T& value = T(0), const T& deriv = T(0)) noexcept : v(value), d(deriv) {}

    // --- 단항 및 이항 연산자 오버로딩 ---

    /// @brief 단항 음수 연산자 (f(x) = -x -> f'(x) = -x')
    [[nodiscard]] constexpr Dual operator-() const noexcept { return {-v, -d}; }

    /// @brief 덧셈 연산자 (f + g)' = f' + g'
    [[nodiscard]] constexpr Dual operator+(const Dual& rhs) const noexcept {
        return {v + rhs.v, d + rhs.d};
    }

    /// @brief 뺄셈 연산자 (f - g)' = f' - g'
    [[nodiscard]] constexpr Dual operator-(const Dual& rhs) const noexcept {
        return {v - rhs.v, d - rhs.d};
    }

    /// @brief 곱셈 연산자 (f * g)' = f'g + fg' (곱의 미분법)
    [[nodiscard]] constexpr Dual operator*(const Dual& rhs) const noexcept {
        return {v * rhs.v, (d * rhs.v + v * rhs.d)};
    }

    /// @brief 나눗셈 연산자 (f / g)' = (f'g - fg') / g^2 (몫의 미분법)
    [[nodiscard]] constexpr Dual operator/(const Dual& rhs) const noexcept {
        T den = rhs.v * rhs.v;
        return {v / rhs.v, (d * rhs.v - v * rhs.d) / den};
    }

    // --- 우측 스칼라(상수) 연산 (Dual * Scalar 등) ---
    // 상수의 미분은 0이므로 값(Value)만 연산되고 미분값에는 적절히 상수가 곱/나눠집니다.
    [[nodiscard]] constexpr Dual operator+(const T& rhs) const noexcept { return {v + rhs, d}; }
    [[nodiscard]] constexpr Dual operator-(const T& rhs) const noexcept { return {v - rhs, d}; }
    [[nodiscard]] constexpr Dual operator*(const T& rhs) const noexcept {
        return {v * rhs, d * rhs};
    }
    [[nodiscard]] constexpr Dual operator/(const T& rhs) const noexcept {
        return {v / rhs, d / rhs};
    }

    // --- 제자리(In-place) 연산자 ---
    // 객체를 새로 생성(메모리 재할당)하지 않고 현재 객체에 바로 연산 결과를 반영하여 성능을
    // 최적화합니다.
    constexpr Dual& operator+=(const Dual& rhs) noexcept {
        v += rhs.v;
        d += rhs.d;
        return *this;
    }
    constexpr Dual& operator-=(const Dual& rhs) noexcept {
        v -= rhs.v;
        d -= rhs.d;
        return *this;
    }
    constexpr Dual& operator*=(const Dual& rhs) noexcept {
        d = d * rhs.v + v * rhs.d;  // 미분값 먼저 갱신
        v *= rhs.v;
        return *this;
    }
    constexpr Dual& operator/=(const Dual& rhs) noexcept {
        d = (d * rhs.v - v * rhs.d) / (rhs.v * rhs.v);  // 미분값 먼저 갱신
        v /= rhs.v;
        return *this;
    }

    constexpr Dual& operator+=(const T& rhs) noexcept {
        v += rhs;
        return *this;
    }
    constexpr Dual& operator-=(const T& rhs) noexcept {
        v -= rhs;
        return *this;
    }
    constexpr Dual& operator*=(const T& rhs) noexcept {
        v *= rhs;
        d *= rhs;
        return *this;
    }
    constexpr Dual& operator/=(const T& rhs) noexcept {
        v /= rhs;
        d /= rhs;
        return *this;
    }
};

// --- 좌측 스칼라(상수) 연산 (Scalar + Dual 등) ---
template <typename T>
[[nodiscard]] constexpr Dual<T> operator+(const T& lhs, const Dual<T>& rhs) noexcept {
    return {lhs + rhs.v, rhs.d};  // 상수 + 함수
}

template <typename T>
[[nodiscard]] constexpr Dual<T> operator-(const T& lhs, const Dual<T>& rhs) noexcept {
    return {lhs - rhs.v, -rhs.d};  // 상수 - 함수
}

template <typename T>
[[nodiscard]] constexpr Dual<T> operator*(const T& lhs, const Dual<T>& rhs) noexcept {
    return {lhs * rhs.v, lhs * rhs.d};  // 상수 * 함수
}

template <typename T>
[[nodiscard]] constexpr Dual<T> operator/(const T& lhs, const Dual<T>& rhs) noexcept {
    T den = rhs.v * rhs.v;
    return {lhs / rhs.v, (-lhs * rhs.d) / den};  // 상수 / 함수: (0*g - c*g')/g^2
}

/**
 * @brief N차원 다변수 함수 자동 미분을 위한 벡터 듀얼 구조체 (Gradient Vector 계산)
 * @details
 * 다변수 함수 f(x1, x2, ..., xn)에 대하여 함수 값과 N차원 기울기 벡터(Gradient)를 동시에
 * 추적합니다. 스택 할당(std::array)을 사용하여 힙 메모리 할당으로 인한 오버헤드가 없습니다.
 *
 * @tparam T 데이터 타입
 * @tparam N 변수의 차원 (개수)
 */
// =====================================================================
// 2. Vector Dual (N-dimension Auto Differentiation)
// =====================================================================
template <typename T, size_t N>
struct DualVec {
    T v;                 ///< 평가된 함수 값 (Value)
    std::array<T, N> g;  ///< 각 변수에 대한 편미분값을 저장하는 기울기(Gradient) 벡터

    /**
     * @brief 초기값을 지정하여 DualVec을 생성합니다. Gradient는 모두 0으로 초기화됩니다.
     */
    constexpr DualVec(const T& value = T(0)) noexcept : v(value), g{} {}

    /**
     * @brief 독립 변수 초기화 함수
     * @details 특정 차원(index)의 독립 변수임을 나타내기 위해 해당 차원의 미분값만 1로 설정합니다.
     * @param value 해당 변수의 평가 위치(값)
     * @param index 해당 변수의 차원 인덱스
     * @return 초기화된 독립 변수 DualVec 객체
     */
    [[nodiscard]] static constexpr DualVec make_variable(T value, size_t index) noexcept {
        DualVec res(value);
        if (index < N) {
            res.g[index] = T(1);  // 자기 자신에 대한 편미분은 1
        }
        return res;
    }

    /// @brief 단항 음수 연산자 (모든 기울기도 음수화)
    [[nodiscard]] constexpr DualVec operator-() const noexcept {
        DualVec res;
        res.v = -v;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = -g[i];
        }
        return res;
    }

    /// @brief 덧셈 연산자
    [[nodiscard]] constexpr DualVec operator+(const DualVec& rhs) const noexcept {
        DualVec res;
        res.v = v + rhs.v;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = g[i] + rhs.g[i];
        }
        return res;
    }

    /// @brief 뺄셈 연산자
    [[nodiscard]] constexpr DualVec operator-(const DualVec& rhs) const noexcept {
        DualVec res;
        res.v = v - rhs.v;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = g[i] - rhs.g[i];
        }
        return res;
    }

    /// @brief 곱셈 연산자 (편미분에 대해 각각 곱의 미분법 적용)
    [[nodiscard]] constexpr DualVec operator*(const DualVec& rhs) const noexcept {
        DualVec res;
        res.v = v * rhs.v;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = g[i] * rhs.v + v * rhs.g[i];
        }
        return res;
    }

    /// @brief 나눗셈 연산자 (편미분에 대해 각각 몫의 미분법 적용)
    [[nodiscard]] constexpr DualVec operator/(const DualVec& rhs) const noexcept {
        DualVec res;
        T den = rhs.v * rhs.v;
        res.v = v / rhs.v;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = (g[i] * rhs.v - v * rhs.g[i]) / den;
        }
        return res;
    }

    // --- 우측 스칼라 (DualVec + Scalar 등) ---
    [[nodiscard]] constexpr DualVec operator+(const T& rhs) const noexcept {
        return *this + DualVec(rhs);  // 상수는 기울기가 0인 DualVec으로 취급
    }
    [[nodiscard]] constexpr DualVec operator-(const T& rhs) const noexcept {
        return *this - DualVec(rhs);
    }
    [[nodiscard]] constexpr DualVec operator*(const T& rhs) const noexcept {
        DualVec res;
        res.v = v * rhs;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = g[i] * rhs;
        }
        return res;
    }
    [[nodiscard]] constexpr DualVec operator/(const T& rhs) const noexcept {
        DualVec res;
        res.v = v / rhs;
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = g[i] / rhs;
        }
        return res;
    }

    // --- 제자리(In-place) 연산자 ---
    // 복사 오버헤드를 없애고 루프 성능을 극대화합니다.
    constexpr DualVec& operator+=(const DualVec& rhs) noexcept {
        v += rhs.v;
        for (size_t i = 0; i < N; ++i) {
            g[i] += rhs.g[i];
        }
        return *this;
    }
    constexpr DualVec& operator-=(const DualVec& rhs) noexcept {
        v -= rhs.v;
        for (size_t i = 0; i < N; ++i) {
            g[i] -= rhs.g[i];
        }
        return *this;
    }
    constexpr DualVec& operator*=(const DualVec& rhs) noexcept {
        for (size_t i = 0; i < N; ++i) {
            g[i] = g[i] * rhs.v + v * rhs.g[i];
        }
        v *= rhs.v;
        return *this;
    }
    constexpr DualVec& operator/=(const DualVec& rhs) noexcept {
        T den = rhs.v * rhs.v;
        for (size_t i = 0; i < N; ++i) {
            g[i] = (g[i] * rhs.v - v * rhs.g[i]) / den;
        }
        v /= rhs.v;
        return *this;
    }

    // Scalar In-place
    constexpr DualVec& operator+=(const T& rhs) noexcept {
        v += rhs;
        return *this;
    }
    constexpr DualVec& operator-=(const T& rhs) noexcept {
        v -= rhs;
        return *this;
    }
    constexpr DualVec& operator*=(const T& rhs) noexcept {
        v *= rhs;
        for (size_t i = 0; i < N; ++i) {
            g[i] *= rhs;
        }
        return *this;
    }
    constexpr DualVec& operator/=(const T& rhs) noexcept {
        v /= rhs;
        for (size_t i = 0; i < N; ++i) {
            g[i] /= rhs;
        }
        return *this;
    }
};

// --- 좌측 스칼라 (Scalar + DualVec 등) ---
template <typename T, size_t N>
[[nodiscard]] constexpr DualVec<T, N> operator+(const T& lhs, const DualVec<T, N>& rhs) noexcept {
    return rhs + lhs;  // 교환 법칙 성립
}
template <typename T, size_t N>
[[nodiscard]] constexpr DualVec<T, N> operator-(const T& lhs, const DualVec<T, N>& rhs) noexcept {
    DualVec<T, N> res(lhs - rhs.v);
    for (size_t i = 0; i < N; ++i) {
        res.g[i] = -rhs.g[i];
    }
    return res;
}
template <typename T, size_t N>
[[nodiscard]] constexpr DualVec<T, N> operator*(const T& lhs, const DualVec<T, N>& rhs) noexcept {
    return rhs * lhs;  // 교환 법칙 성립
}
template <typename T, size_t N>
[[nodiscard]] constexpr DualVec<T, N> operator/(const T& lhs, const DualVec<T, N>& rhs) noexcept {
    DualVec<T, N> res(lhs / rhs.v);
    T factor = -lhs / (rhs.v * rhs.v);
    for (size_t i = 0; i < N; ++i) {
        res.g[i] = factor * rhs.g[i];
    }
    return res;
}

// =====================================================================
// 3. 범용 수학 함수 (초월 함수 등에 대한 연쇄 법칙(Chain Rule) 정의)
// =====================================================================
/**
 * @namespace ad
 * @brief Automatic Differentiation(자동 미분)을 위한 수학 함수들의 오버로딩을 제공하는 네임스페이스
 * @details
 * std 네임스페이스의 함수들과의 충돌을 방지하고, 연쇄 법칙(Chain Rule)을 적용하여
 * 함수값 평가와 동시에 미분값(기울기)을 평가합니다. (예: sin(f(x))' = cos(f(x)) * f'(x))
 */
namespace ad {

/**
 * @def DUAL_MATH_OVERLOAD
 * @brief 1D Dual 및 N-D DualVec에 대해 단항 수학 함수의 오버로딩을 생성하는 매크로
 * @param funcName 오버로딩할 수학 함수 이름 (예: sin)
 * @param derivativeExpr 내부 값을 이용해 계산된 도함수 표현식 (예: std::cos(u.v))
 */
#define DUAL_MATH_OVERLOAD(funcName, derivativeExpr)        \
    template <typename T>                                   \
    inline Dual<T> funcName(const Dual<T>& u) {             \
        using std::funcName;                                \
        T res_v = funcName(u.v);                            \
        return {res_v, (derivativeExpr) * u.d};             \
    }                                                       \
    template <typename T, size_t N>                         \
    inline DualVec<T, N> funcName(const DualVec<T, N>& u) { \
        using std::funcName;                                \
        T res_v = funcName(u.v);                            \
        DualVec<T, N> res(res_v);                           \
        T factor = (derivativeExpr);                        \
        for (size_t i = 0; i < N; ++i) {                    \
            res.g[i] = factor * u.g[i];                     \
        }                                                   \
        return res;                                         \
    }

// 삼각함수, 지수/로그, 쌍곡선함수 등의 미분 규칙(Chain Rule) 정의
DUAL_MATH_OVERLOAD(sin, std::cos(u.v))                            // sin'(x) = cos(x)
DUAL_MATH_OVERLOAD(cos, -std::sin(u.v))                           // cos'(x) = -sin(x)
DUAL_MATH_OVERLOAD(exp, std::exp(u.v))                            // exp'(x) = exp(x)
DUAL_MATH_OVERLOAD(log, T(1.0) / u.v)                             // log'(x) = 1/x
DUAL_MATH_OVERLOAD(tanh, T(1.0) - std::pow(std::tanh(u.v), 2.0))  // tanh'(x) = 1 - tanh^2(x)
DUAL_MATH_OVERLOAD(tan, T(1.0) + std::pow(std::tan(u.v), 2.0))  // tan'(x) = sec^2(x) = 1 + tan^2(x)
DUAL_MATH_OVERLOAD(atan, T(1.0) / (T(1.0) + u.v * u.v))         // atan'(x) = 1 / (1 + x^2)

/**
 * @brief 제곱근(Square Root) 함수 오버로딩
 * @details 0 근처에서의 특이점(Singularity) 문제를 방지하기 위해 v <= 1e-16 일 경우 미분값을 0으로
 * 처리합니다.
 */
template <typename T>
inline Dual<T> sqrt(const Dual<T>& u) {
    using std::sqrt;
    T res_v = sqrt(u.v);
    T deriv = (u.v <= T(1e-16)) ? T(0.0) : (T(0.5) / res_v);
    return {res_v, deriv * u.d};
}
template <typename T, size_t N>
inline DualVec<T, N> sqrt(const DualVec<T, N>& u) {
    using std::sqrt;
    T res_v = sqrt(u.v);
    DualVec<T, N> res(res_v);
    T factor = (u.v <= T(1e-16)) ? T(0.0) : (T(0.5) / res_v);
    for (size_t i = 0; i < N; ++i) {
        res.g[i] = factor * u.g[i];
    }
    return res;
}

/**
 * @brief 거듭제곱(Power) 함수 오버로딩 (f(x)^n)' = n * f(x)^(n-1) * f'(x)
 */
template <typename T>
inline Dual<T> pow(const Dual<T>& u, double n) {
    using std::pow;
    T res_v = pow(u.v, n);
    T deriv = (u.v <= T(1e-16) && n < 1.0) ? T(0.0) : (n * pow(u.v, n - 1.0));
    return {res_v, deriv * u.d};
}

template <typename T, size_t N>
inline DualVec<T, N> pow(const DualVec<T, N>& u, double n) {
    using std::pow;
    T res_v = pow(u.v, n);
    DualVec<T, N> res(res_v);
    T factor = (u.v <= T(1e-16) && n < 1.0) ? T(0.0) : (n * pow(u.v, n - 1.0));
    for (size_t i = 0; i < N; ++i) {
        res.g[i] = factor * u.g[i];
    }
    return res;
}

/**
 * @brief 2변수 아크탄젠트(atan2) 함수 오버로딩
 */
template <typename T>
inline Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) {
    using std::atan2;
    T res_v = atan2(y.v, x.v);
    T den = x.v * x.v + y.v * y.v;
    if (den <= T(1e-16)) {
        return {res_v, T(0.0)};  // 원점 근처 특이점 방어
    }
    return {res_v, (x.v * y.d - y.v * x.d) / den};
}

template <typename T, size_t N>
inline DualVec<T, N> atan2(const DualVec<T, N>& y, const DualVec<T, N>& x) {
    using std::atan2;
    T res_v = atan2(y.v, x.v);
    DualVec<T, N> res(res_v);
    T den = x.v * x.v + y.v * y.v;
    if (den <= T(1e-16)) {
        return res;
    }
    for (size_t i = 0; i < N; ++i) {
        res.g[i] = (x.v * y.g[i] - y.v * x.g[i]) / den;
    }
    return res;
}

// 스칼라 타입에 대한 오버로딩 (템플릿 코드를 작성할 때 Dual과 스칼라 타입이 섞이는 모호성을 방지)
template <typename T>
inline T abs(const T& x) {
    return std::abs(x);
}
template <typename T>
inline T sqrt(const T& x) {
    return std::sqrt(x);
}

}  // namespace ad

// =====================================================================
// 4. std::complex 초월함수 해석적 확장 (Complex Step Differentiation 지원)
// =====================================================================
/**
 * @brief 복소수 스텝 미분(Complex Step Differentiation, CSD)을 위한 매크로
 * @details
 * CSD는 f(x + ih)를 테일러 전개했을 때 허수부가 f'(x)h와 매우 근사해짐을 이용합니다.
 * 즉, Im(f(x + ih)) / h ≈ f'(x) 가 되며, 아주 작은 h (예: 1e-100)를 사용하더라도
 * 유효숫자 상실(Subtractive Cancellation) 오류가 발생하지 않는 수치 미분 기법입니다.
 * 이 매크로는 std::complex에 대해 Chain Rule을 적용하여 CSD가 초월 함수를 안전하게 통과하도록
 * 확장합니다.
 */
#define CSD_MATH_OVERLOAD(funcName, derivativeExpr)                \
    template <typename T>                                          \
    inline std::complex<T> funcName(const std::complex<T>& u) {    \
        using std::funcName;                                       \
        T r = u.real();                                            \
        T i = u.imag();                                            \
        return std::complex<T>(funcName(r), i * (derivativeExpr)); \
    }
CSD_MATH_OVERLOAD(sin, std::cos(r))
CSD_MATH_OVERLOAD(cos, -std::sin(r))
CSD_MATH_OVERLOAD(tan, T(1.0) + std::pow(std::tan(r), 2.0))
CSD_MATH_OVERLOAD(atan, T(1.0) / (T(1.0) + r * r))
CSD_MATH_OVERLOAD(exp, std::exp(r))
CSD_MATH_OVERLOAD(log, T(1.0) / r)
CSD_MATH_OVERLOAD(tanh, T(1.0) - std::pow(std::tanh(r), 2.0))

/**
 * @brief 복소수 거듭제곱 확장 (CSD)
 */
template <typename T>
inline std::complex<T> pow(const std::complex<T>& u, double n) {
    using std::pow;
    T r = u.real(), i = u.imag();
    T res_v = pow(r, n);
    T deriv = (r <= T(1e-16) && n < 1.0) ? T(1.0) : (n * pow(r, n - 1.0));
    return std::complex<T>(res_v, i * deriv);
}

/**
 * @brief 복소수 atan2 확장 (CSD)
 */
template <typename T>
inline std::complex<T> atan2(const std::complex<T>& y, const std::complex<T>& x) {
    T yr = y.real(), yi = y.imag();
    T xr = x.real(), xi = x.imag();
    T den = xr * xr + yr * yr;

    if (den <= T(1e-16)) {
        return std::complex<T>(std::atan2(yr, xr), T(0.0));
    }
    T real_part = std::atan2(yr, xr);
    T imag_part = std::atan2(xr * yi - yr * xi) / den;  // CSD 확장 규칙 적용

    return std::complex<T>(real_part, imag_part);
}

// =====================================================================
// 5. Helper Function : 값(Value) 추출 유틸리티
// =====================================================================

/// @brief 스칼라 타입에서 값을 추출 (단순 반환)
template <typename T>
inline auto get_value(const T& x) {
    return x;
}

/// @brief 1D Dual 타입에서 값(Value)만 추출 (미분값 무시)
template <typename T>
inline auto get_value(const Dual<T>& x) {
    return x.v;
}

/// @brief N-D DualVec 타입에서 값(Value)만 추출 (기울기 벡터 무시)
template <typename T, size_t N>
inline auto get_value(const DualVec<T, N>& x) {
    return x.v;
}

}  // namespace Optimization

#endif  // OPTIMIZATION_DUAL_HPP_