#ifndef OPTIMIZATION_DUAL_HPP_
#define OPTIMIZATION_DUAL_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>

// [Architect's Update] MathTraits와의 중복 방지 및 연동
// (MathTraits.hpp에서 get_value 등을 처리하므로 Dual.hpp에서는 순수 수학 엔진만 담당)

namespace Optimization {

// =====================================================================
// 1. Scalar Dual (1D Auto Differentiation)
// =====================================================================
template <typename T>
struct Dual {
    T v;  
    T d;  

    constexpr Dual(const T& value = T(0), const T& deriv = T(0)) noexcept : v(value), d(deriv) {}

    [[nodiscard]] constexpr Dual operator-() const noexcept { return {-v, -d}; }

    [[nodiscard]] constexpr Dual operator+(const Dual& rhs) const noexcept { return {v + rhs.v, d + rhs.d}; }
    [[nodiscard]] constexpr Dual operator-(const Dual& rhs) const noexcept { return {v - rhs.v, d - rhs.d}; }
    [[nodiscard]] constexpr Dual operator*(const Dual& rhs) const noexcept { return {v * rhs.v, (d * rhs.v + v * rhs.d)}; }
    [[nodiscard]] constexpr Dual operator/(const Dual& rhs) const noexcept {
        T den = rhs.v * rhs.v;
        return {v / rhs.v, (d * rhs.v - v * rhs.d) / den};
    }

    // 우측 스칼라
    [[nodiscard]] constexpr Dual operator+(const T& rhs) const noexcept { return {v + rhs, d}; }
    [[nodiscard]] constexpr Dual operator-(const T& rhs) const noexcept { return {v - rhs, d}; }
    [[nodiscard]] constexpr Dual operator*(const T& rhs) const noexcept { return {v * rhs, d * rhs}; }
    [[nodiscard]] constexpr Dual operator/(const T& rhs) const noexcept { return {v / rhs, d / rhs}; }

    // 제자리(In-place) 연산자
    constexpr Dual& operator+=(const Dual& rhs) noexcept { v += rhs.v; d += rhs.d; return *this; }
    constexpr Dual& operator-=(const Dual& rhs) noexcept { v -= rhs.v; d -= rhs.d; return *this; }
    constexpr Dual& operator*=(const Dual& rhs) noexcept { d = d * rhs.v + v * rhs.d; v *= rhs.v; return *this; }
    constexpr Dual& operator/=(const Dual& rhs) noexcept { d = (d * rhs.v - v * rhs.d) / (rhs.v * rhs.v); v /= rhs.v; return *this; }

    constexpr Dual& operator+=(const T& rhs) noexcept { v += rhs; return *this; }
    constexpr Dual& operator-=(const T& rhs) noexcept { v -= rhs; return *this; }
    constexpr Dual& operator*=(const T& rhs) noexcept { v *= rhs; d *= rhs; return *this; }
    constexpr Dual& operator/=(const T& rhs) noexcept { v /= rhs; d /= rhs; return *this; }
};

template <typename T> [[nodiscard]] constexpr Dual<T> operator+(const T& lhs, const Dual<T>& rhs) noexcept { return {lhs + rhs.v, rhs.d}; }
template <typename T> [[nodiscard]] constexpr Dual<T> operator-(const T& lhs, const Dual<T>& rhs) noexcept { return {lhs - rhs.v, -rhs.d}; }
template <typename T> [[nodiscard]] constexpr Dual<T> operator*(const T& lhs, const Dual<T>& rhs) noexcept { return {lhs * rhs.v, lhs * rhs.d}; }
template <typename T> [[nodiscard]] constexpr Dual<T> operator/(const T& lhs, const Dual<T>& rhs) noexcept {
    T den = rhs.v * rhs.v;
    return {lhs / rhs.v, (-lhs * rhs.d) / den};
}

// =====================================================================
// 2. Vector Dual (N-dimension Auto Differentiation)
// =====================================================================
template <typename T, size_t N>
struct DualVec {
    T v;                 
    std::array<T, N> g;  

    constexpr DualVec(const T& value = T(0)) noexcept : v(value), g{} {}

    [[nodiscard]] static constexpr DualVec make_variable(T value, size_t index) noexcept {
        DualVec res(value);
        if (index < N) res.g[index] = T(1);
        return res;
    }

    [[nodiscard]] constexpr DualVec operator-() const noexcept {
        DualVec res;
        res.v = -v;
        // [Architect's Note] N이 컴파일 타임 상수이므로 컴파일러가 강제 루프 언롤링(Unrolling) 수행
        for (size_t i = 0; i < N; ++i) res.g[i] = -g[i];
        return res;
    }

    [[nodiscard]] constexpr DualVec operator+(const DualVec& rhs) const noexcept {
        DualVec res;
        res.v = v + rhs.v;
        for (size_t i = 0; i < N; ++i) res.g[i] = g[i] + rhs.g[i];
        return res;
    }

    [[nodiscard]] constexpr DualVec operator-(const DualVec& rhs) const noexcept {
        DualVec res;
        res.v = v - rhs.v;
        for (size_t i = 0; i < N; ++i) res.g[i] = g[i] - rhs.g[i];
        return res;
    }

    [[nodiscard]] constexpr DualVec operator*(const DualVec& rhs) const noexcept {
        DualVec res;
        res.v = v * rhs.v;
        for (size_t i = 0; i < N; ++i) res.g[i] = g[i] * rhs.v + v * rhs.g[i];
        return res;
    }

    [[nodiscard]] constexpr DualVec operator/(const DualVec& rhs) const noexcept {
        DualVec res;
        T den = rhs.v * rhs.v;
        res.v = v / rhs.v;
        for (size_t i = 0; i < N; ++i) res.g[i] = (g[i] * rhs.v - v * rhs.g[i]) / den;
        return res;
    }

    // 우측 스칼라
    [[nodiscard]] constexpr DualVec operator+(const T& rhs) const noexcept { return *this + DualVec(rhs); }
    [[nodiscard]] constexpr DualVec operator-(const T& rhs) const noexcept { return *this - DualVec(rhs); }
    [[nodiscard]] constexpr DualVec operator*(const T& rhs) const noexcept {
        DualVec res; res.v = v * rhs;
        for (size_t i = 0; i < N; ++i) res.g[i] = g[i] * rhs;
        return res;
    }
    [[nodiscard]] constexpr DualVec operator/(const T& rhs) const noexcept {
        DualVec res; res.v = v / rhs;
        for (size_t i = 0; i < N; ++i) res.g[i] = g[i] / rhs;
        return res;
    }

    // 제자리(In-place) 연산자
    constexpr DualVec& operator+=(const DualVec& rhs) noexcept {
        v += rhs.v;
        for (size_t i = 0; i < N; ++i) g[i] += rhs.g[i];
        return *this;
    }
    constexpr DualVec& operator-=(const DualVec& rhs) noexcept {
        v -= rhs.v;
        for (size_t i = 0; i < N; ++i) g[i] -= rhs.g[i];
        return *this;
    }
    constexpr DualVec& operator*=(const DualVec& rhs) noexcept {
        for (size_t i = 0; i < N; ++i) g[i] = g[i] * rhs.v + v * rhs.g[i];
        v *= rhs.v;
        return *this;
    }
    constexpr DualVec& operator/=(const DualVec& rhs) noexcept {
        T den = rhs.v * rhs.v;
        for (size_t i = 0; i < N; ++i) g[i] = (g[i] * rhs.v - v * rhs.g[i]) / den;
        v /= rhs.v;
        return *this;
    }

    constexpr DualVec& operator+=(const T& rhs) noexcept { v += rhs; return *this; }
    constexpr DualVec& operator-=(const T& rhs) noexcept { v -= rhs; return *this; }
    constexpr DualVec& operator*=(const T& rhs) noexcept {
        v *= rhs;
        for (size_t i = 0; i < N; ++i) g[i] *= rhs;
        return *this;
    }
    constexpr DualVec& operator/=(const T& rhs) noexcept {
        v /= rhs;
        for (size_t i = 0; i < N; ++i) g[i] /= rhs;
        return *this;
    }
};

template <typename T, size_t N> [[nodiscard]] constexpr DualVec<T, N> operator+(const T& lhs, const DualVec<T, N>& rhs) noexcept { return rhs + lhs; }
template <typename T, size_t N> [[nodiscard]] constexpr DualVec<T, N> operator-(const T& lhs, const DualVec<T, N>& rhs) noexcept {
    DualVec<T, N> res(lhs - rhs.v);
    for (size_t i = 0; i < N; ++i) res.g[i] = -rhs.g[i];
    return res;
}
template <typename T, size_t N> [[nodiscard]] constexpr DualVec<T, N> operator*(const T& lhs, const DualVec<T, N>& rhs) noexcept { return rhs * lhs; }
template <typename T, size_t N> [[nodiscard]] constexpr DualVec<T, N> operator/(const T& lhs, const DualVec<T, N>& rhs) noexcept {
    DualVec<T, N> res(lhs / rhs.v);
    T factor = -lhs / (rhs.v * rhs.v);
    for (size_t i = 0; i < N; ++i) res.g[i] = factor * rhs.g[i];
    return res;
}

// =====================================================================
// 3. 범용 수학 함수 (초월 함수 등에 대한 연쇄 법칙(Chain Rule) 정의)
// =====================================================================
namespace ad {

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

DUAL_MATH_OVERLOAD(sin, std::cos(u.v))
DUAL_MATH_OVERLOAD(cos, -std::sin(u.v))
DUAL_MATH_OVERLOAD(exp, std::exp(u.v))
DUAL_MATH_OVERLOAD(log, T(1.0) / u.v)
DUAL_MATH_OVERLOAD(tanh, T(1.0) - std::pow(std::tanh(u.v), 2.0))
DUAL_MATH_OVERLOAD(tan, T(1.0) + std::pow(std::tan(u.v), 2.0))
DUAL_MATH_OVERLOAD(atan, T(1.0) / (T(1.0) + u.v * u.v))

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
    for (size_t i = 0; i < N; ++i) res.g[i] = factor * u.g[i];
    return res;
}

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
    for (size_t i = 0; i < N; ++i) res.g[i] = factor * u.g[i];
    return res;
}

template <typename T>
inline Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) {
    using std::atan2;
    T res_v = atan2(y.v, x.v);
    T den = x.v * x.v + y.v * y.v;
    if (den <= T(1e-16)) return {res_v, T(0.0)};
    return {res_v, (x.v * y.d - y.v * x.d) / den};
}

template <typename T, size_t N>
inline DualVec<T, N> atan2(const DualVec<T, N>& y, const DualVec<T, N>& x) {
    using std::atan2;
    T res_v = atan2(y.v, x.v);
    DualVec<T, N> res(res_v);
    T den = x.v * x.v + y.v * y.v;
    if (den <= T(1e-16)) return res;
    for (size_t i = 0; i < N; ++i) res.g[i] = (x.v * y.g[i] - y.v * x.g[i]) / den;
    return res;
}

template <typename T>
inline Dual<T> abs(const Dual<T>& u) {
    T res_v = std::abs(u.v);
    T deriv = (u.v == T(0)) ? T(0) : ((u.v > T(0)) ? u.d : -u.d);
    return {res_v, deriv};
}

template <typename T, size_t N>
inline DualVec<T, N> abs(const DualVec<T, N>& u) {
    T res_v = std::abs(u.v);
    DualVec<T, N> res(res_v);
    T sign = (u.v == T(0)) ? T(0) : ((u.v > T(0)) ? T(1) : T(-1));
    for (size_t i = 0; i < N; ++i) res.g[i] = sign * u.g[i];
    return res;
}

// 스칼라 타입 오버로딩 (Fallback)
template <typename T> inline T abs(const T& x) { return std::abs(x); }
template <typename T> inline T sqrt(const T& x) { return std::sqrt(x); }

}  // namespace ad

// =====================================================================
// 4. std::complex 초월함수 해석적 확장 (Complex Step Differentiation 지원)
// =====================================================================
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

template <typename T>
inline std::complex<T> pow(const std::complex<T>& u, double n) {
    using std::pow;
    T r = u.real(), i = u.imag();
    T res_v = pow(r, n);
    // [Architect's Fix] 0^n 일 때의 수치적 안정성 교정
    T deriv = (r <= T(1e-16) && n < 1.0) ? T(0.0) : (n * pow(r, n - 1.0));
    return std::complex<T>(res_v, i * deriv);
}

template <typename T>
inline std::complex<T> atan2(const std::complex<T>& y, const std::complex<T>& x) {
    T yr = y.real(), yi = y.imag();
    T xr = x.real(), xi = x.imag();
    T den = xr * xr + yr * yr;

    if (den <= T(1e-16)) {
        return std::complex<T>(std::atan2(yr, xr), T(0.0));
    }
    
    // [Architect's Fix] 치명적인 std::atan2() 컴파일/논리 버그 교정
    T real_part = std::atan2(yr, xr);
    T imag_part = (xr * yi - yr * xi) / den; 

    return std::complex<T>(real_part, imag_part);
}

// =====================================================================
// 5. Output Stream & Helper Functions
// =====================================================================

// [Architect's Fix] 전역 값 추출기 복구 (C++ ADL 및 하위 호환성 보장)
template <typename T> inline T get_value(const T& x) { return x; }
template <typename T> inline T get_value(const Dual<T>& x) { return x.v; }
template <typename T, size_t N> inline T get_value(const DualVec<T, N>& x) { return x.v; }

template <typename T>
std::ostream& operator<<(std::ostream& os, const Dual<T>& x) {
    os << "[" << x.v << ", d:" << x.d << "]";
    return os;
}

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const DualVec<T, N>& x) {
    os << "[" << x.v << ", g:(";
    for (size_t i = 0; i < N; ++i) {
        os << x.g[i] << (i < N - 1 ? ", " : "");
    }
    os << ")]";
    return os;
}

}  // namespace Optimization

#endif  // OPTIMIZATION_DUAL_HPP_