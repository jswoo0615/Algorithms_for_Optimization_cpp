#ifndef OPTIMIZATION_DUAL_HPP_
#define OPTIMIZATION_DUAL_HPP_

#include <cmath>
#include <complex>
#include <array>
#include <algorithm>

namespace Optimization {
    // ===============================================
    // 1. Scalar Dual (1D AD)
    // ===============================================
    template <typename T>
    struct Dual {
        T v;        // Value
        T d;        // Derivative
        constexpr Dual(const T& value=T(0), const T& deriv=T(0)) : v(value), d(deriv) {}

        // 단일 음수 연산자 (Unary Minus)
        constexpr Dual operator-() const {
            return (-v, -d);
        }
        constexpr Dual operator+(const Dual& rhs) const {
            return {v + rhs.v, d + rhs.d};
        }
        constexpr Dual operator-(const Dual& rhs) const {
            return {v - rhs.v, d - rhs.d};
        }
        constexpr Dual operator*(const Dual& rhs) const {
            return {v * rhs.v, d * rhs.v + v * rhs.d};
        }
        constexpr Dual operator/(const Dual& rhs) const {
            T den = rhs.v * rhs.v;
            return {v / rhs.v, (d * rhs.v - v * rhs.d) / den};
        }

        // 우측 스칼라 (Dual * Scalar)
        constexpr Dual operator+(const T& rhs) const {
            return {v + rhs, d};
        }
        constexpr Dual operator-(const T& rhs) const {
            return {v - rhs, d};
        }
        constexpr Dual operator*(const T& rhs) const {
            return {v * rhs, d * rhs};
        }
        constexpr Dual operator/(const T& rhs) const {
            return {v / rhs, d / rhs};
        }
    };

    // 좌측 스칼라 (Scalar + Dual)
    template <typename T>
    constexpr Dual<T> operator+(const T& lhs, const Dual<T>& rhs) {
        return {lhs + rhs.v, rhs.d};
    }
    template <typename T>
    constexpr Dual<T> operator-(const T& lhs, const Dual<T>& rhs) {
        return {lhs - rhs.v, -rhs.d};
    }
    template <typename T>
    constexpr Dual<T> operator*(const T& lhs, const Dual<T>& rhs) {
        return {lhs * rhs.v, lhs * rhs.d};
    }
    template <typename T>
    constexpr Dual<T> operator/(const T& lhs, const Dual<T>& rhs) {
        T den = rhs.v * rhs.v;
        return {lhs / rhs.v, (-lhs * rhs.d) / den};
    }

    // =========================================
    // 2. Vector Dual (N-Dimensional AD)
    // =========================================
    template <typename T, size_t N>
    struct DualVec {
        T v;
        std::array<T, N> g;     // Gradient Vector
        constexpr DualVec(const T& value=T(0)) : v(value) {
            g.fill(T(0));
        }
        static constexpr DualVec make_variable(T value, size_t index) {
            DualVec res(value);
            if (index < N) {
                res.g[index] = T(1);
            }
            return res;
        }

        // 단항 음수 연산자 (Unary Minus)
        constexpr DualVec operator-() const {
            DualVec res;
            res.v = -v;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = -g[i];
            }
            return res;
        }

        constexpr DualVec operator+(const DualVec& rhs) const {
            DualVec res;
            res.v = v + rhs.v;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = g[i] + rhs.g[i];
            }
            return res;
        }

        constexpr DualVec operator-(const DualVec& rhs) const {
            DualVec res;
            res.v = v - rhs.v;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = g[i] - rhs.g[i];
            }
            return res;
        }

        constexpr DualVec operator*(const DualVec& rhs) const {
            DualVec res;
            res.v = v * rhs.v;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = g[i] * rhs.v + v * rhs.g[i];
            }
            return res;
        }

        constexpr DualVec operator/(const DualVec& rhs) const {
            DualVec res;
            T den = rhs.v * rhs.v;
            res.v = v / rhs.v;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = (g[i] * rhs.v - v * rhs.g[i]) / den;
            }
            return res;
        }

        // 우측 스칼라 (DualVec + Scalar)
        constexpr DualVec operator+(const T& rhs) const {
            return *this + DualVec(rhs);
        }
        constexpr DualVec operator-(const T& rhs) const {
            return *this - DualVec(rhs);
        }
        constexpr DualVec operator*(const T& rhs) const {
            DualVec res;
            res.v = v * rhs;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = g[i] * rhs;
            }
            return res;
        }
        constexpr DualVec operator/(const T& rhs) const {
            DualVec res;
            res.v = v / rhs;
            for (size_t i = 0; i < N; ++i) {
                res.g[i] = g[i] / rhs;
            }
            return res;
        }
    };

    // -- 좌측 스칼라 (Scalar + DualVec) ---
    template <typename T, size_t N>
    constexpr DualVec<T, N> operator+(const T& lhs, const DualVec<T, N>& rhs) {
        return rhs + lhs;
    }
    template <typename T, size_t N>
    constexpr DualVec<T, N> operator-(const T& lhs, const DualVec<T, N>& rhs) {
        DualVec<T, N> res(lhs - rhs.v);
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = -rhs.g[i];
        }
        return res;
    }

    template <typename T, size_t N>
    constexpr DualVec<T, N> operator*(const T& lhs, const DualVec<T, N>& rhs) {
        return rhs * lhs;
    }

    template <typename T, size_t N>
    constexpr DualVec<T, N> operator/(const T& lhs, const DualVec<T, N>& rhs) {
        DualVec<T, N> res(lhs / rhs.v);
        T factor = -lhs / (rhs.v * rhs.v);
        for (size_t i = 0; i < N; ++i) {
            res.g[i] = factor * rhs.g[i];
        }
        return res;
    }

    // ==================================================
    // 3. 범용 수학 함수 (Macro & Overloading)
    // ==================================================
    #define DUAL_MATH_OVERLOAD(funcName, derivativeExpr) \
    template <typename T> \
    inline Dual<T> funcName(const Dual<T>& u) { \
        using std::funcName; \
        T res_v = funcName(u.v); \
        return {res_v, (derivativeExpr) * u.d}; \
    } \
    template <typename T, size_t N> \
    inline DualVec<T, N> funcName(const DualVec<T, N>& u) { \
        using std::funcName; \
        T res_v = funcName(u.v); \
        DualVec<T, N> res(res_v); \
        T factor = (derivativeExpr); \
        for (size_t i = 0; i < N; ++i) { \
            res.g[i] = factor * u.g[i]; \
        } \
        return res; \
    }

    DUAL_MATH_OVERLOAD(sin, std::cos(u.v))
    DUAL_MATH_OVERLOAD(cos, -std::sin(u.v))
    DUAL_MATH_OVERLOAD(exp, std::exp(u.v))
    DUAL_MATH_OVERLOAD(log, T(1.0) / u.v)
    DUAL_MATH_OVERLOAD(tanh, T(1.0) - std::pow(std::tanh(u.v), 2.0))
    DUAL_MATH_OVERLOAD(tan, T(1.0) + std::pow(std::tan(u.v), 2.0))
    DUAL_MATH_OVERLOAD(atan, T(1.0) / (T(1.0) + u.v * u.v))

    // sqrt
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

    // pow 함수 오버로딩
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

    // atan2(y, x) 미분 : dy = x / (x^2 + y^2), dx = -y / (x^2 + y^2)
    template <typename T>
    inline Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) {
        using std::atan2;
        T res_v = atan2(y.v, x.v);
        T den = x.v * x.v + y.v * y.v;
        // x = 0, y = 0 특이점 방어
        if (den <= T(1e-16)) {
            return {res_v, T(0.0)};
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

    template <typename T>
    inline std::complex<T> atan2(const std::complex<T>& y, const std::complex<T>& x) {
        T yr = y.real(), yi = y.imag();
        T xr = x.real(), xi = x.imag();
        T den = xr * xr + yr * yr;

        if (den <= T(1e-16)) {
            return std::complex<T>(std::atan2(yr, xr), T(0.0));
        }
        T real_part = std::atan2(yr, xr);
        T imag_part = (xr * yi - yr * xi) / den;

        return std::complex<T>(real_part, imag_part);
    }

    // ======================================================
    // 4. CSD 엔진을 위한 std::complex 초월함수 해석적 확장
    // 표준 라이브러리의 복소수 함수에서 발생하는 1 + 1e-100 증발 방지
    // 1차 테일러 전개 기반으로 직접 오버로딩
    // ======================================================
    #define CSD_MATH_OVERLOAD(funcName, derivativeExpr) \
    template <typename T> \
    inline std::complex<T> funcName(const std::complex<T>& u) { \
        using std::funcName; \
        T r = u.real(); \
        T i = u.imag(); \
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
        T deriv = (r <= T(1e-16) && n < 1.0) ? T(0.0) : (n * pow(r, n-1.0));
        return std::complex<T>(res_v, i * deriv);
    }

    // --- Helper Function : Value extraction ---
    template <typename T>
    inline auto get_value(const T& x) {
        return x;
    }

    template <typename T>
    inline auto get_value(const Dual<T>& x) {
        return x.v;
    }

    template <typename T, size_t N>
    inline auto get_value(const DualVec<T, N>& x) {
        return x.v;
    }
} // namespace Optimization

#endif // OPTIMIZATION_DUAL_HPP_