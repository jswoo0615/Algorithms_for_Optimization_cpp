#ifndef OPTIMIZATION_MATH_TRAITS_HPP_
#define OPTIMIZATION_MATH_TRAITS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

#include "Optimization/Dual.hpp"

namespace Optimization {

// =============================================
// Layer 2: 수학 연산 상태 코드 (Math Status)
// =============================================
enum class MathStatus { 
    SUCCESS = 1, 
    SINGULAR = -1, 
    ILL_CONDITIONED = -2, 
    NUMERICAL_ERROR = -5 
};

// =============================================
// AD (Auto Differentiation) 호환 Traits 구조체
// =============================================

template <typename T>
struct MathTraits {
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }
    static T max(const T& x, const T& y) { return std::max(x, y); }
    static T min(const T& x, const T& y) { return std::min(x, y); }
    static bool near_zero(const T& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) {
        return std::abs(x) <= tol;
    }
    static T get_value(const T& x) { return x; }
};

// [Specialization 1] 1차원 Dual 객체 지원
template <typename T>
struct MathTraits<Optimization::Dual<T>> {
    static Optimization::Dual<T> abs(const Optimization::Dual<T>& x) { return Optimization::ad::abs(x); }
    static Optimization::Dual<T> sqrt(const Optimization::Dual<T>& x) { return Optimization::ad::sqrt(x); }
    static Optimization::Dual<T> max(const Optimization::Dual<T>& x, const Optimization::Dual<T>& y) { return (x.v > y.v) ? x : y; }
    static Optimization::Dual<T> min(const Optimization::Dual<T>& x, const Optimization::Dual<T>& y) { return (x.v < y.v) ? x : y; }
    static bool near_zero(const Optimization::Dual<T>& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) {
        return std::abs(x.v) <= tol;
    }
    static T get_value(const Optimization::Dual<T>& x) { return x.v; }
};

// [Architect's Fix: Specialization 2] N차원 DualVec 객체 완벽 지원
template <typename T, size_t N>
struct MathTraits<Optimization::DualVec<T, N>> {
    static Optimization::DualVec<T, N> abs(const Optimization::DualVec<T, N>& x) { return Optimization::ad::abs(x); }
    static Optimization::DualVec<T, N> sqrt(const Optimization::DualVec<T, N>& x) { return Optimization::ad::sqrt(x); }
    static Optimization::DualVec<T, N> max(const Optimization::DualVec<T, N>& x, const Optimization::DualVec<T, N>& y) { return (x.v > y.v) ? x : y; }
    static Optimization::DualVec<T, N> min(const Optimization::DualVec<T, N>& x, const Optimization::DualVec<T, N>& y) { return (x.v < y.v) ? x : y; }
    static bool near_zero(const Optimization::DualVec<T, N>& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) {
        return std::abs(x.v) <= tol;
    }
    static T get_value(const Optimization::DualVec<T, N>& x) { return x.v; }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_MATH_TRAITS_HPP_