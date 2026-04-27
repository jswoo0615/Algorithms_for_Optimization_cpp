#ifndef OPTIMIZATION_MATH_TRAITS_HPP_
#define OPTIMIZATION_MATH_TRAITS_HPP_

#include <cmath>
#include <limits>

#include "Optimization/Dual.hpp"

namespace Optimization {
// =============================================
// AD (Auto Differentiation) 호환 Traits 구조체
// =============================================

/**
 * @brief Layer 2: 수학 연산 상태 코드 (Math Status)
 * * [입법자의 철학]
 * SUCCESS: 계산 결과가 완벽하며 신뢰할 수 있음.
 * SINGULAR: 물리적 구속 조건의 충돌 등으로 인해 역행렬 계산이 원천 불가함.
 * ILL_CONDITIONED: 계산은 되었으나, 수치적 오차가 커서 제어 명령으로 쓰기에 위험함.
 * NUMERICAL_ERROR: NaN 또는 Inf가 발생하여 메모리 오염이 시작됨.
 */
enum class MathStatus { SUCCESS = 1, SINGULAR = -1, ILL_CONDITIONED = -2, NUMERICAL_ERROR = -5 };

template <typename T>
struct MathTraits {
    static T abs(const T& x) { return std::abs(x); }
    static T sqrt(const T& x) { return std::sqrt(x); }
    static bool near_zero(const T& x) {
        // 하드웨어 머신 입실론을 기준으로 수치적 0을 판단
        return std::abs(x) <= std::numeric_limits<T>::epsilon();
    }
};

template <typename T>
struct MathTraits<Optimization::Dual<T>> {
    static Optimization::Dual<T> abs(const Optimization::Dual<T>& x) {
        return Optimization::ad::abs(x);
    }
    static Optimization::Dual<T> sqrt(const Optimization::Dual<T>& x) {
        return Optimization::ad::sqrt(x);
    }
    static bool near_zero(const Optimization::Dual<T>& x) {
        // Dual 타입은 실수부(v)의 값으로 특이성을 판단
        return std::abs(x.v) <= std::numeric_limits<T>::epsilon();
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_MATH_TRAITS_HPP_