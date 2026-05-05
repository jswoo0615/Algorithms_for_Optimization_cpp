#ifndef OPTIMIZATION_MATH_TRAITS_HPP_
#define OPTIMIZATION_MATH_TRAITS_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

#include "Optimization/Dual.hpp"

// ======================================
// Layer 2 : 수학 연산 상태 코드 (Math Status)
// ======================================
/**
 * @brief 수학 연산 및 행렬 분해의 신뢰도 상태
 * SUCCESS : 계산 결과가 완벽하며 신뢰할 수 있음
 * SINGULAR : 물리적 구속 조건의 충돌 등으로 인해 역행렬 계산이 원천 불가함 (Det = 0)
 * ILL_CONDITIONED : 계산은 되었으나, 수치적 오차가 커서 제어 명령으로 사용하기에는 위험함
 * NUMERICAL_ERROR : NaN 또는 Inf가 발생하여 메모리 오염이 시작됨
 */
enum class MathStatus {
    SUCCESS = 1,
    SINGULAR = -1,
    ILL_CONDITIONED = -2,
    NUMERICAL_ERROR = -5
};

// ============================================
// AD (Auto Differentitation) 호환 Traits 구조체
// ============================================
template <typename T>
struct MathTraits {
    static T abs(const T& x) {
        return std::abs(x);
    }
    static T sqrt(const T& x) {
        return std::sqrt(x);
    }

    // IPM 장벽 계산용 Min / Max 추가
    static T max(const T& x, const T& y) {
        return std::max(x, y);
    }
    static T min(const T& x, const T& y) {
        return std::min(x, y);
    }

    // FMA 연산 오차를 고려한 Tolerance (기본값 : Epsilon * 10)
    static bool near_zero(const T& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) {
        return std::abs(x) <= tol;
    }

    // 값 추출기 : 일반 타입은 자기 자신을 반환
    static T get_value(const T& x) {
        return x;
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

    // Dual 타입의 비교는 오직 실제 수치 (v)를 기준으로 판단합니다.
    static Optimization::Dual<T> max(const Optimization::Dual<T>& x, const Optimization::Dual<T>& y) {
        return (x.v > y.v) ? x : y;
    }
    static Optimization::Dual<T> min(const Optimization::Dual<T>& x, const Optimization::Dual<T>& y) {
        return (x.v < y.v) ? x : y;
    }
    static bool near_zero(const Optimization::Dual<T>& x, T tol = std::numeric_limits<T>::epsilon() * static_cast<T>(10.0)) {
        return std::abs(x.v) <= tol;
    }
    // 값 추출기 : Dual 객체에서 수치값 (v)만 추출
    static T get_value(const Optimization::Dual<T>& x) {
        return x.v;
    }
};

// ====================================================
// 편의성 유틸리티 함수 (Global Accessors)
// ====================================================
/**
 * @brief 값 강제 추출 함수
 * NMPC 솔버 내에 템플릿 타입 (Dual or Scalar)에 종속되지 않고
 * if 문이나 비교 연산을 수행하기 위해 값만 빼냅니다.
 */
template <typename T>
inline auto get_value(const T& x) {
    return MathTraits<T>::get_value(x);
}

#endif // OPTIMIZATION_MATH_TRAITS_HPP_