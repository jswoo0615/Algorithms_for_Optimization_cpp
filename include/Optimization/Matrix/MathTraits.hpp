#ifndef OPTIMIZATION_MATH_TRAITS_HPP_
#define OPTIMIZATION_MATH_TRAITS_HPP_

#include "Optimization/Dual.hpp"
#include <cmath>
#include <limits>

namespace Optimization {
    // =============================================
    // AD (Auto Differentiation) 호환 Traits 구조체
    // =============================================
    template <typename T>
    struct MathTraits {
        static T abs(const T& x) {
            return std::abs(x);
        }
        static T sqrt(const T& x) [
            return std::sqrt(x);
        ]
        static bool near_zero(const T& x) {
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
            return std::abs(x.v) <= std::numeric_limits<T>::epsilon();
        }
    };

} // namespace Optimization

#endif // OPTIMIZATION_MATH_TRAITS_HPP_