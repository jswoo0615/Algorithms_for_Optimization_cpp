#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_

#include "Optimization/Matrix/LinearAlgebra_Core.hpp"
#include "Optimization/Matrix/MathTraits.hpp"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace Optimization {
namespace linalg {

/**
 * @brief 고속 LUP 분해 (In-place, Column-Oriented Rank-1 Update)
 * @details 
 * 부분 피벗팅(Partial Pivoting)을 포함한 LU 분해입니다.
 * 내부 Schur Complement 업데이트를 Column-major에 최적화된 SAXPY 루프로 변환하여
 * L1 캐시 히트율을 극대화하고 하드웨어 SIMD 가속을 적용했습니다.
 */
template <typename T, size_t N>
inline MathStatus LU_decompose(StaticMatrix<T, N, N>& mat, StaticVector<int, N>& P) {
    for (size_t i = 0; i < N; ++i) {
        P(i) = static_cast<int>(i);
    }

    for (size_t i = 0; i < N; ++i) {
        // 1. Pivot selection (Column 단위 순회이므로 캐시 친화적)
        T max_val = static_cast<T>(0);
        size_t pivot_row = i;
        for (size_t j = i; j < N; ++j) {
            T abs_val = MathTraits<T>::abs(mat(j, i));
            // Dual 타입 호환성을 위해 get_value 사용
            if (Optimization::get_value(abs_val) > Optimization::get_value(max_val)) {
                max_val = abs_val;
                pivot_row = j;
            }
        }

        if (MathTraits<T>::near_zero(max_val)) {
            return MathStatus::SINGULAR;
        }

        // 2. Row Swap (물리적 행 교환)
        if (pivot_row != i) {
            for (size_t k = 0; k < N; ++k) {
                T temp = mat(i, k);
                mat(i, k) = mat(pivot_row, k);
                mat(pivot_row, k) = temp;
            }
            int temp_p = P(i);
            P(i) = P(pivot_row);
            P(pivot_row) = temp_p;
        }

        // 3. Scale column i (L 행렬 요소 생성)
        T inv_pivot = static_cast<T>(1.0) / mat(i, i);
        for (size_t j = i + 1; j < N; ++j) {
            mat(j, i) *= inv_pivot;
        }

        // 4. Rank-1 Update (Schur Complement) -> Column-oriented SAXPY
        for (size_t j = i + 1; j < N; ++j) {
            T temp = mat(i, j); 
            size_t k = i + 1;

            // --- SIMD 가속 구간: col_j[k...] -= temp * col_i[k...] ---
            #if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if constexpr (std::is_same_v<T, float>) {
                float32x4_t v_temp = vdupq_n_f32(-temp);
                for (; k + 3 < N; k += 4) {
                    float32x4_t v_col_i = vld1q_f32(&mat(k, i));
                    float32x4_t v_col_j = vld1q_f32(&mat(k, j));
                    vst1q_f32(&mat(k, j), vfmaq_f32(v_col_j, v_col_i, v_temp));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                float64x2_t v_temp = vdupq_n_f64(-temp);
                for (; k + 1 < N; k += 2) {
                    float64x2_t v_col_i = vld1q_f64(&mat(k, i));
                    float64x2_t v_col_j = vld1q_f64(&mat(k, j));
                    vst1q_f64(&mat(k, j), vfmaq_f64(v_col_j, v_col_i, v_temp));
                }
            }
            #elif defined(__AVX2__)
            if constexpr (std::is_same_v<T, float>) {
                __m256 v_temp = _mm256_set1_ps(-temp);
                for (; k + 7 < N; k += 8) {
                    __m256 v_col_i = _mm256_loadu_ps(&mat(k, i));
                    __m256 v_col_j = _mm256_loadu_ps(&mat(k, j));
                    _mm256_storeu_ps(&mat(k, j), _mm256_fmadd_ps(v_col_i, v_temp, v_col_j));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                __m256d v_temp = _mm256_set1_pd(-temp);
                for (; k + 3 < N; k += 4) {
                    __m256d v_col_i = _mm256_loadu_pd(&mat(k, i));
                    __m256d v_col_j = _mm256_loadu_pd(&mat(k, j));
                    _mm256_storeu_pd(&mat(k, j), _mm256_fmadd_pd(v_col_i, v_temp, v_col_j));
                }
            }
            #endif

            // --- Scalar Fallback ---
            for (; k < N; ++k) {
                mat(k, j) -= mat(k, i) * temp;
            }
        }
    }
    return MathStatus::SUCCESS;
}

/**
 * @brief Zero-Allocation 무할당 LUP 전진/후진 대입 (Solve)
 * @details 임시 메모리를 할당하지 않고, Column-oriented 방식으로 캐시 미스를 억제합니다.
 */
template <typename T, size_t N>
inline void LU_solve(const StaticMatrix<T, N, N>& mat,
                     const StaticVector<int, N>& P, 
                     const StaticVector<T, N>& b,
                     StaticVector<T, N>& x) {
    
    // 1. Apply Permutation
    for (size_t i = 0; i < N; ++i) {
        x(i) = b(P(i));
    }

    // 2. Forward substitution (L * y = b) -> x에 in-place 수행
    // L 행렬의 대각 원소는 1이라고 가정합니다.
    for (size_t k = 0; k < N; ++k) {
        T x_k = x(k);
        // Column-oriented update
        for (size_t i = k + 1; i < N; ++i) {
            x(i) -= mat(i, k) * x_k;
        }
    }

    // 3. Backward substitution (U * x = y) -> x에 in-place 수행
    // [Architect's Note] 통상적인 Row-oriented 내적이 아닌, Column-oriented 소거법 적용
    for (int k = static_cast<int>(N) - 1; k >= 0; --k) {
        x(k) /= mat(k, k); // 대각 원소로 나누어 x(k) 확정
        T x_k = x(k);
        // 확정된 x(k)를 이용하여 그 위에 있는 모든 행의 값을 미리 빼줍니다 (Column 순회)
        for (int i = 0; i < k; ++i) {
            x(i) -= mat(i, k) * x_k;
        }
    }
}

// 편의성 래퍼 (RVO 지원)
template <typename T, size_t N>
inline StaticVector<T, N> LU_solve(const StaticMatrix<T, N, N>& mat,
                                   const StaticVector<int, N>& P, 
                                   const StaticVector<T, N>& b) {
    StaticVector<T, N> x;
    LU_solve(mat, P, b, x);
    return x;
}

}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_LU_HPP_