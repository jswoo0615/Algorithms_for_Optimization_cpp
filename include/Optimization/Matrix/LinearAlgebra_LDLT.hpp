#ifndef OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_

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
 * @brief 고속 LDLT 분해 (In-place, Column-Oriented)
 * @details
 * [Architect's Update]
 * 기존의 Row-major 순회 방식을 폐기하고, A의 k번째 열(Column)을 참조하여
 * j번째 열을 업데이트하는 SAXPY(외적) 형태로 재설계했습니다.
 * 이 구조는 메모리를 일렬로 긁어내므로 캐시 미스가 사실상 제로(0)에 수렴합니다.
 */
template <typename T, size_t N>
inline MathStatus LDLT_decompose(StaticMatrix<T, N, N>& mat) {
    for (size_t j = 0; j < N; ++j) {
        // 1. Column j 업데이트 (k < j 인 이전 열들을 이용)
        for (size_t k = 0; k < j; ++k) {
            // temp = L_{j,k} * D_{kk}
            T temp = mat(j, k) * mat(k, k);

            size_t i = j;

// --- SIMD 가속 구간: col_j -= temp * col_k ---
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if constexpr (std::is_same_v<T, float>) {
                float32x4_t v_temp = vdupq_n_f32(-temp);  // 빼기를 위해 음수화
                for (; i + 3 < N; i += 4) {
                    float32x4_t v_a_ik = vld1q_f32(&mat(i, k));
                    float32x4_t v_a_ij = vld1q_f32(&mat(i, j));
                    vst1q_f32(&mat(i, j), vfmaq_f32(v_a_ij, v_a_ik, v_temp));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                float64x2_t v_temp = vdupq_n_f64(-temp);
                for (; i + 1 < N; i += 2) {
                    float64x2_t v_a_ik = vld1q_f64(&mat(i, k));
                    float64x2_t v_a_ij = vld1q_f64(&mat(i, j));
                    vst1q_f64(&mat(i, j), vfmaq_f64(v_a_ij, v_a_ik, v_temp));
                }
            }
#elif defined(__AVX2__)
            if constexpr (std::is_same_v<T, float>) {
                __m256 v_temp = _mm256_set1_ps(-temp);
                for (; i + 7 < N; i += 8) {
                    // 열 중간부터 시작하므로 정렬이 깨질 수 있어 loadu(Unaligned) 사용
                    __m256 v_a_ik = _mm256_loadu_ps(&mat(i, k));
                    __m256 v_a_ij = _mm256_loadu_ps(&mat(i, j));
                    _mm256_storeu_ps(&mat(i, j), _mm256_fmadd_ps(v_a_ik, v_temp, v_a_ij));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                __m256d v_temp = _mm256_set1_pd(-temp);
                for (; i + 3 < N; i += 4) {
                    __m256d v_a_ik = _mm256_loadu_pd(&mat(i, k));
                    __m256d v_a_ij = _mm256_loadu_pd(&mat(i, j));
                    _mm256_storeu_pd(&mat(i, j), _mm256_fmadd_pd(v_a_ik, v_temp, v_a_ij));
                }
            }
#endif

            // --- Scalar Fallback ---
            for (; i < N; ++i) {
                mat(i, j) -= mat(i, k) * temp;
            }
        }

        // 2. 대각 원소 D_{jj} 추출 및 특이성(Singularity) 검사
        T D_jj = mat(j, j);
        if (MathTraits<T>::near_zero(D_jj)) {
            return MathStatus::SINGULAR;
        }

        // 3. L_{i,j} 계산 (D_{jj}로 나누기)
        T inv_D = static_cast<T>(1.0) / D_jj;
        for (size_t i = j + 1; i < N; ++i) {
            mat(i, j) *= inv_D;
        }
    }
    return MathStatus::SUCCESS;
}

/**
 * @brief Zero-Allocation 무할당 전진/후진 대입 (Solve)
 * @details 임시 벡터(y, z)를 완전히 제거하고 결과 벡터 x 공간에서 in-place 연산을 수행합니다.
 */
template <typename T, size_t N>
inline void LDLT_solve(const StaticMatrix<T, N, N>& mat, const StaticVector<T, N>& b,
                       StaticVector<T, N>& x) {
    x = b;  // 초기화

    // 1. Forward substitution (L * z = b) -> x에 in-place 수행
    // 열 기반(Column-oriented) 순회로 L1 캐시 효율 극대화
    for (size_t k = 0; k < N; ++k) {
        T x_k = x(k);
        for (size_t i = k + 1; i < N; ++i) {
            x(i) -= mat(i, k) * x_k;
        }
    }

    // 2. Diagonal scaling (D * y = z)
    for (size_t i = 0; i < N; ++i) {
        x(i) /= mat(i, i);
    }

    // 3. Backward substitution (L^T * x = y)
    // L의 전치행렬이므로 내적(Dot-product) 구조가 Column-major에 완벽히 부합함
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        T sum = static_cast<T>(0);
        for (size_t k = i + 1; k < N; ++k) {
            sum += mat(k, i) * x(k);
        }
        x(i) -= sum;
    }
}

// 편의성 래퍼: 테스트 코드 등에서 기존처럼 return 값으로 받을 수 있게 지원 (RVO 최적화)
template <typename T, size_t N>
inline StaticVector<T, N> LDLT_solve(const StaticMatrix<T, N, N>& mat,
                                     const StaticVector<T, N>& b) {
    StaticVector<T, N> x;
    LDLT_solve(mat, b, x);
    return x;
}

}  // namespace linalg
}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_LDLT_HPP_