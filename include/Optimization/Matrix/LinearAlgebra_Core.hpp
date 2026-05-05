#ifndef OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_

#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"

// [Architect's Update] 하드웨어 가속 헤더
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Optimization {
namespace linalg {

/**
 * @brief 고속 행렬 곱셈 (C = A * B)
 * @details
 * [최적화 철학]
 * 1. Column-major에 최적화된 j-k-i 루프 순서 채택 (공간 지역성 극대화).
 * 2. C의 한 열(Column)에 A의 열들을 스칼라 배(B의 원소)하여 누적하는 SAXPY 방식.
 * 3. Zero-Allocation: 결과 행렬 C를 참조로 받아 메모리 할당을 방지.
 */
template <typename T, size_t M, size_t K, size_t N>
inline void multiply(const StaticMatrix<T, M, K>& A, const StaticMatrix<T, K, N>& B,
                     StaticMatrix<T, M, N>& C) {
    C.set_zero();
    const T* a_ptr = A.data_ptr();
    const T* b_ptr = B.data_ptr();
    T* c_ptr = C.data_ptr();

    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < K; ++k) {
            T b_val = b_ptr[j * K + k];  // B(k, j) (스칼라 값)

            // A의 k번째 열과 C의 j번째 열의 시작 포인터
            const T* a_col = a_ptr + k * M;
            T* c_col = c_ptr + j * M;

            size_t i = 0;

// --- SIMD 가속 구간 (SAXPY) ---
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if constexpr (std::is_same_v<T, float>) {
                float32x4_t v_b = vdupq_n_f32(b_val);
                for (; i + 3 < M; i += 4) {
                    float32x4_t v_a = vld1q_f32(&a_col[i]);
                    float32x4_t v_c = vld1q_f32(&c_col[i]);
                    vst1q_f32(&c_col[i], vfmaq_f32(v_c, v_a, v_b));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                float64x2_t v_b = vdupq_n_f64(b_val);
                for (; i + 1 < M; i += 2) {
                    float64x2_t v_a = vld1q_f64(&a_col[i]);
                    float64x2_t v_c = vld1q_f64(&c_col[i]);
                    vst1q_f64(&c_col[i], vfmaq_f64(v_c, v_a, v_b));
                }
            }
#elif defined(__AVX2__)
            if constexpr (std::is_same_v<T, float>) {
                __m256 v_b = _mm256_set1_ps(b_val);
                for (; i + 7 < M; i += 8) {
                    // 열의 크기(M)에 따라 정렬이 깨질 수 있으므로 반드시 loadu(Unaligned) 사용
                    __m256 v_a = _mm256_loadu_ps(&a_col[i]);
                    __m256 v_c = _mm256_loadu_ps(&c_col[i]);
                    _mm256_storeu_ps(&c_col[i], _mm256_fmadd_ps(v_a, v_b, v_c));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                __m256d v_b = _mm256_set1_pd(b_val);
                for (; i + 3 < M; i += 4) {
                    __m256d v_a = _mm256_loadu_pd(&a_col[i]);
                    __m256d v_c = _mm256_loadu_pd(&c_col[i]);
                    _mm256_storeu_pd(&c_col[i], _mm256_fmadd_pd(v_a, v_b, v_c));
                }
            }
#endif

            // --- Scalar Fallback 구간 ---
            for (; i < M; ++i) {
                c_col[i] += a_col[i] * b_val;
            }
        }
    }
}

/**
 * @brief 고속 전치 행렬 곱셈 (C = A^T * B)
 * @details NMPC 자코비안 연산 (J^T * Q * J 등)에 필수적입니다.
 * A의 전치(Transpose)를 물리적으로 생성하지 않고, A의 열과 B의 열을 직접 내적(Dot Product)합니다.
 */
template <typename T, size_t M, size_t K, size_t N>
inline void multiply_AT_B(const StaticMatrix<T, K, M>& A, const StaticMatrix<T, K, N>& B,
                          StaticMatrix<T, M, N>& C) {
    const T* a_ptr = A.data_ptr();
    const T* b_ptr = B.data_ptr();
    T* c_ptr = C.data_ptr();

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            // C(i, j)는 A의 i번째 열과 B의 j번째 열의 내적
            const T* a_col = a_ptr + i * K;
            const T* b_col = b_ptr + j * K;

            T sum = 0.0;
            size_t k = 0;

// --- SIMD 가속 구간 (Dot Product) ---
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if constexpr (std::is_same_v<T, float>) {
                float32x4_t v_sum = vdupq_n_f32(0.0f);
                for (; k + 3 < K; k += 4) {
                    float32x4_t v_a = vld1q_f32(&a_col[k]);
                    float32x4_t v_b = vld1q_f32(&b_col[k]);
                    v_sum = vfmaq_f32(v_sum, v_a, v_b);
                }
                float sum_arr[4];
                vst1q_f32(sum_arr, v_sum);
                sum += sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
            } else if constexpr (std::is_same_v<T, double>) {
                float64x2_t v_sum = vdupq_n_f64(0.0);
                for (; k + 1 < K; k += 2) {
                    float64x2_t v_a = vld1q_f64(&a_col[k]);
                    float64x2_t v_b = vld1q_f64(&b_col[k]);
                    v_sum = vfmaq_f64(v_sum, v_a, v_b);
                }
                double sum_arr[2];
                vst1q_f64(sum_arr, v_sum);
                sum += sum_arr[0] + sum_arr[1];
            }
#elif defined(__AVX2__)
            if constexpr (std::is_same_v<T, float>) {
                __m256 v_sum = _mm256_setzero_ps();
                for (; k + 7 < K; k += 8) {
                    __m256 v_a = _mm256_loadu_ps(&a_col[k]);
                    __m256 v_b = _mm256_loadu_ps(&b_col[k]);
                    v_sum = _mm256_fmadd_ps(v_a, v_b, v_sum);
                }
                // 수평 덧셈 (Horizontal Add)을 통한 결과 취합
                alignas(32) float sum_arr[8];
                _mm256_store_ps(sum_arr, v_sum);
                for (int s = 0; s < 8; ++s) sum += sum_arr[s];
            } else if constexpr (std::is_same_v<T, double>) {
                __m256d v_sum = _mm256_setzero_pd();
                for (; k + 3 < K; k += 4) {
                    __m256d v_a = _mm256_loadu_pd(&a_col[k]);
                    __m256d v_b = _mm256_loadu_pd(&b_col[k]);
                    v_sum = _mm256_fmadd_pd(v_a, v_b, v_sum);
                }
                alignas(32) double sum_arr[4];
                _mm256_store_pd(sum_arr, v_sum);
                for (int s = 0; s < 4; ++s) sum += sum_arr[s];
            }
#endif

            // --- Scalar Fallback ---
            for (; k < K; ++k) {
                sum += a_col[k] * b_col[k];
            }
            c_ptr[j * M + i] = sum;
        }
    }
}

}  // namespace linalg

// =======================================================================================
// 연산자 오버로딩 (편의성용 - 실시간 루프 외곽의 초기화 등에 사용)
// =======================================================================================
template <typename T, size_t M, size_t K, size_t N>
inline StaticMatrix<T, M, N> operator*(const StaticMatrix<T, M, K>& A,
                                       const StaticMatrix<T, K, N>& B) {
    StaticMatrix<T, M, N> C;
    linalg::multiply(A, B, C);
    return C;
}

// 스칼라 곱셈
template <typename T, size_t M, size_t N>
inline StaticMatrix<T, M, N> operator*(T scalar, const StaticMatrix<T, M, N>& A) {
    StaticMatrix<T, M, N> C = A;
    constexpr size_t size = M * N;
    for (size_t i = 0; i < size; ++i) {
        C(i) *= scalar;
    }
    return C;
}

template <typename T, size_t M, size_t N>
inline StaticMatrix<T, M, N> operator*(const StaticMatrix<T, M, N>& A, T scalar) {
    return scalar * A;
}

// ---------------------------------------------------------------------------------------
// [Architect's Fix] 행렬 덧셈 / 뺄셈 오버로딩 추가 (Riccati Solver 지원)
// ---------------------------------------------------------------------------------------

/**
 * @brief 행렬 덧셈 (A + B)
 * @details 내부적으로 SIMD 가속이 적용된 operator+= 를 호출합니다.
 */
template <typename T, size_t M, size_t N>
inline StaticMatrix<T, M, N> operator+(const StaticMatrix<T, M, N>& A,
                                       const StaticMatrix<T, M, N>& B) {
    StaticMatrix<T, M, N> C = A;  // 복사 생성 (RVO 최적화됨)
    C += B;                       // SIMD 엔진 타격
    return C;
}

/**
 * @brief 행렬 뺄셈 (A - B)
 * @details C = C + (-1.0 * B) 형태의 FMA 연산으로 치환하여 SIMD 가속을 달성합니다.
 */
template <typename T, size_t M, size_t N>
inline StaticMatrix<T, M, N> operator-(const StaticMatrix<T, M, N>& A,
                                       const StaticMatrix<T, M, N>& B) {
    StaticMatrix<T, M, N> C = A;
    C.saxpy(static_cast<T>(-1.0), B);  // SIMD FMA 타격
    return C;
}

}  // namespace Optimization

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_CORE_HPP_