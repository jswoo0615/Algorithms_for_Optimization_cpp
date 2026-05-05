#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>  // Jetson Nano (ARMv8 Cortex-A57)
#elif defined(__AVX2__)
#include <immintrin.h>  // PC (x86_64 AVX2/FMA)
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <type_traits>

namespace Optimization {

template <typename T, size_t Rows, size_t Cols>
class StaticMatrix;

template <typename T, size_t N>
using StaticVector = StaticMatrix<T, N, 1>;

/**
 * @brief Layer 1 : SIMD/NEON Optimized Static Matrix Engine
 *
 * [설계 지침]
 * 1. 64바이트 정렬: 캐시 라인 패널티 방지 및 SIMD 가속 호환성 확보
 * 2. Column-major 레이아웃: 수치 해석 표준 라이브러리와의 호환성 유지
 * 3. Zero-Allocation: 모든 메모리는 스택 혹은 정적 영역에 할당됨
 * 4. Multi-Precision: float 및 double 타입에 대한 완벽한 하드웨어 가속 분기
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
   private:
    // SIMD(AVX2/NEON) 및 캐시 최적화를 위한 64바이트 메모리 정렬
    alignas(64) T data[Rows * Cols]{};

   public:
    StaticMatrix() { set_zero(); }

    // =======================================================================================
    // 메모리 접근자 (Accessor - Column-major 방식)
    // =======================================================================================
    T& operator()(int r, int c) {
        assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
        return data[c * Rows + r];
    }
    const T& operator()(int r, int c) const {
        assert(r >= 0 && r < static_cast<int>(Rows) && c >= 0 && c < static_cast<int>(Cols));
        return data[c * Rows + r];
    }

    T& operator()(size_t i) {
        assert(i < Rows * Cols);
        return data[i];
    }
    const T& operator()(size_t i) const {
        assert(i < Rows * Cols);
        return data[i];
    }

    T* data_ptr() { return data; }
    const T* data_ptr() const { return data; }

    void set_zero() { std::fill(data, data + (Rows * Cols), static_cast<T>(0)); }

    // =======================================================================================
    // 하드웨어 가속 연산 모듈 (SAXPY & ADD)
    // =======================================================================================

    /**
     * @brief SAXPY 연산 (this = scalar * rhs + this)
     * FMA(Fused Multiply-Add)를 사용하여 단일 명령어로 정밀도와 속도를 동시에 확보합니다.
     */
    void saxpy(T scalar, const StaticMatrix& rhs) {
        constexpr size_t size = Rows * Cols;
        size_t i = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if constexpr (std::is_same_v<T, float>) {
            float32x4_t v_scalar = vdupq_n_f32(scalar);
            for (; i + 3 < size; i += 4) {
                float32x4_t v_rhs = vld1q_f32(&rhs.data[i]);
                float32x4_t v_res = vld1q_f32(&data[i]);
                v_res = vfmaq_f32(v_res, v_rhs, v_scalar);
                vst1q_f32(&data[i], v_res);
            }
        } else if constexpr (std::is_same_v<T, double>) {
            // ARMv8 NEON 64-bit 부동소수점 (double) 가속
            float64x2_t v_scalar = vdupq_n_f64(scalar);
            for (; i + 1 < size; i += 2) {
                float64x2_t v_rhs = vld1q_f64(&rhs.data[i]);
                float64x2_t v_res = vld1q_f64(&data[i]);
                v_res = vfmaq_f64(v_res, v_rhs, v_scalar);
                vst1q_f64(&data[i], v_res);
            }
        }
#elif defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            __m256 v_scalar = _mm256_set1_ps(scalar);
            for (; i + 7 < size; i += 8) {
                __m256 v_rhs = _mm256_load_ps(&rhs.data[i]);
                __m256 v_res = _mm256_load_ps(&data[i]);
                _mm256_store_ps(&data[i], _mm256_fmadd_ps(v_rhs, v_scalar, v_res));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            // AVX2 256-bit 부동소수점 (double) 가속
            __m256d v_scalar = _mm256_set1_pd(scalar);
            for (; i + 3 < size; i += 4) {
                __m256d v_rhs = _mm256_load_pd(&rhs.data[i]);
                __m256d v_res = _mm256_load_pd(&data[i]);
                _mm256_store_pd(&data[i], _mm256_fmadd_pd(v_rhs, v_scalar, v_res));
            }
        }
#endif

        // SIMD로 처리하지 못한 잔여 데이터 처리 (Scalar Fallback)
        for (; i < size; ++i) data[i] += scalar * rhs.data[i];
    }

    /**
     * @brief 행렬 요소별 가산 (operator +=)
     */
    StaticMatrix& operator+=(const StaticMatrix& rhs) {
        constexpr size_t size = Rows * Cols;
        size_t i = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if constexpr (std::is_same_v<T, float>) {
            for (; i + 3 < size; i += 4) {
                float32x4_t v1 = vld1q_f32(&data[i]);
                float32x4_t v2 = vld1q_f32(&rhs.data[i]);
                vst1q_f32(&data[i], vaddq_f32(v1, v2));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (; i + 1 < size; i += 2) {
                float64x2_t v1 = vld1q_f64(&data[i]);
                float64x2_t v2 = vld1q_f64(&rhs.data[i]);
                vst1q_f64(&data[i], vaddq_f64(v1, v2));
            }
        }
#elif defined(__AVX2__)
        if constexpr (std::is_same_v<T, float>) {
            for (; i + 7 < size; i += 8) {
                __m256 v1 = _mm256_load_ps(&data[i]);
                __m256 v2 = _mm256_load_ps(&rhs.data[i]);
                _mm256_store_ps(&data[i], _mm256_add_ps(v1, v2));
            }
        } else if constexpr (std::is_same_v<T, double>) {
            for (; i + 3 < size; i += 4) {
                __m256d v1 = _mm256_load_pd(&data[i]);
                __m256d v2 = _mm256_load_pd(&rhs.data[i]);
                _mm256_store_pd(&data[i], _mm256_add_pd(v1, v2));
            }
        }
#endif

        for (; i < size; ++i) data[i] += rhs.data[i];
        return *this;
    }

    // =======================================================================================
    // 블록 연산 모듈 (Block Operations)
    // =======================================================================================
    template <size_t SubRows, size_t SubCols>
    void insert_block(size_t start_row, size_t start_col,
                      const StaticMatrix<T, SubRows, SubCols>& block) {
        static_assert(SubRows <= Rows && SubCols <= Cols, "SubMatrix size mismatch");
        assert(start_row + SubRows <= Rows && start_col + SubCols <= Cols);

        for (size_t j = 0; j < SubCols; ++j) {
            const T* src = block.data_ptr() + (j * SubRows);
            T* dest = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            std::copy(src, src + SubRows, dest);
        }
    }

    template <size_t SubRows, size_t SubCols>
    void insert_transposed_block(size_t start_row, size_t start_col,
                                 const StaticMatrix<T, SubRows, SubCols>& block) {
        static_assert(SubCols <= Rows && SubRows <= Cols, "Transposed block size mismatch");
        assert(start_col + SubRows <= Cols && start_row + SubCols <= Rows);

        for (size_t j = 0; j < SubCols; ++j) {
            for (size_t i = 0; i < SubRows; ++i) {
                (*this)(static_cast<int>(start_row + j), static_cast<int>(start_col + i)) =
                    block(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    StaticMatrix<T, Cols, Rows> transpose() const {
        StaticMatrix<T, Cols, Rows> res;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                res(static_cast<int>(j), static_cast<int>(i)) =
                    (*this)(static_cast<int>(i), static_cast<int>(j));
            }
        }
        return res;
    }

    template <size_t SubRows, size_t SubCols>
    StaticMatrix<T, SubRows, SubCols> extract_block(size_t start_row, size_t start_col) const {
        assert(start_row + SubRows <= Rows && start_col + SubCols <= Cols);
        StaticMatrix<T, SubRows, SubCols> result;
        for (size_t j = 0; j < SubCols; ++j) {
            const T* src = this->data_ptr() + ((start_col + j) * Rows) + start_row;
            T* dest = result.data_ptr() + (j * SubRows);
            std::copy(src, src + SubRows, dest);
        }
        return result;
    }

    void print(const char* name) const {
        std::cout << "Matrix [" << name << "] (Col-major, SIMD-aligned, " << Rows << "x" << Cols
                  << "):\n";
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                std::cout << std::fixed << std::setw(10) << std::setprecision(4)
                          << (*this)(static_cast<int>(i), static_cast<int>(j)) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
};

}  // namespace Optimization

#endif  // STATIC_MATRIX_HPP_