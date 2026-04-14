#ifndef OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_
#define OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_

#include <cmath>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @brief 정적 할당 기반의 희소 행렬(Sparse Matrix) 클래스 (CSR 포맷 사용)
 *
 * @tparam T 데이터 타입 (예: float, double)
 * @tparam Rows 행렬의 행(Row) 개수
 * @tparam Cols 행렬의 열(Column) 개수
 * @tparam MaxNNZ 행렬 내 0이 아닌 값(Non-Zero)의 최대 개수
 *
 * 이 클래스는 CSR (Compressed Sparse Row) 형식을 사용하여 희소 행렬을 메모리 효율적으로 표현합니다.
 * 동적 할당을 피하기 위해 StaticVector를 사용하여 컴파일 타임에 결정된 크기의 버퍼를 사용합니다.
 */
template <typename T, size_t Rows, size_t Cols, size_t MaxNNZ>
class StaticSparseMatrix {
   public:
    /**
     * @brief 0이 아닌 실제 데이터 값들을 순차적으로 저장하는 배열
     */
    StaticVector<T, MaxNNZ> values;

    /**
     * @brief `values` 배열의 각 원소가 속한 열(Column)의 인덱스를 저장하는 배열
     */
    StaticVector<int, MaxNNZ> col_index;

    /**
     * @brief 각 행(Row)의 첫 번째 0이 아닌 원소가 `values` 배열에서 어느 위치부터 시작하는지
     * 나타내는 배열. 크기는 (Rows + 1)이며, 마지막 원소는 총 NNZ(Non-Zero) 개수를 나타냅니다.
     */
    StaticVector<int, Rows + 1> row_ptr;

    /**
     * @brief 현재까지 저장된 0이 아닌 값(NNZ)의 총 개수
     */
    size_t nnz_count;

    /**
     * @brief 생성자: 초기화 및 메모리 0으로 설정
     */
    StaticSparseMatrix() : nnz_count(0) {
        values.set_zero();
        col_index.set_zero();
        row_ptr.set_zero();
    }

    /**
     * @brief CSR 구조 조립을 위한 요소 추가 함수
     *
     * @param r 행(Row) 인덱스
     * @param c 열(Column) 인덱스
     * @param val 행렬에 삽입할 0이 아닌 데이터 값
     *
     * 주의: 이 함수를 호출할 때는 행(Row) 순서대로(위에서 아래로, 왼쪽에서 오른쪽으로) 추가하는
     * 것이 좋습니다. 이 함수를 모두 호출한 후에는 반드시 `finalize()`를 호출하여 `row_ptr`를 올바른
     * CSR 형식으로 완성해야 합니다.
     */
    void add_value(int r, int c, T val) {
        if (nnz_count < MaxNNZ) {
            // 현재 NNZ 위치에 값과 열 인덱스를 저장
            values(static_cast<int>(nnz_count)) = val;
            col_index(static_cast<int>(nnz_count)) = c;

            // 해당 행(Row)의 원소 개수를 임시로 누적 (finalize()에서 누적합으로 변환됨)
            row_ptr(r + 1) += 1;

            nnz_count++;
        }
    }

    /**
     * @brief CSR 포맷의 `row_ptr` 구성을 완료하는 함수
     *
     * `add_value()`를 통해 각 행의 원소 개수만 임시로 저장된 `row_ptr` 배열을
     * 누적합(Prefix Sum)으로 변환하여, 각 행의 시작 인덱스를 정확히 가리키도록 만듭니다.
     * 희소 행렬 요소 추가가 모두 끝난 후 곱셈 등의 연산을 수행하기 전에 반드시 한 번 호출해야
     * 합니다.
     */
    void finalize() {
        for (size_t i = 0; i < Rows; ++i) {
            row_ptr(i + 1) += row_ptr(i);
        }
    }

    /**
     * @brief 희소 행렬과 벡터의 곱셈 (y = A * x)
     *
     * @param x 곱할 열 벡터 (크기: Cols)
     * @param y 결과가 저장될 열 벡터 (크기: Rows)
     */
    void multiply(const StaticVector<T, Cols>& x, StaticVector<T, Rows>& y) const {
        y.set_zero();
        for (size_t i = 0; i < Rows; ++i) {
            T sum = 0.0;
            // row_ptr(i)부터 row_ptr(i+1)-1 까지가 i번째 행에 존재하는 0이 아닌 원소들입니다.
            for (int j = row_ptr(i); j < row_ptr(i + 1); ++j) {
                // 원소 값 * 해당 열 인덱스에 위치한 x 벡터의 값
                sum += values(j) * x(col_index(j));
            }
            y(static_cast<int>(i)) = sum;
        }
    }

    /**
     * @brief 희소 행렬의 전치 행렬과 벡터의 곱셈 (y = A^T * x)
     *
     * 원본 행렬 A를 전치하지 않고, A의 CSR 구조를 그대로 사용하여 전치 행렬 곱셈을 수행합니다.
     *
     * @param x 곱할 열 벡터 (크기: Rows)
     * @param y 결과가 저장될 열 벡터 (크기: Cols)
     */
    void multiply_transpose(const StaticVector<T, Rows>& x, StaticVector<T, Cols>& y) const {
        y.set_zero();
        for (size_t i = 0; i < Rows; ++i) {
            T xi = x(i);
            // i번째 행에 있는 모든 0이 아닌 원소들에 대해 연산
            for (int j = row_ptr(i); j < row_ptr(i + 1); ++j) {
                // A(i, j) * x(i) 는 y(j)에 더해집니다. (A^T의 관점에서는 A^T(j, i) * x(i)가 되기
                // 때문)
                y(col_index(j)) += values(j) * xi;
            }
        }
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_SPARSE_MATRIX_ENGINE_HPP_