#ifndef STATIC_MATRIX_HPP_
#define STATIC_MATRIX_HPP_

#include <algorithm>        // std::copy, std::fill (연속 메모리 블록 단위의 고속 복사 및 초기화 기능 제공)
#include <cassert>          // 런타임 인덱스 경계 검사 (디버깅 용도로 쓰이며, 릴리즈 시 NDEBUG 매크로로 오버헤드 없이 비활성화 가능)
#include <cstddef>
#include <iomanip>          // 콘솔 출력 시 열을 맞추기 위한 포맷팅 조절용 (디버깅 목적)
#include <iostream>         // 행렬 상태를 확인하기 위한 디버깅 콘솔 입출력

namespace Optimization {
    /**
     * @brief 전방 선언 (Forward Declaration)
     * 
     * 컴파일러에게 `StaticMatrix` 템플릿 클래스의 존재를 미리 알려줍니다.
     * 이를 통해 클래스 본문이 정의되기 전에도, 자기 자신을 참조하는 멤버 함수나
     * `StaticVector`와 같은 연관된 별칭 (Alias) 타입에서 해당 이르믕ㄹ 자유롭게 사용할 수 있습니다.
     */
    template <typename T, size_t Rows, size_t Cols>
    class StaticMatrix;

    /**
     * @brief Alias Template (상태 벡터를 위한 별칭)
     * 
     * 선형 대수학에서 수학적 벡터 (Column Vector, 열 벡터)는 본질적으로 N행 1열 (N x 1) 형태의
     * 행렬입니다. 따라서 벡터를 위한 별도의 클래스를 구현하지 않고, `StaticMatrix<T, N, 1>`을
     * `StaticVector<T, N>`으로 이름 붙여 (Alias) 재사용합니다.
     * 이를 통해 메모리 구조상 행렬과 완벽히 동일하게 동작하며, 행렬-벡터 연산 간의 코드 일관성을 100% 유지합니다.
     */
    template <typename T, size_t N>
    using StaticVector = StaticMatrix<T, N, 1>;

    /**
     * @brief Layer 1 : Static Matrix Engine (Column-major 레이아웃 기반 정적 메모리 버퍼)
     * 
     * [핵심 설계 철학 : Static Memory Allocation]
     * 수학 연산은 Layer 2 (LinearAlgebra)로 분리되었으며,
     * 이 클래스는 오직 SIMD와 CUDA를 대비한 64바이트 정렬 선형 메모리의 I/O만을 담당합니다.
     */
    template <typename T, size_t Rows, size_t Cols>
    class StaticMatrix {
        private:
            // =======================================================================================
            // [메모리 최적화 : 정렬 (Alignment) 및 공간 지역성 (Spartial Locallity)]
            // alignas(64) : 최신 CPU (x86_64, ARM 등)의 L1 캐시 라인 (Cache Line) 크기인 64바이트 단위로
            // 메모리 시작 주소를 정렬합니다.
            //      - 캐시 히트율 (Cache Hit Rate) 극대화 : 프로세서가 메모리를 읽을 때 64바이트 덩어리 (정크)
            //      단위로 가져오므로, 정렬이 어긋나서 캐시 라인을 두 번 읽는 패널티 (False Sharing 등)를 방지합니다.
            //      - SIMD 가속 호환 : AVX-512 등의 벡터화 (Vectorization) 명령어는 메모리가 정렬되어 있지 않으면
            //      심각한 성능 저하나 하드웨어 예외를 발생시킵니다.
            //      - 이 배열은 Column-major (열 우선) 방식으로, `data` 배열 전체가 메모리 상에 완벽하게 일렬로
            //      연속된 (Contiguous) 하나의 1차원 블록을 형성합니다.
            // =======================================================================================
            alignas(64) T data[Rows * Cols] {};

        public:
            // =======================================================================================
            // 메모리 접근자 (Accessor - Column-major 방식)
            // 
            // [Column-major (열 우선) 레이아웃의 이유]
            // Fortran, BLAS, LAPACK, Eigen 등 세계적인 표준 수치해석 라이브러리들은 모두 Column-major를 사용합니다
            // 메모리 주소 계산 공식 : `data[col * Rows + row]` 즉, 같은 열 (Column)에 있는 원소들이 메모리 상에
            // 연속적으로 배치되어 있습니다. 열 단위 연산 (예 : 행렬 곱셈, 블록 삽입/추출) 시 공간 지역성을 극대화하여 
            // 캐시 미스 (Cache Miss)를 획기적으로 줄입니다.
            // =======================================================================================
            StaticMatrix() {
                set_zero();
            }
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

            T& data_ptr() {
                return data;
            }
            const T& data_ptr() const {
                return data;
            }
            
            /**
             * @brief 영행렬 초기화 (Zero Initialization)
             * 
             * 매 제어 주기 (Control Loop, ex: 10ms 마다 1회)가 시작될 때, 이전 주기의 KKT 행렬 데이터 (찌꺼기)를
             * 깨끗하게 0으로 비워야 새로운 최적화 계산을 꼬임 없이 시작할 수 있습니다.
             * 이중 루릎 (for i, for j)를 돌며 0을 개별 할당하는 것보다 `std::fill`을 1차원적으로 적용하면,
             * 최신 C++ 최적화 컴파일러가 이를 감지하고 단일 어셈블리 명령어인 고속 `memset` (메모리 블록 채우기)
             * 수준으로 치환 (Intrinsic)하여 연산 사이클을 극한으로 단축합니다.
             */
            void set_zero() {
                std::fill(data, data + (Rows * Cols), static_cast<T>(0));
            }

            // =======================================================================================
            // 블록 연산 모듈 (Block Operations & Assembly Utilities)
            // =======================================================================================
            template <size_t SubRows, size_t SubCols>
            void insert_block(size_t start_row, size_t start_col, const StaticMatrix<T, SubRows, SubCols>& block) {
                static_assert(SubRows <= Rows, "SubMatrix rows exceed target");
                static_assert(SubCols <= Cols, "SubMatrix cols exceed target");
                assert(start_row + SubRows <= Rows && "Row index out of bounds");
                assert(start_col + SubCols <= Cols && "Col index out of bounds");


                for (size_t j = 0; j < SubCols; ++j) {
                    const T* src = block.data_ptr() + (j * SubRows);
                    T* dest = this->data_ptr() + ((start_col + j) * Rows) + start_row;
                    std::copy(src, src + SubRows, dest);
                }
            }

            template <size_t SubRows, size_t SubCols>
            void insert_transposed_block(size_t start_row, size_t start_col, const StaticMatrix<T, SubRows, SubCols>& block) {
                static_assert(SubCols <= Rows, "Trnasposed block rows exceed target");
                static_assert(SubRows <= Cols, "Transposed block cols exceed target");
                assert(start_col + SubRows <= Cols && "Col index out of bounds");
                assert(start_row + SubCols <= Rows && "Row index out of bounds");
                
                for (size_t j = 0; j < SubCols; ++j) {
                    for (size_t i = 0; i < SubRows; ++i) {
                        (*this)(static_cast<int>(start_row + j), static_cast<int>(start_col + i)) = block(static_cast<int>(i), static_cast<int>(j));
                    }
                }
            }

            template <size_t SubRows, size_t SubCols>
            StaticMatrix<T, SubRows, SubCols> extra_block(size_t start_row, size_t start_col) const {
                assert(start_row + SubRows <= Rows && "Row index out of bounds");
                assert(start_col + SubCols <= Cols && "Col index out of bounds");

                StaticMatrix<T, SubRows, SubCols> result;
                for (size_t j = 0; j < SubCols; ++j) {
                    const T* src = this->data_ptr() + ((start_col + j) * Rows) + start_row;
                    T* dest = result.data_ptr() * (j * SubRows);
                    std::copy(src, src + SubRows, dest);
                }
                return result;
            }

            void print(const char* name) const {
            std::cout << "Matrix [" << name << "] (Col-major, " << Rows << "x" << Cols << "):\n";
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
} // namespace Optimization
#endif // STATIC_MATRIX_HPP_