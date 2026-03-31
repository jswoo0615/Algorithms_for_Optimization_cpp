#include <iostream>
#include <cassert>
#include <iomanip>

/**
 * @brief 전방 선언 (Forward Declaration)
 * 컴파일러에게 StaticMatrix 존재 미리 명시
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix;

/**
 * @brief Alias Template
 * 상태 벡터를 명확히 명시
 */
template <typename T, size_t N>
using StaticVector = StaticMatrix<T, N, 1>;

/**
 * @brief 구현
 */
template <typename T, size_t Rows, size_t Cols>
class StaticMatrix {
    private:
        /**
         * @brief 정적 메모리 할당 (Column-Major 최적화를 위한 1차원 배열)
         */
        T data[Rows * Cols] {0.0};

    public:
        /**
         * @brief 메모리 접근자 (Accessors)
         */
        T& operator()(int r, int c) {
            assert(r >= 0 && r < Rows && c >= 0 && c < Cols);
            return data[c * Rows + r];  // Column-major
        }
        const T& operator()(int r, int c) const {
            assert(r >= 0 && r < Rows && c >= 0 && c < Cols);
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

        T* data_ptr() {
            return data;
        }
        const T* data_ptr() const {
            return data;
        }

        /**
         * @brief 기본 선형 대수 (Algebraic Operators)
         */
        StaticMatrix<T, Rows, Cols> operator+(const StaticMatrix<T, Rows, Cols>& Other) const {
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] + Other.data[i];
            }
            return Result;
        }

        StaticMatrix<T, Rows, Cols> operator-(const StaticMatrix<T, Rows, Cols>& Other) const {
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] - Other.data[i];
            }
            return Result;
        }

        StaticMatrix<T, Rows, Cols> operator*(T scalar) const {
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] * scalar;
            }
            return Result;
        }

        StaticMatrix<T, Rows, Cols> operator/(T scalar) const {
            assert(scalar != static_cast<T>(0) && "Division by zero in Matrix");
            StaticMatrix<T, Rows, Cols> Result;
            const size_t total = Rows * Cols;
            T inv_scalar = static_cast<T>(1.0) / scalar;
            for (size_t i = 0; i < total; ++i) {
                Result.data[i] = this->data[i] * inv_scalar;
            }
            return Result;
        }

        // 행렬 곱셈 (Cache-friendly Column-major 루프)
        template <size_t OtherCols>
        StaticMatrix<T, Rows, OtherCols> operator*(const StaticMatrix<T, Cols, OtherCols>& Other) const {
            StaticMatrix<T, Rows, OtherCols> Result;
            T* res_base = Result.data_ptr();
            const T* a_base = this->data;
            const T* b_base = Other.data_ptr();

            for (size_t j = 0; j < OtherCols; ++j) {
                T* res_ptr = res_base + (j * Rows);
                for (size_t k = 0; k < Cols; ++k) {
                    const T* a_col_ptr = a_base + (k * Rows);
                    T temp = Other(k, j);

                    T* r_ptr = res_ptr;
                    const T* a_ptr = a_col_ptr;

                    for (size_t i = 0; i < Rows; ++i) {
                        *r_ptr++ += (*a_ptr++) * temp;
                    }
                }
            }
            return Result;
        }

        /**
         * @brief LU 분해 및 솔버
         * @note In-place Doolittle 알고리즘
         */
        bool LU_decompose() {
            assert(Rows == Cols && "LU Decomposition requires a square matrix");

            for (size_t i = 0; i < Rows; ++i) {
                // 1. U 행렬 계산
                for (size_t j = i; j < Cols; ++j) {
                    T sum = 0.0;
                    for (size_t k = 0; k < i; ++k) {
                        sum += (*this)(i, k) * (*this)(k, j);
                    }
                    (*this)(i, j) = (*this)(i, j) - sum;
                }

                // 2. L 행렬 계산
                for (size_t j = i + 1; j < Rows; ++j) {
                    T sum = 0.0;
                    for (size_t k = 0; k < i; ++k) {
                        sum += (*this)(j, k) * (*this)(k, i);
                    }

                    if ((*this)(i, i) == static_cast<T>(0)) {
                        return false;   // Singular Matrix
                    }
                    (*this)(j, i) = ((*this)(j, i) - sum) / (*this)(i, i);
                }
            }
            return true;
        }

        // LU 행렬을 이용한 해 탐색 (StaticVector 활용)
        StaticVector<T, Rows> LU_solve(const StaticVector<T, Rows>& b) const {
            assert(Rows == Cols && "Solve requires a square matrix");

            StaticVector<T, Rows> y;        // Ly = b
            StaticVector<T, Rows> x;        // Ux = y

            // 1. 전진 대입 (Forward Substitution)
            for (size_t i = 0; i < Rows; ++i) {
                T sum = 0.0;
                for (size_t j = 0; j < i; ++j) {
                    sum += (*this)(i, j) * y(j);
                }
                y(i) = b(i) - sum;
            }

            // 2. 후진 대입 (Backward Substitution)
            for (size_t i = Rows - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t j = i + 1; j < Cols; ++j) {
                    sum += (*this)(i, j) * x(j);
                }
                x(i) = (y(i) - sum) / (*this)(i, i);
            }
            return x;
        }

        // -----------------------------------------------------------
        // 촐레스키 분해 (Cholesky Decomposition)
        // A = L * L^T 하삼각 부분에 L을 덮어씁니다
        // -----------------------------------------------------------

        bool Cholesky_decompose() {
            assert(Rows == Cols && "Cholesky requires a square matrix");
            for (size_t j = 0; j < Cols; ++j) {
                T s = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    s += (*this)(j, k) * (*this)(j, k);
                }

                T d = (*this)(j, j) - s;
                // 대각 원소가 0 이하면 분해 불가
                if (d <= 0.0) {
                    return false;
                }

                (*this)(j, j) = std::sqrt(d);

                for (size_t i = j + 1; i < Rows; ++i) {
                    T s_ij = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        s_ij += (*this)(i, k) * (*this)(j, k);
                    }
                    (*this)(i, j) = ((*this)(i, j) - s_ij) / (*this)(j, j);
                }
            }
            return true;
        }

        // -----------------------------------------------------------
        // 촐레스키 솔버 (Solve Ax = b via LL^T)
        // 1. Ly = b (Forward)
        // 2. L^Tx = y (Backward)
        // -----------------------------------------------------------

        StaticVector<T, Rows> Cholesky_solve(const StaticVector<T, Rows>& b) const {
            assert(Rows == Cols);
            StaticVector<T, Rows> x;
            StaticVector<T, Rows> y;

            // 1. 전진 대입 (Forward Substitution : Ly = b)
            for (size_t i = 0; i < Rows; ++i) {
                T sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(i, k) * y(k);
                }
                y(i) = (b(i) - sum) / (*this)(i, i);
            }

            // 2. 후진 대입 (Backward Substitution : L^T x = y)
            // L^T의 원소는 (*this)(j, i)와 같음
            for (int i = Rows - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t k = i + 1; k < Rows; ++k) {
                    sum += (*this)(k, i) * x(k);    // L^T(i, k) == L(k, i)
                }
                x(i) = (y(i) - sum) / (*this)(i, i);
            }
            return x;
        }

        // ------------------------------------------------------
        // LDLT 분해 (Square-Root-Free)
        // 대각선에는 D를, 하삼각에는 L을 덮어씁니다 (L의 대각선 1은 생략)
        // ------------------------------------------------------
        bool LDLT_decompose() {
            assert(Rows == Cols && "LDLT requires a square matrix");

            for (size_t j = 0; j < Cols; ++j) {
                // 1. D의 j번째 대각 원소 계산
                T sum_D = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    // L_jk^2 * D_kk
                    T L_jk = (*this)(j, k);
                    sum_D += L_jk * L_jk * (*this)(k, k);
                }

                T D_jj = (*this)(j, j) - sum_D;

                // 특이 행렬 (Singular) 방어
                // Cholesky와 달리 D_jj가 음수여도 알고리즘 자체는 돌아가지만, 
                // 0인 경우는 0으로 나누기가 발생하므로 막아야 합니다
                if (D_jj == static_cast<T>(0)) {
                    return false;
                }

                (*this)(j, j) = D_jj;       // 대각선 자리에 D 저장

                // 2. L의 j번째 열 계산 (i > j)
                for (size_t i = j + 1; i < Rows; ++i) {
                    T sum_L = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        // L_ik * L_jk * D_kk
                        sum_L += (*this)(i, k) * (*this)(j, k) * (*this)(k, k);
                    }

                    // L_ij 저장 (대각선에 이미 저장된 D_jj로 나눔)
                    (*this)(i, j) = ((*this)(i, j) - sum_L) / (*this)(j, j);
                }
            }
            return true;
        }

        // ------------------------------------------------------
        // LDLT 솔버 (Solve Ax=b via LDL^T)
        // 3단 분리 : Lz = b -> Dy = z -> L^T x = y
        // ------------------------------------------------------
        StaticVector<T, Rows> LDLT_solve(const StaticVector<T, Rows>& b) const {
            assert(Rows == Cols);
            StaticVector<T, Rows> z;        // Lz = b
            StaticVector<T, Rows> y;        // Dy = z
            StaticVector<T, Rows> x;        // L^Tx = y

            // 1. 전진 대입 (Lz = b)
            // L의 대각 원소는 1이므로 나눌 필요가 없음
            for (size_t i = 0; i < Rows; ++i) {
                T sum = 0.0;
                for (size_t k = 0; k < i; ++k) {
                    sum += (*this)(i, k) * z(k);
                }
                z(i) = b(i) - sum;
            }

            // 2. 대각 행렬 스케일링 (Dy = z)
            // D는 대각 행렬이므로 자기 위치의 대각 원소로 나누면 됩니다
            for (size_t i = 0; i < Rows; ++i) {
                y(i) = z(i) / (*this)(i, i);
            }

            // 3. 후진 대입 (L^T x = y)
            // 역시 L^T의 대각 원소는 1이므로 나눌 필요가 없습니다
            for (int i = Rows - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t k = i + 1; k < Rows; ++k) {
                    sum += (*this)(k, i) * x(k);        // L^T 인덱스 반전
                }
                x(i) = y(i) - sum;
            }
            return x;
        }

        // ------------------------------------------------------
        // MGS-QR 분해 (Modified Gram-Schmidt)
        // 원본 행렬을 직교 행렬 Q로 덮어쓰고, 상삼각 행렬 R을 반환합니다
        // ------------------------------------------------------
        bool QR_decompose_MGS(StaticMatrix<T, Cols, Cols>& R) {
            // 직사각 행렬 (Rows >= Cols)도 지원합니다. (최소자승법 용도)
            assert(Rows >= Cols && "QR requires Rows >= Cols");

            for (size_t i = 0; i < Cols; ++i) {
                // 1. 현재 열 벡터의 길이 (Norm) 계산 (R_ii)
                T norm_sq = 0.0;
                for (size_t k = 0; k < Rows; ++k) {
                    norm_sq += (*this)(k, i) * (*this)(k, i);
                }
                R(i, i) = std::sqrt(norm_sq);

                // Rank 부족 (선형 종속) 검사
                if (R(i, i) < 1e-12) {
                    return false;
                }

                // 2. 현재 열 벡터를 정규화하여 직교 벡터 q_i 생성 (Q 덮어쓰기)
                for (size_t k = 0; k < Rows; ++k) {
                    (*this)(k, i) /= R(i, i);
                }

                // 3. MGS 핵심 : 직교화된 q_i를 이용해 뒤에 남은 모든 벡터 (v_j)에서 그림자를 뺌
                for (size_t j = i + 1; j < Cols; ++j) {
                    // 내적 (Dot product) 계산 : q_i^T * v_j (이 값이 R_ij가 됨)
                    T dot_product = 0.0;
                    for (size_t k = 0; k < Rows; ++k) {
                        dot_product += (*this)(k, i) * (*this)(k, j);
                    }
                    R(i, j) = dot_product;

                    // v_j = v_j - R_ij * q_i (그림자 빼기)
                    for (size_t k = 0; k < Rows; ++k) {
                        (*this)(k, j) -= dot_product * (*this)(k, i);
                    }
                }
            }
            return true;
        }

        // ------------------------------------------------------
        // QR 솔버 (Solve Rx = Q^T b)
        // 현재 엔진은 직교 행렬 Q가 되어 있어야 하며, 외부에서 구한 R을 받습니다
        // ------------------------------------------------------
        StaticVector<T, Cols> QR_solve(const StaticMatrix<T, Cols, Cols>& R, const StaticVector<T, Rows>& b) const {
            StaticVector<T, Cols> y;        // y = Q^T * b
            // 1. 직교 투영 (Q^T * b 연산)
            for (size_t i = 0; i < Cols; ++i) {
                T dot = 0.0;
                for (size_t k = 0; k < Rows; ++k) {
                    dot += (*this)(k, i) * b(k);            // Q의 i번째 열과 b의 내적
                }
                y(i) = dot;
            }

            StaticVector<T, Cols> x;

            // 2. 후진 대입 (Backward Substitution : R * x = y)
            for (int i = Cols - 1; i >= 0; --i) {       // 언더플로우 방지
                T sum = 0.0;
                for (size_t j = i + 1; j < Cols; ++j) {
                    sum += R(i, j) * x(j);
                }
                x(i) = (y(i) - sum) / R(i, i);
            }
            return x;
        }

        // ------------------------------------------------------
        // Householder QR Decomposition (Householder Reflection)
        // A 행렬을 파괴하며 상삼각에 R을, 하삼각에 거울 벡터 v를 덮어씁니다
        // 스케일 팩터 tau를 외부 벡터로 반환합니다
        // ------------------------------------------------------
        bool QR_decompose_Householder(StaticVector<T, Cols>& tau) {
            assert(Rows >= Cols && "QR requires Rows >= Cols");

            for (size_t i = 0; i < Cols; ++i) {
                // 1. 현재 열의 i번째 행부터 끝까지의 크기 (Norm) 계산
                T norm_x = 0.0;
                for (size_t k = i; k < Rows; ++k) {
                    norm_x += (*this)(k, i) * (*this)(k, i);
                }
                norm_x = std::sqrt(norm_x);

                // 이미 0으로 깎여있다면 반사할 필요가 없음
                if (norm_x < 1e-12) {
                    tau(i) = 0.0;
                    continue;
                }

                // 2. 수치적 안정성을 위한 부호 선택 (Cancellation 방지)
                T sign = ((*this)(i, i) >= 0.0) ? 1.0 : -1.0;
                T v0 = (*this)(i, i) + sign * norm_x;       // 첫 번째 거울 성분

                // 3. 거울 벡터 v 정규화 (v0가 1이 되도록 전체를 나눔)
                for (size_t k = i + 1; k < Rows; ++k) {
                    (*this)(k, i) /= v0;                    // 하삼각 영역에 v를 덮어씀
                }

                // 4. 스케일 팩터 tau 계산 : 2 / (v^T v)
                T v_sq_norm = 1.0;                          // v0는 1이므로 A(i, j) * 1
                for (size_t k = i + 1; k < Rows; ++k) {
                    v_sq_norm += (*this)(k, i) * (*this)(k, i);
                }
                tau(i) = 2.0 / v_sq_norm;

                // 5. R의 대각 원소 갱신 (반사된 후의 길이)
                (*this)(i, i) = -sign * norm_x;

                // 6. 남은 열 (j)들에 대해 거울 반사 (H) 적용 : A_j = A_j - tau * v * (v^T A_j)
                for (size_t j = i + 1; j < Cols; ++j) {
                    // 내적 계산 : dot = v^T * A_j
                    T dot = (*this)(i, j);                  // v0는 1이므로 A(i, j) * 1
                    for (size_t k = i + 1; k < Rows; ++k) {
                        dot += (*this)(k, i) * (*this)(k, j);
                    }

                    // A_j 업데이트
                    T tau_dot = tau(i) * dot;
                    (*this)(i, j) -= tau_dot;               // v0 = 1
                    for (size_t k = i + 1; k < Rows; ++k) {
                        (*this)(k, j) -= tau_dot * (*this)(k, i);
                    }
                }
            }
            return true;
        }

        // ------------------------------------------------------
        // Householder QR Solver (Solve Ax = b -> Rx = Q^T b)
        // b벡터에 연속적으로 거울 반사를 적용하여 Q^T b를 먼저 만들고, R로 풉니다
        // ------------------------------------------------------
        StaticVector<T, Cols> QR_solve_Householder(const StaticVector<T, Cols>& tau, const StaticVector<T, Rows>& b) const {
            StaticVector<T, Rows> y = b;            // 복사본 생성 (y를 변형하여 Q^T b로 만듦)

            // 1. 직교 투영 (y = Q^T b)
            // 저장된 거울 벡터 v들을 꺼내어 연속으로 반사시킵니다
            for (size_t i = 0; i < Cols; ++i) {
                if (tau(i) == 0) {
                    continue;
                }

                // 내적 : dot = v^T * y
                T dot = y(i);                       // v0 = 1
                for (size_t k = i + 1; k < Rows; ++k) {
                    dot += (*this)(k, i) * y(k);
                }

                // y 업데이트 : y = y - tau * v * dot
                T tau_dot = tau(i) * dot;
                y(i) -= tau_dot;                    // v0 = 1
                for (size_t k = i + 1; k < Rows; ++k) {
                    y(k) -= tau_dot * (*this)(k, i);
                }
            }

            StaticVector<T, Cols> x;

            // 2. 후진 대입 (Backward Substitution : R * x = y)
            // R은 (*this)의 상삼각 영역에 존재합니다
            for (int i = Cols - 1; i >= 0; --i) {
                T sum = 0.0;
                for (size_t j = i + 1; j < Cols; ++j) {
                    sum += (*this)(i, j) * x(j);
                }
                x(i) = (y(i) - sum) / (*this)(i, i);
            }
            return x;
        }

        // ------------------------------------------------------
        // 디버깅 유틸리티
        // ------------------------------------------------------
        void print(const char* name) const {
            std::cout << "Matrix [" << name << "] (Col-major, " << Rows << "x" << Cols << "):\n";
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    std::cout << std::setw(8) << std::setprecision(4) << (*this)(i, j) << "\t";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }

        void print_LU_separated() const {
            assert(Rows == Cols);
            
            std::cout << "--- Separated L (Lower) ---\n";
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    if (i > j) std::cout << std::setw(8) << (*this)(i, j) << "\t";
                    else if (i == j) std::cout << std::setw(8) << 1.0 << "\t"; // 대각선은 1
                    else std::cout << std::setw(8) << 0.0 << "\t";
                }
                std::cout << "\n";
            }

            std::cout << "\n--- Separated U (Upper) ---\n";
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < Cols; ++j) {
                    if (i <= j) std::cout << std::setw(8) << (*this)(i, j) << "\t";
                    else std::cout << std::setw(8) << 0.0 << "\t";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }

};