#ifndef OPTIMIZATION_RICCATI_SOLVER_HPP_
#define OPTIMIZATION_RICCATI_SOLVER_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Solver/SolverStatus.hpp"
#include <array>

namespace Optimization {
namespace solver {

    /**
     * @brief Discrete-time Riccati Recursion Solver (LQR / KKT Block Solver)
     * @details 
     * O(H) 선형 시간 복잡도로 블록-희소(Block-Sparse) NMPC 선형 시스템의 해를 구합니다.
     * 166x166 Dense 역행렬 대신, 2x2 역행렬을 H번 수행하여 압도적인 속도를 달성합니다.
     */
    template <size_t H, size_t Nx, size_t Nu>
    struct RiccatiSolver {
        // [입력 데이터] 각 스텝 k에서의 선형화된 행렬들
        std::array<StaticMatrix<double, Nx, Nx>, H> A; // 상태 자코비안
        std::array<StaticMatrix<double, Nx, Nu>, H> B; // 입력 자코비안
        
        std::array<StaticMatrix<double, Nx, Nx>, H + 1> Q; // 상태 비용 (Hessian)
        std::array<StaticMatrix<double, Nu, Nu>, H> R;     // 입력 비용 (Hessian)
        
        std::array<StaticVector<double, Nx>, H + 1> q; // 상태 그래디언트 (Residual)
        std::array<StaticVector<double, Nu>, H> r;     // 입력 그래디언트 (Residual)
        
        std::array<StaticVector<double, Nx>, H> d;     // 동역학 오차 (Dynamics Gap: x_{next} - f(x,u))

        // [출력 데이터] 계산된 최적 스텝 (탐색 방향)
        std::array<StaticVector<double, Nx>, H + 1> dx;
        std::array<StaticVector<double, Nu>, H> du;

        /**
         * @brief Backward Pass & Forward Pass를 통해 최적의 dx, du를 계산합니다.
         */
        SolverStatus solve() {
            // Value Function P (Cost-to-go Hessian) 및 p (Cost-to-go Gradient)
            std::array<StaticMatrix<double, Nx, Nx>, H + 1> P;
            std::array<StaticVector<double, Nx>, H + 1> p;

            // Feedback Gain K 및 Feedforward term k
            std::array<StaticMatrix<double, Nu, Nx>, H> K;
            std::array<StaticVector<double, Nu>, H> k_ff;

            // --- 1. Backward Pass (꼬리에서 머리로) ---
            // Terminal 조건 초기화
            P[H] = Q[H];
            p[H] = q[H];

            for (int k = H - 1; k >= 0; --k) {
                // Quu = R_k + B_k^T * P_{k+1} * B_k
                StaticMatrix<double, Nu, Nu> Quu = R[k] + B[k].transpose() * P[k + 1] * B[k];
                // Qux = B_k^T * P_{k+1} * A_k
                StaticMatrix<double, Nu, Nx> Qux = B[k].transpose() * P[k + 1] * A[k];
                // Qxx = Q_k + A_k^T * P_{k+1} * A_k
                StaticMatrix<double, Nx, Nx> Qxx = Q[k] + A[k].transpose() * P[k + 1] * A[k];

                // qu = r_k + B_k^T * p_{k+1} + B_k^T * P_{k+1} * d_k
                StaticVector<double, Nu> qu = r[k] + B[k].transpose() * p[k + 1] + B[k].transpose() * (P[k + 1] * d[k]);
                // qx = q_k + A_k^T * p_{k+1} + A_k^T * P_{k+1} * d_k
                StaticVector<double, Nx> qx = q[k] + A[k].transpose() * p[k + 1] + A[k].transpose() * (P[k + 1] * d[k]);

                // Quu의 역행렬을 구하여 Gain 계산 (Nu가 2이므로 초고속 연산 가능)
                // -Quu * K = Qux  =>  K = -Quu^{-1} * Qux
                // -Quu * k_ff = qu => k_ff = -Quu^{-1} * qu
                StaticMatrix<double, Nu, Nu> neg_Quu;
                for(size_t i=0; i<Nu; ++i) for(size_t j=0; j<Nu; ++j) neg_Quu(i,j) = -Quu(i,j);

                // Nu 크기(2x2)의 LU 분해 (혹은 LDLT)
                StaticVector<int, Nu> pivot;
                if (linalg::LU_decompose(neg_Quu, pivot) != MathStatus::SUCCESS) {
                    return SolverStatus::MATH_ERROR; // 제어 불가능한 특이 지점
                }

                // 열벡터 단위로 해를 구하여 K 행렬 조립
                for (size_t j = 0; j < Nx; ++j) {
                    StaticVector<double, Nu> Qux_col;
                    for (size_t i = 0; i < Nu; ++i) Qux_col(i) = Qux(i, j);
                    StaticVector<double, Nu> K_col = linalg::LU_solve(neg_Quu, pivot, Qux_col);
                    for (size_t i = 0; i < Nu; ++i) K[k](i, j) = K_col(i);
                }
                
                // k_ff 계산
                k_ff[k] = linalg::LU_solve(neg_Quu, pivot, qu);

                // P_k = Qxx + Qux^T * K
                P[k] = Qxx + Qux.transpose() * K[k];
                // p_k = qx + Qux^T * k_ff
                p[k] = qx + Qux.transpose() * k_ff[k];
            }

            // --- 2. Forward Pass (머리에서 꼬리로) ---
            // 초기 상태는 고정되어 있으므로 변화량(dx_0)은 0
            for (size_t i = 0; i < Nx; ++i) dx[0](i) = 0.0;

            for (size_t k = 0; k < H; ++k) {
                // du_k = K_k * dx_k + k_ff_k
                du[k] = K[k] * dx[k] + k_ff[k];
                
                // dx_{k+1} = A_k * dx_k + B_k * du_k + d_k
                dx[k + 1] = A[k] * dx[k] + B[k] * du[k] + d[k];
            }

            return SolverStatus::SUCCESS;
        }
    };

} // namespace solver
} // namespace Optimization

#endif // OPTIMIZATION_RICCATI_SOLVER_HPP_