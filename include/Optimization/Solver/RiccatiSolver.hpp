#ifndef OPTIMIZATION_RICCATI_SOLVER_HPP_
#define OPTIMIZATION_RICCATI_SOLVER_HPP_

#include <array>

#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
namespace solver {

/**
 * @brief 고속 이산시간 리카티 재귀 솔버 (Zero-Allocation & Cache-Friendly)
 * @details
 * 블록-희소(Block-Sparse) 구조를 파괴하지 않고 O(H) 시간 복잡도로 역행렬을 계산합니다.
 * LU 대신 SPD(대칭 양의 정치) 행렬 전용 LDLT 분해를 사용하여 연산 속도를 극대화했습니다.
 */
template <size_t H, size_t Nx, size_t Nu>
struct RiccatiSolver {
    std::array<StaticMatrix<double, Nx, Nx>, H> A;
    std::array<StaticMatrix<double, Nx, Nu>, H> B;

    std::array<StaticMatrix<double, Nx, Nx>, H + 1> Q;
    std::array<StaticMatrix<double, Nu, Nu>, H> R;

    std::array<StaticVector<double, Nx>, H + 1> q;
    std::array<StaticVector<double, Nu>, H> r;

    std::array<StaticVector<double, Nx>, H> d;

    // [출력 데이터]
    std::array<StaticVector<double, Nx>, H + 1> dx;
    std::array<StaticVector<double, Nu>, H> du;

    SolverStatus solve() {
        std::array<StaticMatrix<double, Nx, Nx>, H + 1> P;
        std::array<StaticVector<double, Nx>, H + 1> p;

        std::array<StaticMatrix<double, Nu, Nx>, H> K;
        std::array<StaticVector<double, Nu>, H> k_ff;

        // --- 1. Backward Pass (꼬리에서 머리로) ---
        P[H] = Q[H];
        p[H] = q[H];

        // 루프 내부의 스택 생성/소멸 오버헤드를 막기 위한 캐시 변수 사전 선언
        StaticMatrix<double, Nx, Nu> PB;
        StaticMatrix<double, Nx, Nx> PA;
        StaticMatrix<double, Nu, Nu> Bt_PB;
        StaticMatrix<double, Nu, Nx> Qux;
        StaticMatrix<double, Nx, Nx> At_PA;
        StaticVector<double, Nx> Pd;
        StaticVector<double, Nx> p_next_mod;
        StaticVector<double, Nu> Bt_p;
        StaticVector<double, Nx> At_p;

        for (int k = H - 1; k >= 0; --k) {
            // [결합 법칙 최적화] P_{k+1} * B_k 와 P_{k+1} * A_k 를 미리 계산 (재사용)
            linalg::multiply(P[k + 1], B[k], PB);
            linalg::multiply(P[k + 1], A[k], PA);

            // Quu = R_k + B_k^T * (P_{k+1} * B_k)
            StaticMatrix<double, Nu, Nu> Quu = R[k];
            linalg::multiply_AT_B(B[k], PB, Bt_PB); // 가상 전치(Virtual Transpose) 타격
            Quu += Bt_PB;

            // Qux = B_k^T * (P_{k+1} * A_k)
            linalg::multiply_AT_B(B[k], PA, Qux);

            // Qxx = Q_k + A_k^T * (P_{k+1} * A_k)
            StaticMatrix<double, Nx, Nx> Qxx = Q[k];
            linalg::multiply_AT_B(A[k], PA, At_PA);
            Qxx += At_PA;

            // p_next_mod = p_{k+1} + P_{k+1} * d_k
            linalg::multiply(P[k + 1], d[k], Pd);
            p_next_mod = p[k + 1] + Pd;

            // qu = r_k + B_k^T * p_next_mod
            StaticVector<double, Nu> qu = r[k];
            linalg::multiply_AT_B(B[k], p_next_mod, Bt_p);
            qu += Bt_p;

            // qx = q_k + A_k^T * p_next_mod
            StaticVector<double, Nx> qx = q[k];
            linalg::multiply_AT_B(A[k], p_next_mod, At_p);
            qx += At_p;

            // --- 고속 LDLT 분해 (Quu는 항상 SPD) ---
            StaticMatrix<double, Nu, Nu> Quu_factored = Quu;
            if (linalg::LDLT_decompose(Quu_factored) != MathStatus::SUCCESS) {
                return SolverStatus::MATH_ERROR; // 제어 불가능 지점
            }

            // K = -Quu^{-1} * Qux 계산 (Column by Column in-place solve)
            for (size_t j = 0; j < Nx; ++j) {
                StaticVector<double, Nu> neg_Qux_col;
                for (size_t i = 0; i < Nu; ++i) neg_Qux_col(i) = -Qux(i, j); // 부호 반전
                
                StaticVector<double, Nu> K_col;
                linalg::LDLT_solve(Quu_factored, neg_Qux_col, K_col); // Zero-Allocation Solve
                for (size_t i = 0; i < Nu; ++i) K[k](i, j) = K_col(i);
            }

            // k_ff = -Quu^{-1} * qu 계산
            StaticVector<double, Nu> neg_qu;
            for (size_t i = 0; i < Nu; ++i) neg_qu(i) = -qu(i);
            linalg::LDLT_solve(Quu_factored, neg_qu, k_ff[k]);

            // P_k = Qxx + Qux^T * K_k
            P[k] = Qxx;
            StaticMatrix<double, Nx, Nx> QuxT_K;
            linalg::multiply_AT_B(Qux, K[k], QuxT_K);
            P[k] += QuxT_K;

            // p_k = qx + Qux^T * k_ff_k
            p[k] = qx;
            StaticVector<double, Nx> QuxT_kff;
            linalg::multiply_AT_B(Qux, k_ff[k], QuxT_kff);
            p[k] += QuxT_kff;
        }

        // --- 2. Forward Pass (머리에서 꼬리로) ---
        dx[0].set_zero(); // 초기 상태 섭동은 0

        for (size_t k = 0; k < H; ++k) {
            // du_k = K_k * dx_k + k_ff_k
            StaticVector<double, Nu> K_dx;
            linalg::multiply(K[k], dx[k], K_dx);
            du[k] = K_dx + k_ff[k];

            // dx_{k+1} = A_k * dx_k + B_k * du_k + d_k
            StaticVector<double, Nx> A_dx, B_du;
            linalg::multiply(A[k], dx[k], A_dx);
            linalg::multiply(B[k], du[k], B_du);
            dx[k + 1] = A_dx + B_du + d[k];
        }

        return SolverStatus::SUCCESS;
    }
};

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_RICCATI_SOLVER_HPP_