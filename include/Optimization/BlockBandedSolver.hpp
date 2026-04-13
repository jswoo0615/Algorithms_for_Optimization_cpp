#ifndef OPTIMIZATION_BLOCK_BANDED_SOLVER_HPP_
#define OPTIMIZATION_BLOCK_BANDED_SOLVER_HPP_

#include <iostream>

#include "Optimization/Matrix/BlockBandedStorage.hpp"

namespace Optimization {
template <size_t Nx, size_t Nu, size_t Np>
class BlockBandedSolver {
   public:
    // 호라이즌 전체의 데이터를 담는 배열 (힙 할당 없음)
    std::array<NMPCStorageData<Nx, Nu>, Np> stages;

    // 터미널 코스트 (Terminal Stage)
    StaticMatrix<double, Nx, Nx> P_N;
    StaticVector<double, Nx> p_N_vec;

    // 전체 시스템의 해 (Delta x, Delta u)
    std::array<StaticVector<double, Nx>, Np + 1> dx;
    std::array<StaticVector<double, Nu>, Np> du;

    BlockBandedSolver() {
        P_N.set_zero();
        p_N_vec.set_zero();
    }

    /**
     * @brief 역방향 탐색 (Backward Pass)
     * 목적지 (N_p)에서 시작하여 현재 (0)까지 Value Function을 갱신하며
     * 최적 제어 법칙 (K)과 피드포워드 (k_ff)를 도출합니다.
     */
    bool backward_pass() {
        // 1. 종단 조건 (Terminal Condition) 초기화
        StaticMatrix<double, Nx, Nx> P_next = P_N;
        StaticVector<double, Nx> p_next = p_N_vec;

        // 2. 미래 (Np-1)부터 현재 (0)로 역행
        for (int k = static_cast<int>(Np) - 1; k >= 0; --k) {
            auto& s = stages[k];

            // H_xx = Q_k + A_k^T * P_{k+1} * A_k
            StaticMatrix<double, Nx, Nx> H_xx = s.Q + s.A.quadratic_multiply(P_next);

            // H_uu = R_k + B_k^T * P_{k+1} * B_k
            StaticMatrix<double, Nu, Nu> H_uu = s.R + s.B.quadratic_multiply(P_next);

            // H_ux = B_k^T * P_{k+1} * A_k
            StaticMatrix<double, Nx, Nx> P_A = P_next * s.A;
            StaticMatrix<double, Nu, Nx> H_ux = s.B.transpose() * P_A;

            // 벡터 항 계산 (Gradient)
            // g_u = r_k + B_k^T * (P_{k+1} * d_k + p_{k+1})
            StaticVector<double, Nx> P_d_p = (P_next * s.d) + p_next;
            StaticVector<double, Nu> g_u = s.r + (s.B.transpose() * P_d_p);

            // g_x = q_k + A_k^T * (P_{k+1} * d_k + p_{k+1})
            StaticVector<double, Nx> g_x = s.q + (s.A.transpose() * P_d_p);

            // H_uu 분해 (Positive Definite 검증)
            if (!H_uu.LDLT_decompose()) {
                // 수치적 불안정성 방어 (Damping)
                for (size_t i = 0; i < Nu; ++i) {
                    H_uu(i, i) += 1e-4;
                }
                if (!H_uu.LDLT_decompose()) {
                    return false;
                }
            }

            // 피드백 게인 K_k = -H_uu^{-1} * H_ux
            StaticMatrix<double, Nu, Nx> neg_H_ux = H_ux * -1.0;
            s.K = H_uu.template solve_multiple<Nx>(neg_H_ux);

            // 피드포워드 게인 k_ff = -H_uu^{-1} * g_u
            StaticVector<double, Nu> neg_g_u;
            for (size_t i = 0; i < Nu; ++i) {
                neg_g_u(i) = -g_u(i);
            }
            s.k_ff = H_uu.LDLT_solve(neg_g_u);

            // Value Function 갱신 (P_k, p_k)
            // P_k = H_xx + K_k^T * H_uu * K_k + K_k^T * H_ux + H_ux^T * K_k
            // 구조적 단순화 : P_k = H_xx + H_ux^T * K_k
            s.P = H_xx + (H_ux.transpose() * s.K);

            // p_k = g_x + H_ux^T * k_ff
            s.p_vec = g_x + (H_ux.transpose() * s.k_ff);

            // 다음 스텝 (뒤로 이동)을 위해 갱신
            P_next = s.P;
            p_next = s.p_vec;
        }
        return true;
    }

    /**
     * @brief 순방향 탐색 (Forward Pass)
     * Backward Pass에서 만든 지도 (K, k_ff)를 따라가며
     * 실제 제어 입력 (du)과 예상 궤적 (dx)을 산출합시다.
     */
    void forward_pass(const StaticVector<double, Nx>& dx_0) {
        dx[0] = dx_0;  // 초기 상태 오차
        for (size_t k = 0; k < Np; ++k) {
            auto& s = stages[k];

            // u_k = K_k * x_k + k_ff
            du[k] = (s.K * dx[k]) + s.k_ff;

            // x_{k+1} = A_k * x_k + B_k * u_k + d_k
            dx[k + 1] = (s.A * dx[k]) + (s.B * du[k]) + s.d;
        }
    }

    /**
     * @brief 전체 해법 실행 (O(Np) Complexity)
     */
    bool solve(const StaticVector<double, Nx>& dx_0) {
        if (!backward_pass()) {
            return false;  // H_uu 행렬 특이성 발생 (Infeasible)
        }
        forward_pass(dx_0);
        return true;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_BLOCK_BANDED_SOLVER_HPP_