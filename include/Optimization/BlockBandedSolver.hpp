#ifndef OPTIMIZATION_BLOCK_BANDED_SOLVER_HPP_
#define OPTIMIZATION_BLOCK_BANDED_SOLVER_HPP_

#include <iostream>
#include <array>

#include "Optimization/Matrix/BlockBandedStorage.hpp"

namespace Optimization {

/**
 * @brief NMPC(Nonlinear Model Predictive Control) 최적화를 위한 블록 밴드형(Block-Banded) 리카티(Riccati) 솔버 클래스.
 * 
 * LQR(Linear Quadratic Regulator) 문제의 KKT(Karush-Kuhn-Tucker) 시스템 행렬은 
 * 시간이 지남에 따라 상태와 제어 입력이 엮여있는 "블록 밴드(Block-Banded)" 구조를 가집니다.
 * 이 클래스는 시간 역순으로 진행하는 Backward Pass(리카티 재귀 연산)와 
 * 시간 순서대로 진행하는 Forward Pass(상태 및 제어 입력 업데이트)를 통해 O(Np)의 선형 시간 복잡도로 
 * 최적 제어 문제를 해결합니다. (여기서 Np는 예측 호라이즌 길이입니다.)
 * 
 * 이 솔버는 동적 메모리 할당(힙 할당)을 배제하여, 임베디드 실시간 제어(Hard Real-Time Control)에 적합합니다.
 * 
 * @tparam Nx 시스템 상태 변수(State variables)의 차원
 * @tparam Nu 시스템 제어 입력(Control inputs)의 차원
 * @tparam Np 예측 호라이즌(Prediction Horizon)의 길이 (총 스텝 수)
 */
template <size_t Nx, size_t Nu, size_t Np>
class BlockBandedSolver {
   public:
    /** 
     * @brief 호라이즌 전체의 시점(t = 0 ~ Np-1)마다 필요한 선형화 모델 및 비용 함수 데이터를 담는 배열.
     * 동적 할당 없이 std::array를 사용하여 스택(또는 BSS 섹션)에 할당됩니다.
     */
    std::array<NMPCStorageData<Nx, Nu>, Np> stages;

    // =====================================================================
    // 종단 비용 (Terminal Stage Cost) 블록
    // 호라이즌의 마지막 시점(t = Np)에 도달했을 때 적용되는 2차 비용 및 선형 비용입니다.
    // L_N(x) = 0.5 * x_N^T * P_N * x_N + p_N_vec^T * x_N
    // =====================================================================
    StaticMatrix<double, Nx, Nx> P_N;
    StaticVector<double, Nx> p_N_vec;

    // =====================================================================
    // 최적 해(Optimal Solution) 저장 배열
    // =====================================================================
    
    /** @brief 0부터 Np까지 각 시점의 상태 궤적 오차(State trajectory error, Delta x). 크기는 Np + 1 입니다. */
    std::array<StaticVector<double, Nx>, Np + 1> dx;
    
    /** @brief 0부터 Np-1까지 각 시점의 최적 제어 입력 오차(Control input error, Delta u). 크기는 Np 입니다. */
    std::array<StaticVector<double, Nu>, Np> du;

    /**
     * @brief 생성자.
     * 종단 비용을 나타내는 행렬 P_N과 벡터 p_N_vec를 0으로 초기화합니다.
     */
    BlockBandedSolver() {
        P_N.set_zero();
        p_N_vec.set_zero();
    }

    /**
     * @brief 역방향 탐색 (Backward Pass / Riccati Recursion)
     * 
     * 목적지 (t = Np)에서 시작하여 현재 시점 (t = 0)까지 시간을 역행하며 가치 함수(Value Function)의 
     * 2차 미분(Hessian) 행렬인 `P`와 1차 미분(Gradient) 벡터인 `p_vec`를 갱신합니다.
     * 이 과정을 통해 각 시점 k에서 최적의 피드백 제어 게인(K_k)과 피드포워드 제어 게인(k_ff_k)을 도출합니다.
     * 
     * @return bool 연산 중 H_uu 행렬의 역행렬을 구할 수 없는 상태(수치적 불안정성, Positive Definite 조건 위배)가 
     *         발생하여 실패하면 false, 정상적으로 역방향 탐색을 마치면 true를 반환합니다.
     */
    bool backward_pass() {
        // 1. 종단 조건 (Terminal Condition)을 초기 가치 함수의 파라미터로 설정 (V_N(x) = Terminal Cost)
        StaticMatrix<double, Nx, Nx> P_next = P_N;
        StaticVector<double, Nx> p_next = p_N_vec;

        // 2. 미래 (Np-1)부터 현재 (0)로 역행하며 동적 계획법(Dynamic Programming) 수행
        for (int k = static_cast<int>(Np) - 1; k >= 0; --k) {
            auto& s = stages[k];

            // [Step 2-1] Q 함수 (State-Action Value Function) 근사를 위한 블록 텐서(Hessian 블록) 계산
            // H_xx = Q_k + A_k^T * P_{k+1} * A_k (상태에 대한 2차 미분)
            StaticMatrix<double, Nx, Nx> H_xx = s.Q + s.A.quadratic_multiply(P_next);

            // H_uu = R_k + B_k^T * P_{k+1} * B_k (제어 입력에 대한 2차 미분)
            StaticMatrix<double, Nu, Nu> H_uu = s.R + s.B.quadratic_multiply(P_next);

            // H_ux = B_k^T * P_{k+1} * A_k (상태와 제어 입력 간의 교차 2차 미분)
            StaticMatrix<double, Nx, Nx> P_A = P_next * s.A;
            StaticMatrix<double, Nu, Nx> H_ux = s.B.transpose() * P_A;

            // [Step 2-2] Q 함수 (State-Action Value Function) 근사를 위한 1차 Gradient 벡터 계산
            // 보조 벡터: P_{k+1} * d_k + p_{k+1}
            StaticVector<double, Nx> P_d_p = (P_next * s.d) + p_next;
            
            // g_u = r_k + B_k^T * (P_{k+1} * d_k + p_{k+1}) (제어 입력에 대한 1차 미분)
            StaticVector<double, Nu> g_u = s.r + (s.B.transpose() * P_d_p);

            // g_x = q_k + A_k^T * (P_{k+1} * d_k + p_{k+1}) (상태에 대한 1차 미분)
            StaticVector<double, Nx> g_x = s.q + (s.A.transpose() * P_d_p);

            // [Step 2-3] H_uu의 역행렬 연산을 위한 LDLT 분해 (H_uu는 Positive Definite여야 함)
            if (!H_uu.LDLT_decompose()) {
                // H_uu가 양의 정부호(Positive Definite)가 아니어 역행렬을 구할 수 없는 경우 (수치적 불안정성 발생)
                // 대각 원소에 작은 양수(1e-4)를 더해주는 Damping(Tikhonov Regularization)을 적용하여 방어합니다.
                for (size_t i = 0; i < Nu; ++i) {
                    H_uu(i, i) += 1e-4;
                }
                // Damping 적용 후 다시 분해 시도. 그래도 실패하면 문제를 풀 수 없으므로 false 반환.
                if (!H_uu.LDLT_decompose()) {
                    return false;
                }
            }

            // [Step 2-4] 최적 제어 이득(Gain) 계산
            // 피드백 게인: K_k = -H_uu^{-1} * H_ux
            // (H_ux 각 열에 대해 H_uu의 역행렬을 곱하는 solve_multiple 연산 수행)
            StaticMatrix<double, Nu, Nx> neg_H_ux = H_ux * -1.0;
            s.K = H_uu.template solve_multiple<Nx>(neg_H_ux);

            // 피드포워드 게인: k_ff_k = -H_uu^{-1} * g_u
            StaticVector<double, Nu> neg_g_u;
            for (size_t i = 0; i < Nu; ++i) {
                neg_g_u(i) = -g_u(i);
            }
            s.k_ff = H_uu.LDLT_solve(neg_g_u);

            // [Step 2-5] 가치 함수(Value Function) 갱신 (P_k, p_k 계산)
            // 원래 식: P_k = H_xx + K_k^T * H_uu * K_k + K_k^T * H_ux + H_ux^T * K_k
            // 위 식은 K_k = -H_uu^{-1} * H_ux 를 대입하면 구조적으로 아래와 같이 단순화됩니다:
            // P_k = H_xx + H_ux^T * K_k
            s.P = H_xx + (H_ux.transpose() * s.K);

            // p_k = g_x + H_ux^T * k_ff_k
            s.p_vec = g_x + (H_ux.transpose() * s.k_ff);

            // 다음 시간 스텝(더 과거의 시간, k-1)의 계산을 위해 P_next와 p_next를 갱신합니다.
            P_next = s.P;
            p_next = s.p_vec;
        }
        return true;
    }

    /**
     * @brief 순방향 탐색 (Forward Pass)
     * 
     * 역방향 탐색(Backward Pass)을 통해 도출된 제어 이득(K, k_ff)이라는 '지도'를 바탕으로,
     * 실제 현재 상태 오차(dx_0)부터 시작하여 제어 입력(du)을 계산하고, 
     * 선형화된 시스템 모델을 따라 다음 상태(dx_{k+1})를 예측(시뮬레이션)합니다.
     * 
     * @param dx_0 시점 t = 0 에서의 초기 상태 오차 벡터 (현재 실제 상태 - 기준 궤적 상태)
     */
    void forward_pass(const StaticVector<double, Nx>& dx_0) {
        dx[0] = dx_0;  // 시점 t = 0 의 초기 상태 세팅
        
        // 현재 (0)부터 미래 (Np-1)까지 순행하며 궤적 생성
        for (size_t k = 0; k < Np; ++k) {
            auto& s = stages[k];

            // 최적 제어 입력 산출: u_k = K_k * x_k + k_ff_k
            du[k] = (s.K * dx[k]) + s.k_ff;

            // 다음 상태 예측 (선형화된 시스템 동역학 방정식 적용): 
            // x_{k+1} = A_k * x_k + B_k * u_k + d_k
            dx[k + 1] = (s.A * dx[k]) + (s.B * du[k]) + s.d;
        }
    }

    /**
     * @brief 전체 해법 실행 (Solve)
     * 
     * 역방향 탐색과 순방향 탐색을 순차적으로 실행하여 최종적인 최적 상태 궤적(dx)과 
     * 제어 입력(du)을 도출합니다. O(Np)의 선형 시간 복잡도로 빠르게 해를 구합니다.
     * 
     * @param dx_0 초기 시점(t=0)에서의 상태 오차 벡터
     * @return bool 성공적으로 해를 찾았으면 true, 역행렬 계산 불가 등 수치적 오류로 실패했으면 false 반환
     */
    bool solve(const StaticVector<double, Nx>& dx_0) {
        // 1. 역방향 탐색을 통해 최적 제어 이득(K, k_ff) 계산
        if (!backward_pass()) {
            // H_uu 행렬이 특이행렬(Singular)이 되거나 Damping으로도 해결 불가능한 경우 (Infeasible)
            return false;  
        }
        
        // 2. 도출된 제어 이득을 바탕으로 초기 상태 dx_0부터 순방향으로 제어 입력과 상태 궤적 시뮬레이션
        forward_pass(dx_0);
        
        return true;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_BLOCK_BANDED_SOLVER_HPP_