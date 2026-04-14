#ifndef OPTIMIZATION_BLOCK_BANDED_STORAGE_HPP_
#define OPTIMIZATION_BLOCK_BANDED_STORAGE_HPP_

#include <array>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @brief NMPC (Nonlinear Model Predictive Control) 최적화 문제에서 단일 예측 스텝 (t = k)의 데이터를 담는 블록 구조체.
 * 
 * LQR/Riccati 방정식 기반의 솔버에서 각 시점(stage)마다 필요한 선형화된 시스템 동역학,
 * 2차 비용 함수의 가중치, 그리고 최적화 수행 후 얻어지는 제어 이득(gain) 행렬들을 한 곳에 모아 관리합니다.
 * 이 구조체는 메모리를 연속적이고 정적으로 할당하기 위해 동적 할당 없이 배열(StaticMatrix/StaticVector)만을 사용합니다.
 * 
 * @tparam Nx 상태 변수(State variables)의 개수 (예: 위치, 속도, 각도 등)
 * @tparam Nu 제어 변수(Control inputs)의 개수 (예: 조향각, 가속도 등)
 */
template <size_t Nx, size_t Nu>
struct NMPCStorageData {
    // =====================================================================
    // 1. 선형화된 시스템 동역학 모델 (Linearized System Dynamics)
    // 비선형 동역학을 기준 궤적 주변에서 테일러 전개하여 얻은 이산화(Discrete-time) 선형 모델:
    // x_{k+1} = A_k * x_k + B_k * u_k + d_k
    // =====================================================================
    
    /** @brief 상태 행렬 (State Transition Matrix). 이전 상태 x_k 가 다음 상태 x_{k+1} 에 미치는 영향을 나타냄 */
    StaticMatrix<double, Nx, Nx> A;
    
    /** @brief 제어 입력 행렬 (Control Input Matrix). 현재 제어 입력 u_k 가 다음 상태 x_{k+1} 에 미치는 영향을 나타냄 */
    StaticMatrix<double, Nx, Nu> B;
    
    /** @brief 비선형 시스템의 선형화 과정에서 발생하는 영점 잔차 또는 외란 (Linearization residual/Affine term) */
    StaticVector<double, Nx> d;


    // =====================================================================
    // 2. 2차 비용 함수 블록 (Quadratic Cost Function)
    // 시점 k에서 발생하는 단일 스테이지 비용(Stage Cost):
    // L_k = 0.5 * (x_k^T * Q_k * x_k + u_k^T * R_k * u_k) + q_k^T * x_k + r_k^T * u_k
    // =====================================================================
    
    /** @brief 상태 변수에 대한 2차 가중치 행렬 (State Cost Weight Matrix). 주로 양의 준정부호(Positive Semi-Definite) 사용 */
    StaticMatrix<double, Nx, Nx> Q;
    
    /** @brief 제어 입력에 대한 2차 가중치 행렬 (Control Cost Weight Matrix). 주로 양의 정부호(Positive Definite) 사용 */
    StaticMatrix<double, Nu, Nu> R;
    
    /** @brief 상태 변수에 대한 1차 선형 비용 벡터 (Linear State Cost). 목표 궤적 추종 시 추종 오차를 나타내는데 사용됨 */
    StaticVector<double, Nx> q;
    
    /** @brief 제어 입력에 대한 1차 선형 비용 벡터 (Linear Control Cost) */
    StaticVector<double, Nu> r;


    // =====================================================================
    // 3. Riccati 연산 결과 저장소 (Riccati Equation Results)
    // 후진 대입(Backward Pass) 과정을 통해 계산되는 가치 함수(Value Function)의 근사치:
    // V_k(x) = 0.5 * x_k^T * P_k * x_k + p_k^T * x_k
    // =====================================================================
    
    /** @brief 가치 함수의 2차 미분(Hessian) 행렬인 Cost-to-Go 행렬. Riccati 방정식의 해 */
    StaticMatrix<double, Nx, Nx> P;
    
    /** @brief 가치 함수의 1차 미분(Gradient) 벡터인 선형 Cost-to-Go 벡터 */
    StaticVector<double, Nx> p_vec;


    // =====================================================================
    // 4. 제어 법칙 (Control Law)
    // 최적 제어 입력은 상태 변수에 대한 아핀 피드백 형태(Affine Feedback)로 계산됨:
    // u_k = K_k * x_k + k_ff_k
    // =====================================================================
    
    /** @brief 피드백 제어 이득 행렬 (Feedback Gain Matrix). 상태 오차를 보정하여 최적 제어를 수행하게 함 */
    StaticMatrix<double, Nu, Nx> K;
    
    /** @brief 피드포워드 제어 이득 벡터 (Feedforward Gain Vector). 예측된 오차를 보상하기 위한 기본 제어 입력 성분 */
    StaticVector<double, Nu> k_ff;


    /**
     * @brief 기본 생성자.
     * 
     * 생성 시 내부에 선언된 모든 정적 행렬 및 벡터의 원소를 0으로 초기화합니다.
     * 이를 통해 초기 쓰레기값으로 인한 연산 오류를 방지하고, 
     * 사용되지 않는 항목(예를 들어 일부 선형화 잔차 등)이 0으로 유지되도록 보장합니다.
     */
    NMPCStorageData() {
        A.set_zero();
        B.set_zero();
        d.set_zero();
        
        Q.set_zero();
        R.set_zero();
        q.set_zero();
        r.set_zero();
        
        P.set_zero();
        p_vec.set_zero();
        
        K.set_zero();
        k_ff.set_zero();
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_BLOCK_BANDED_STORAGE_HPP_