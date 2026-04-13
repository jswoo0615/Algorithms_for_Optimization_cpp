#ifndef OPTIMIZATION_BLOCK_BANDED_STORAGE_HPP_
#define OPTIMIZATION_BLOCK_BANDED_STORAGE_HPP_

#include <array>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {
/**
 * @brief NMPC 단일 스테이지 (t = k)의 데이터 블록 구조체
 * Nx : 상태 변수 갯수 (6)
 * Nu : 제어 변수 갯수 (2)
 */
template <size_t Nx, size_t Nu>
struct NMPCStorageData {
    // 1. 선형화된 시스템 동역학 (Layer 7에서 공급)
    // x_{k+1} = A_k * x_k + B_k * u_k + d_k
    StaticMatrix<double, Nx, Nx> A;
    StaticMatrix<double, Nx, Nu> B;
    StaticVector<double, Nx> d;  // 선형화 잔차 (Residual)

    // 2. 2차 비용 함수 블록 (Cost Function)
    // L_k = 0.5 * (x^T Q x + u^T R u) + q^T x + r^T u
    StaticMatrix<double, Nx, Nx> Q;
    StaticMatrix<double, Nu, Nu> R;
    StaticVector<double, Nx> q;
    StaticVector<double, Nu> r;

    // 3. Riccati 연산 결과 저장소 (Solver가 채움)
    // Value Function V_k(x) = 0.5 * x^T P x + p^T x
    StaticMatrix<double, Nx, Nx> P;
    StaticVector<double, Nx> p_vec;

    // Control Law : u_k = K * x_k + k_ff
    StaticMatrix<double, Nu, Nx> K;  // 피드백 게인 (Feedback Gain)
    StaticVector<double, Nu> k_ff;   // 피드포워드 게인 (feedforward)

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