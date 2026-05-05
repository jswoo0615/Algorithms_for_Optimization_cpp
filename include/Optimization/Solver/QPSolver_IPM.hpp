#ifndef OPTIMIZATION_QP_SOLVER_IPM_HPP_
#define OPTIMIZATION_QP_SOLVER_IPM_HPP_

#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
namespace solver {

/**
 * @brief 고속 Primal-Dual Interior-Point Method (PDIPM) QP 솔버
 * @details
 * min 0.5 * x^T * H * x + g^T * x
 * s.t. C * x <= d  (부등식 제약조건만 처리. 등식 제약조건은 Riccati 솔버가 동역학 단계에서 처리함)
 *
 * 슈어 보수(Schur Complement)를 활용하여 KKT 시스템을 압축하고,
 * 초고속 LDLT 분해를 통해 탐색 방향을 도출합니다.
 *
 * @tparam Nx 최적화 변수의 개수
 * @tparam Nc 부등식 제약조건의 개수
 */
template <size_t Nx, size_t Nc>
class QPSolver_IPM {
   public:
    static constexpr double SIGMA = 0.5;       // Centering parameter
    static constexpr double TAU = 0.995;       // Fraction-to-boundary (경계 접근 한계율)
    static constexpr double MIN_SLACK = 1e-8;  // 수치 붕괴 방지용 최소 슬랙값

    /**
     * @brief 부등식 제약조건이 포함된 QP 문제를 풉니다.
     */
    static SolverStatus solve(const StaticMatrix<double, Nx, Nx>& H,
                              const StaticVector<double, Nx>& g,
                              const StaticMatrix<double, Nc, Nx>& C,
                              const StaticVector<double, Nc>& d, StaticVector<double, Nx>& x_opt,
                              int max_iter = 20, double tol = 1e-6) {
        // 메모리 호이스팅 (Zero-Allocation)
        StaticVector<double, Nc> s, z;    // Slack(s) > 0, Dual(z) > 0
        StaticVector<double, Nc> ds, dz;  // Search directions
        StaticVector<double, Nx> dx;

        // 초기화 (Warm-start가 없다면 충분히 큰 양수로 초기화하여 내부에서 시작)
        for (size_t i = 0; i < Nc; ++i) {
            s(i) = 1.0;
            z(i) = 1.0;
        }

        StaticMatrix<double, Nx, Nx> H_sys;
        StaticVector<double, Nx> g_sys;
        StaticVector<double, Nc> r_c, r_inq;  // Complementarity & Inequality residuals
        StaticVector<double, Nx> r_d;         // Dual residual

        for (int iter = 0; iter < max_iter; ++iter) {
            // 1. KKT 잔차(Residuals) 계산
            // r_inq = C*x - d + s (= 0)
            linalg::multiply(C, x_opt, r_inq);
            r_inq.saxpy(-1.0, d);
            r_inq += s;

            // r_d = H*x + g + C^T*z (= 0)
            linalg::multiply(H, x_opt, r_d);
            r_d += g;
            StaticVector<double, Nx> CT_z;
            linalg::multiply_AT_B(C, z, CT_z);
            r_d += CT_z;

            // Duality Gap 및 Barrier parameter (mu) 계산
            double duality_gap = 0.0;
            for (size_t i = 0; i < Nc; ++i) {
                duality_gap += s(i) * z(i);
            }
            double mu = duality_gap / static_cast<double>(Nc);

            // 종료 조건 검사
            double max_res = 0.0;
            for (size_t i = 0; i < Nx; ++i)
                max_res = MathTraits<double>::max(max_res, MathTraits<double>::abs(r_d(i)));
            for (size_t i = 0; i < Nc; ++i)
                max_res = MathTraits<double>::max(max_res, MathTraits<double>::abs(r_inq(i)));

            if (max_res < tol && duality_gap < tol) {
                return SolverStatus::SUCCESS;
            }

            // Target barrier parameter
            double target_mu = SIGMA * mu;

            // r_c = s * z - target_mu (= 0)
            for (size_t i = 0; i < Nc; ++i) {
                r_c(i) = s(i) * z(i) - target_mu;
            }

            // 2. Schur Complement를 통한 KKT 시스템 압축
            // H_sys = H + C^T * (S^{-1} * Z) * C
            H_sys = H;
            StaticMatrix<double, Nc, Nx> Sigma_C = C;  // Sigma * C
            for (size_t i = 0; i < Nc; ++i) {
                double sigma_i =
                    z(i) / MathTraits<double>::max(s(i), MIN_SLACK);  // 0으로 나누기 방지
                for (size_t j = 0; j < Nx; ++j) {
                    Sigma_C(i, j) *= sigma_i;
                }
            }

            StaticMatrix<double, Nx, Nx> CTSigmaC;
            linalg::multiply_AT_B(C, Sigma_C, CTSigmaC);  // SIMD 가상 전치 내적
            H_sys += CTSigmaC;

            // g_sys = -r_d - C^T * S^{-1} * (z * r_inq - r_c)
            g_sys = r_d;
            for (size_t i = 0; i < Nx; ++i) g_sys(i) = -g_sys(i);  // g_sys = -r_d

            StaticVector<double, Nc> right_term;
            for (size_t i = 0; i < Nc; ++i) {
                right_term(i) =
                    (z(i) * r_inq(i) - r_c(i)) / MathTraits<double>::max(s(i), MIN_SLACK);
            }

            StaticVector<double, Nx> CT_right;
            linalg::multiply_AT_B(C, right_term, CT_right);

            // [Architect's Fix] operator-= 대신 하드웨어 가속 FMA(saxpy) 강제 적용
            g_sys.saxpy(-1.0, CT_right);

            // 3. 탐색 방향(Step Direction) 도출 (LDLT 가속)
            if (linalg::LDLT_decompose(H_sys) != MathStatus::SUCCESS) {
                // 수치적 붕괴 발생: 10ms 실시간 제어에서 뻗는 것을 방지
                return SolverStatus::MATH_ERROR;
            }
            linalg::LDLT_solve(H_sys, g_sys, dx);  // Zero-Allocation

            // ds = -r_inq - C*dx
            linalg::multiply(C, dx, ds);
            for (size_t i = 0; i < Nc; ++i) {
                ds(i) = -r_inq(i) - ds(i);
            }

            // dz = -S^{-1} * (r_c + Z * ds)
            for (size_t i = 0; i < Nc; ++i) {
                dz(i) = -(r_c(i) + z(i) * ds(i)) / MathTraits<double>::max(s(i), MIN_SLACK);
            }

            // 4. Fraction-to-Boundary Rule (스텝 사이즈 결정)
            double alpha_primal = 1.0;
            double alpha_dual = 1.0;

            for (size_t i = 0; i < Nc; ++i) {
                if (ds(i) < 0.0) {
                    alpha_primal = MathTraits<double>::min(alpha_primal, -TAU * s(i) / ds(i));
                }
                if (dz(i) < 0.0) {
                    alpha_dual = MathTraits<double>::min(alpha_dual, -TAU * z(i) / dz(i));
                }
            }

            // [안전장치] Stagnation 방어
            if (alpha_primal < 1e-6 || alpha_dual < 1e-6) {
                return SolverStatus::STEP_SIZE_TOO_SMALL;
            }

            // 5. 변수 업데이트
            for (size_t i = 0; i < Nx; ++i) x_opt(i) += alpha_primal * dx(i);
            for (size_t i = 0; i < Nc; ++i) {
                s(i) += alpha_primal * ds(i);
                z(i) += alpha_dual * dz(i);
            }
        }

        // RTI 기법 등에서 최대 반복 횟수에 도달하더라도 부분 수렴해(Suboptimal)를 제어기에 넘김
        return SolverStatus::SUBOPTIMAL;
    }
};

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_QP_SOLVER_IPM_HPP_