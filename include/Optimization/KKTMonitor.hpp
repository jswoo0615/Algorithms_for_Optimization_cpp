#ifndef KKT_MONITOR_HPP_
#define KKT_MONITOR_HPP_

/**
 * @file KKTMonitor.hpp
 * @brief KKT 조건 감시 및 최적해 검증 모니터 구현 (Zero-Allocation Optimized)
 */

#include <algorithm>
#include <iostream>

#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"

namespace Optimization {

template <size_t N_vars, size_t N_cons>
class KKTMonitor {
   public:
    static constexpr double TOLERANCE = 1e-6;

    struct KKT_Metrics {
        double stationarity_error;        // 정류성(Stationarity) 오차: ||∇L||_inf
        double primal_feasibility_error;  // 원문제 실현가능성 오차: ||Au - b||_inf
        bool is_optimal;
    };

    /**
     * @brief EQP 시스템의 KKT 잔차를 무할당(Zero-Allocation) 방식으로 고속 평가합니다.
     */
    static KKT_Metrics evaluate_EQP(const StaticMatrix<double, N_vars, N_vars>& P,
                                    const StaticVector<double, N_vars>& q,
                                    const StaticMatrix<double, N_cons, N_vars>& A,
                                    const StaticVector<double, N_cons>& b,
                                    const StaticVector<double, N_vars>& u_opt,
                                    const StaticVector<double, N_cons>& lambda_opt) {
        KKT_Metrics metrics;

        // =========================================================================
        // 1. Stationarity (정류성) 검사
        // ∇_u L = P*u + q + A^T*λ = 0
        // =========================================================================

        // q로 초기화 (Zero-allocation 복사)
        StaticVector<double, N_vars> grad_L = q;

        // P * u 계산 후 grad_L에 가산 (SIMD In-place 연산)
        StaticVector<double, N_vars> Pu;
        linalg::multiply(P, u_opt, Pu);
        grad_L += Pu;

        // [Architect's Update] 물리적 Transpose 없이 가상 전치 내적 수행
        StaticVector<double, N_vars> AT_lambda;
        linalg::multiply_AT_B(A, lambda_opt, AT_lambda);
        grad_L += AT_lambda;  // SIMD In-place 연산

        double stat_err = 0.0;
        for (size_t i = 0; i < N_vars; ++i) {
            double abs_val = MathTraits<double>::abs(grad_L(i));
            if (abs_val > stat_err) stat_err = abs_val;
        }
        metrics.stationarity_error = stat_err;

        // =========================================================================
        // 2. Primal Feasibility (원문제 실현가능성) 검사
        // A*u - b = 0
        // =========================================================================

        StaticVector<double, N_cons> eq_res;
        linalg::multiply(A, u_opt, eq_res);  // eq_res = A * u

        // [Architect's Update] 임시 객체 없는 SIMD 뺄셈 연산
        eq_res.saxpy(-1.0, b);  // eq_res = -1.0 * b + eq_res

        double prim_err = 0.0;
        for (size_t i = 0; i < N_cons; ++i) {
            double abs_val = MathTraits<double>::abs(eq_res(i));
            if (abs_val > prim_err) prim_err = abs_val;
        }
        metrics.primal_feasibility_error = prim_err;

        // =========================================================================
        // 3. 최종 최적성 판정
        // =========================================================================
        metrics.is_optimal = (metrics.stationarity_error <= TOLERANCE) &&
                             (metrics.primal_feasibility_error <= TOLERANCE);

        return metrics;
    }

    static void print_metrics(const KKT_Metrics& metrics) {
        std::cout << "========== [ KKT Monitor Report ] ==========\n";
        std::cout << "[1] Stationarity (∇L) Error : " << metrics.stationarity_error << "\n";
        std::cout << "[2] Primal Feasibility Error: " << metrics.primal_feasibility_error << "\n";
        std::cout << ">>> Optimality Validated    : " << (metrics.is_optimal ? "TRUE" : "FALSE")
                  << "\n";
        std::cout << "============================================\n\n";
    }
};

}  // namespace Optimization

#endif  // KKT_MONITOR_HPP_