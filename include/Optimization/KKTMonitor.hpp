#ifndef KKT_MONITOR_HPP_
#define KKT_MONITOR_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

/**
 * @brief KKT 조건 감시 및 최적해 검증 모니터
 */
template <size_t N_vars, size_t N_cons>
class KKTMonitor {
    public:
        // 허용 오차 (Slack Penalty 도입 전 하드 제약 조건 기준)
        static constexpr double TOLERANCE = 1e-6;

        struct KKT_Metrics {
            double stationarity_error;              // ∇L 잔차 (최적성)
            double primal_feasibility_error;        // Au - b 잔차 (실현 가능성)
            bool is_optimal;                        // KKT 만족 여부
        };

        /**
         * @brief EQP 시스템의 KKT 잔차 평가
         * Lagrangian : L(u, λ) = 1/2 u^T P u + q^T u + λ^T (Au - b)
         */
        static KKT_Metrics evaluate_EQP(
            const StaticMatrix<double, N_vars, N_vars>& P,
            const StaticVector<double, N_vars>& q,
            const StaticMatrix<double, N_cons, N_vars>& A,
            const StaticVector<double, N_cons>& b,
            const StaticVector<double, N_vars>& u_opt,
            const StaticVector<double, N_cons>& lambda_opt) {
            KKT_Metrics metrics;

            // 1. Stationarity : || ∇_u L ||_inf = || P*u + q + A^T*λ ||_inf <= TOL
            StaticMatrix<double, N_vars, N_cons> A_T = A.transpose();
            StaticVector<double, N_vars> grad_L = (P * u_opt) + q + (A_T * lambda_opt);

            double stat_err = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                stat_err = std::max(stat_err, std::abs(grad_L(i, 0)));
            }
            metrics.stationarity_error = stat_err;

            // 2. Primal Feasibility: || Au - b ||_inf <= TOL
            StaticVector<double, N_cons> eq_res = (A * u_opt) - b;
            double prim_err = 0.0;
            for (size_t i = 0; i < N_cons; ++i) {
                prim_err = std::max(prim_err, std::abs(eq_res(i, 0)));
            }
            metrics.primal_feasibility_error = prim_err;

            // 판정
            metrics.is_optimal = (metrics.stationarity_error <= TOLERANCE) && (metrics.primal_feasibility_error <= TOLERANCE);

            return metrics;
        }

        static void print_metrics(const KKT_Metrics& metrics) {
            std::cout << "========== [ KKT Monitor Report ] ==========\n";
            std::cout << "[1] Stationarity (∇L) Error : " << metrics.stationarity_error << "\n";
            std::cout << "[2] Primal Feasibility Error: " << metrics.primal_feasibility_error << "\n";
            std::cout << ">>> Optimality Validated    : " << (metrics.is_optimal ? "TRUE" : "FALSE") << "\n";
            std::cout << "============================================\n\n";
        }
    
};
#endif // KKT_MONITOR_HPP_