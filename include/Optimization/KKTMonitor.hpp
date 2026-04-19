#ifndef KKT_MONITOR_HPP_
#define KKT_MONITOR_HPP_

/**
 * @file KKTMonitor.hpp
 * @brief KKT(Karush-Kuhn-Tucker) 조건 감시 및 최적해 검증 모니터 구현
 * 
 * 제약 조건이 있는 최적화 문제에서 찾아낸 해가 수학적으로 올바른 최적해(Optimal Solution)인지 
 * 검증하기 위해 KKT 조건을 모니터링하고 평가하는 유틸리티 클래스입니다.
 * 주로 EQP(Equality Constrained Quadratic Programming) 솔버 등에서 도출된 결과의
 * 최적성(Optimality)과 실현 가능성(Feasibility)을 점검하는 데 사용됩니다.
 */

#include <algorithm>
#include <cmath>
#include <iostream>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @class KKTMonitor
 * @brief 제약 최적화 문제의 해가 KKT 조건을 만족하는지 검증하는 모니터 클래스
 * 
 * @tparam N_vars 최적화할 변수(Primal 변수, u)의 차원 수
 * @tparam N_cons 제약 조건(Dual 변수, lambda)의 개수
 */
template <size_t N_vars, size_t N_cons>
class KKTMonitor {
   public:
    /** 
     * @brief KKT 조건 만족 여부를 판정하기 위한 기본 허용 오차 (Tolerance)
     * Slack Penalty와 같은 소프트 제약이 도입되기 전의 순수 하드 제약 조건 기준입니다.
     */
    static constexpr double TOLERANCE = 1e-6;

    /**
     * @struct KKT_Metrics
     * @brief KKT 검증 과정에서 측정된 오차 지표들을 저장하는 구조체
     */
    struct KKT_Metrics {
        double stationarity_error;        ///< 정류성(Stationarity) 오차: ∇L 잔차의 최댓값 (최적성 기준)
        double primal_feasibility_error;  ///< 원문제 실현가능성(Primal Feasibility) 오차: 제약조건 위반 정도 (Au - b 잔차의 최댓값)
        bool is_optimal;                  ///< 허용 오차 내에서 모든 KKT 조건을 만족하는지 여부
    };

    /**
     * @brief EQP(Equality Constrained Quadratic Programming) 시스템의 KKT 잔차를 평가합니다.
     * 
     * EQP 문제는 다음과 같이 정의됩니다.
     *   Minimize    1/2 * u^T * P * u + q^T * u
     *   Subject to  A * u = b
     * 
     * 이 문제의 라그랑지안(Lagrangian) 함수는 다음과 같습니다.
     *   L(u, λ) = 1/2 * u^T * P * u + q^T * u + λ^T * (A * u - b)
     * 
     * KKT 필요 조건 (First-Order Necessary Conditions):
     * 1. 정류성(Stationarity): ∇_u L(u, λ) = P*u + q + A^T*λ = 0
     * 2. 원문제 실현가능성(Primal Feasibility): ∇_λ L(u, λ) = A*u - b = 0
     * 
     * @param P 목적 함수의 Hessian 행렬
     * @param q 목적 함수의 Gradient 벡터
     * @param A 등식 제약 조건 행렬
     * @param b 등식 제약 조건 벡터
     * @param u_opt 평가할 최적화 변수(Primal 변수)의 후보 해
     * @param lambda_opt 평가할 라그랑주 승수(Dual 변수)의 후보 해
     * @return KKT_Metrics 계산된 오차 및 최적성 판정 결과가 담긴 구조체
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
        // =========================================================================
        // 라그랑지안 함수의 그래디언트(∇_u L)가 0벡터에 얼마나 가까운지(Infinity Norm) 측정합니다.
        // 수식: || P*u + q + A^T*λ ||_inf <= TOLERANCE
        StaticMatrix<double, N_vars, N_cons> A_T = A.transpose();
        StaticVector<double, N_vars> grad_L = (P * u_opt) + q + (A_T * lambda_opt);

        double stat_err = 0.0;
        for (size_t i = 0; i < N_vars; ++i) {
            // 잔차 벡터 원소들 중 가장 큰 절댓값을 찾음 (L-infinity Norm)
            stat_err = std::max(stat_err, std::abs(grad_L(i, 0)));
        }
        metrics.stationarity_error = stat_err;

        // =========================================================================
        // 2. Primal Feasibility (원문제 실현가능성) 검사
        // =========================================================================
        // 도출된 해가 주어진 등식 제약 조건(A*u = b)을 얼마나 잘 만족하는지 측정합니다.
        // 수식: || A*u - b ||_inf <= TOLERANCE
        StaticVector<double, N_cons> eq_res = (A * u_opt) - b;
        double prim_err = 0.0;
        for (size_t i = 0; i < N_cons; ++i) {
            // 위반 정도의 가장 큰 절댓값을 찾음 (L-infinity Norm)
            prim_err = std::max(prim_err, std::abs(eq_res(i, 0)));
        }
        metrics.primal_feasibility_error = prim_err;

        // =========================================================================
        // 3. 최종 최적성(Optimality) 판정
        // =========================================================================
        // 두 가지 오차가 모두 지정된 허용 오차(TOLERANCE) 이하이면 올바른 KKT 포인트(최적해)로 인정합니다.
        metrics.is_optimal = (metrics.stationarity_error <= TOLERANCE) &&
                             (metrics.primal_feasibility_error <= TOLERANCE);

        return metrics;
    }

    /**
     * @brief 측정된 KKT 조건 지표들을 콘솔에 출력합니다.
     * 
     * @param metrics 출력할 KKT_Metrics 구조체 데이터
     */
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