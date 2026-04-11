#ifndef OPTIMIZATION_SQP_SOLVER_HPP_
#define OPTIMIZATION_SQP_SOLVER_HPP_

#include "Optimization/AutoDiff.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"
// [Architect's Upgrade] Active-Set 폐기, IPM 솔버 장착
// 순차적 이차 계획법(Sequential Quadratic Programming, SQP)의 하위 문제로
// 내부 점 기법(Interior Point Method, IPM)을 사용하는 QP(Quadratic Programming) 솔버를 채택합니다.
#include <algorithm>
#include <cmath>

#include "Optimization/IPMQPSolver.hpp"

namespace Optimization {

/**
 * @brief SQPSolver 클래스
 *
 * 순차적 이차 계획법(Sequential Quadratic Programming)을 구현한 비선형 최적화 솔버입니다.
 * 비선형 목적 함수와 비선형 제약 조건(등식 및 부등식)을 가진 문제를 해결하는 데 사용됩니다.
 * 목적 함수는 2차 근사(Hessian 근사)를 수행하고, 제약 조건은 1차 근사(Jacobian)를 수행하여
 * 매 반복마다 2차 계획법(QP) 하위 문제를 생성하고 IPM(Interior Point Method) 솔버를 통해 해를
 * 찾습니다.
 *
 * @tparam N_vars 최적화 변수의 개수
 * @tparam N_eq 등식 제약 조건의 개수
 * @tparam N_ineq 부등식 제약 조건의 개수
 */
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class SQPSolver {
   public:
    // 하부 구조: 정적 메모리 기반 Primal-Dual IPM(Interior Point Method) 엔진
    // 매 SQP 반복(iteration)마다 발생하는 QP 하위 문제를 풀기 위해 사용됩니다.
    IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver;

    // 헤시안 근사 행렬 (Hessian Approximation Matrix)
    // 목적 함수의 라그랑지안에 대한 2차 미분(헤시안) 정보를 담고 있으며,
    // BFGS(Broyden-Fletcher-Goldfarb-Shanno) 알고리즘을 통해 매 반복마다 점진적으로 업데이트됩니다.
    StaticMatrix<double, N_vars, N_vars> H;

    /**
     * @brief SQPSolver 생성자
     *
     * 헤시안 근사 행렬 H를 단위 행렬(Identity Matrix)로 초기화합니다.
     * 이는 BFGS 업데이트의 초기값으로 널리 사용되는 안정적인 방식입니다.
     */
    SQPSolver() {
        H.set_zero();
        for (size_t i = 0; i < N_vars; ++i) {
            H(static_cast<int>(i), static_cast<int>(i)) = 1.0;
        }
    }

    /**
     * @brief 최적화 문제를 푸는 메인 함수
     *
     * @tparam CostFunc 목적 함수 타입
     * @tparam EqFunc 등식 제약 조건 함수 타입
     * @tparam IneqFunc 부등식 제약 조건 함수 타입
     * @param u 초기 추정값 (입력)이자 최적화된 해 (출력)
     * @param cost_f 목적 함수 (최소화 대상)
     * @param eq_f 등식 제약 조건 함수 (eq_f(u) == 0)
     * @param ineq_f 부등식 제약 조건 함수 (ineq_f(u) <= 0)
     * @param max_iter 최대 반복 횟수 (기본값: 50)
     * @return bool 최적화 성공 여부 (현재 버전에서는 수렴 조건 충족 시 true 반환, 루프 종료 시
     * false)
     */
    template <typename CostFunc, typename EqFunc, typename IneqFunc>
    bool solve(StaticVector<double, N_vars>& u, CostFunc cost_f, EqFunc eq_f, IneqFunc ineq_f,
               int max_iter = 50) {
        for (int iter = 0; iter < max_iter; ++iter) {
            // 1. 현재 상태에서 목적 함수의 값과 기울기(Gradient) 계산
            double cost_val = 0.0;
            StaticVector<double, N_vars> grad_f;
            // AutoDiff(자동 미분)를 사용하여 목적 함수의 값과 기울기를 평가합니다.
            AutoDiff::value_and_gradient<N_vars>(cost_f, u, cost_val, grad_f);

            // 2. 등식 제약 조건 선형화 (Linearization of Equality Constraints)
            // 테일러 1차 전개를 통해 등식 제약을 선형 제약으로 변환합니다.
            // c(u + p) ~ c(u) + J_eq * p = 0 -> J_eq * p = -c(u)
            if constexpr (N_eq > 0) {
                StaticVector<double, N_eq> eq_val =
                    eq_f(u);  // 현재 위치에서의 등식 제약 조건 값 평가
                StaticMatrix<double, N_eq, N_vars> J_eq =
                    AutoDiff::jacobian<N_eq, N_vars>(eq_f, u);  // 야코비안 계산
                for (size_t i = 0; i < N_eq; ++i) {
                    qp_solver.b_eq(static_cast<int>(i)) =
                        -eq_val(static_cast<int>(i));  // 우변: -c(u)
                    for (size_t j = 0; j < N_vars; ++j) {
                        qp_solver.A_eq(static_cast<int>(i), static_cast<int>(j)) =
                            J_eq(static_cast<int>(i), static_cast<int>(j));  // 좌변: Jacobian 행렬
                    }
                }
            }

            // 3. 부등식 제약 조건 선형화 (Linearization of Inequality Constraints)
            // 테일러 1차 전개를 통해 부등식 제약을 선형 제약으로 변환합니다.
            // g(u + p) ~ g(u) + J_ineq * p <= 0 -> J_ineq * p <= -g(u)
            if constexpr (N_ineq > 0) {
                StaticVector<double, N_ineq> ineq_val =
                    ineq_f(u);  // 현재 위치에서의 부등식 제약 조건 값 평가
                StaticMatrix<double, N_ineq, N_vars> J_ineq =
                    AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);  // 야코비안 계산
                for (size_t i = 0; i < N_ineq; ++i) {
                    qp_solver.b_ineq(static_cast<int>(i)) =
                        -ineq_val(static_cast<int>(i));  // 우변: -g(u)
                    for (size_t j = 0; j < N_vars; ++j) {
                        qp_solver.A_ineq(static_cast<int>(i), static_cast<int>(j)) = J_ineq(
                            static_cast<int>(i), static_cast<int>(j));  // 좌변: Jacobian 행렬
                    }
                }
            }

            // 4. QP 하위 문제 설정 및 풀이
            // QP 목적 함수: 1/2 p^T * H * p + grad_f^T * p
            qp_solver.P = H;       // 이차항 계수 (헤시안 근사)
            qp_solver.q = grad_f;  // 일차항 계수 (목적 함수의 기울기)

            StaticVector<double, N_vars> p;
            p.set_zero();

            // IPM 솔버 가동 (톨러런스를 1e-4로 설정하여 실시간성 확보)
            // QP 솔버가 성공적으로 탐색 방향 p를 찾지 못한 경우에 대한 Fallback 메커니즘
            if (!qp_solver.solve(p, 50, 1e-4)) {
                // QP 풀이에 실패하면, 단순 경사 하강법(Gradient Descent) 방향으로 약간
                // 이동시킵니다.
                p = grad_f * -0.05;
            }

            // 구한 탐색 방향(p)에 대한 안정성 및 제약 처리
            double p_norm = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                // NaN 또는 Inf 값이 발생한 경우 0으로 초기화하여 발산 방지
                if (std::isnan(p(static_cast<int>(i))) || std::isinf(p(static_cast<int>(i)))) {
                    p(static_cast<int>(i)) = 0.0;
                }
                // 한 제어 주기당 변화할 수 있는 물리적 최대 한계치 클리핑 (예: 가속도 +-3.0, 조향
                // +-3.0 등) 너무 큰 step이 발생하여 시스템이 불안정해지는 것을 막습니다.
                if (p(static_cast<int>(i)) > 3.0) p(static_cast<int>(i)) = 3.0;
                if (p(static_cast<int>(i)) < -3.0) p(static_cast<int>(i)) = -3.0;

                // 탐색 방향 벡터 p의 크기(L-infinity norm) 계산
                p_norm = std::max(p_norm, std::abs(p(static_cast<int>(i))));
            }

            // 5. 수렴 조건 검사 (Convergence Check)
            // 스텝 크기가 1e-6보다 작으면 더 이상 유의미한 변화가 없다고 판단하고 최적해 도달로
            // 간주합니다.
            if (p_norm < 1e-6) return true;

            // 6. 라인 서치 (Line Search) 기반 스텝 크기 결정
            // L1 Merit Function을 사용하여 목적 함수 값의 감소와 제약 조건 위반도 감소의 균형을
            // 맞춥니다.
            double alpha = 1.0;  // 초기 스텝 사이즈 (1.0 = Newton step)
            const double rho = 10.0;  // 패널티 파라미터 (제약 조건 위반에 대한 벌점 가중치)

            // 현재 위치에서의 Merit 함수 값 계산
            double current_merit = cost_val;
            if constexpr (N_eq > 0) {
                StaticVector<double, N_eq> v = eq_f(u);
                for (size_t i = 0; i < N_eq; ++i)
                    current_merit += rho * std::abs(v(static_cast<int>(i)));  // 등식 제약 위반: |v|
            }
            if constexpr (N_ineq > 0) {
                StaticVector<double, N_ineq> v = ineq_f(u);
                for (size_t i = 0; i < N_ineq; ++i)
                    current_merit +=
                        rho * std::max(0.0, v(static_cast<int>(i)));  // 부등식 제약 위반: max(0, v)
            }

            // 백트래킹 라인 서치 (Backtracking Line Search) 수행
            StaticVector<double, N_vars> u_next;
            for (int ls_iter = 0; ls_iter < 10; ++ls_iter) {
                // 다음 시도 위치: u_next = u + alpha * p
                for (size_t i = 0; i < N_vars; ++i)
                    u_next(static_cast<int>(i)) =
                        u(static_cast<int>(i)) + alpha * p(static_cast<int>(i));

                // 시도 위치에서의 Merit 함수 값 평가
                double next_cost = cost_f(u_next);
                double next_merit = next_cost;

                if constexpr (N_eq > 0) {
                    StaticVector<double, N_eq> v = eq_f(u_next);
                    for (size_t i = 0; i < N_eq; ++i)
                        next_merit += rho * std::abs(v(static_cast<int>(i)));
                }
                if constexpr (N_ineq > 0) {
                    StaticVector<double, N_ineq> v = ineq_f(u_next);
                    for (size_t i = 0; i < N_ineq; ++i)
                        next_merit += rho * std::max(0.0, v(static_cast<int>(i)));
                }

                // Armijo 조건과 유사하게, Merit 함수 값이 감소했으면 적절한 스텝 사이즈로 간주하고
                // 루프 종료
                if (next_merit < current_merit) break;

                // 감소하지 않았다면 스텝 사이즈를 절반으로 줄이고 다시 시도
                alpha *= 0.5;
            }

            // 7. Damped BFGS Update (헤시안 근사 행렬 업데이트)
            // 새로운 위치(u_next)에서의 목적 함수 기울기 계산
            StaticVector<double, N_vars> next_grad = AutoDiff::gradient<N_vars>(cost_f, u_next);
            StaticVector<double, N_vars> s;  // 변위 벡터: s = u_next - u
            StaticVector<double, N_vars> y;  // 기울기 변화 벡터: y = grad(u_next) - grad(u)
            double s_norm_sq = 0.0;

            for (size_t i = 0; i < N_vars; ++i) {
                int idx = static_cast<int>(i);
                s(idx) = u_next(idx) - u(idx);
                y(idx) = next_grad(idx) - grad_f(idx);
                s_norm_sq += s(idx) * s(idx);  // ||s||^2
            }

            // 변위가 유의미할 때만(1e-12보다 클 때) 헤시안을 업데이트하여 수치적 불안정성 회피
            if (s_norm_sq > 1e-12) {
                StaticVector<double, N_vars> Hs;
                Hs.set_zero();
                double sHs = 0.0;  // s^T * H * s
                double ys = 0.0;   // y^T * s

                for (size_t i = 0; i < N_vars; ++i) {
                    int r = static_cast<int>(i);
                    for (size_t j = 0; j < N_vars; ++j) {
                        Hs(r) += H(r, static_cast<int>(j)) * s(static_cast<int>(j));  // H * s 계산
                    }
                    sHs += s(r) * Hs(r);
                    ys += y(r) * s(r);
                }

                // Damping factor 계산 (Powell's modification)
                // 곡률 조건(Curvature condition, y^T * s > 0)을 보장하고 헤시안 행렬 H의
                // 양의 정부호(Positive Definite) 성질을 유지하기 위해 사용하는 기법
                double theta = 1.0;
                if (ys < 0.2 * sHs) {
                    theta = (0.8 * sHs) / (sHs - ys + 1e-16);
                }

                // 수정된 y 벡터 (r_vec) 계산: r = theta * y + (1 - theta) * H * s
                StaticVector<double, N_vars> r_vec;
                double rs = 0.0;  // r_vec^T * s
                for (size_t i = 0; i < N_vars; ++i) {
                    int idx = static_cast<int>(i);
                    r_vec(idx) = theta * y(idx) + (1.0 - theta) * Hs(idx);
                    rs += r_vec(idx) * s(idx);
                }

                // BFGS 행렬 업데이트 공식 적용
                // 수치적 에러를 방지하기 위해 분모(rs, sHs)가 매우 작지 않은지 검사
                if (rs > 1e-12 && sHs > 1e-12) {
                    for (size_t i = 0; i < N_vars; ++i) {
                        for (size_t j = 0; j < N_vars; ++j) {
                            int row = static_cast<int>(i);
                            int col = static_cast<int>(j);
                            // H_new = H - (H*s * s^T*H) / (s^T*H*s) + (r * r^T) / (r^T*s)
                            H(row, col) = H(row, col) - (Hs(row) * Hs(col)) / sHs +
                                          (r_vec(row) * r_vec(col)) / rs;
                        }
                    }
                }
            }

            // 8. 변수 업데이트: 다음 반복(iteration)을 위해 u를 갱신
            u = u_next;
        }

        // 최대 반복 횟수(max_iter)에 도달했지만 수렴 조건을 만족하지 못한 경우
        return false;
    }
};

}  // namespace Optimization

#endif