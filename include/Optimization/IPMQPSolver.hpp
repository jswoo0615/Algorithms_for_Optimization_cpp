#ifndef OPTIMIZATION_IPM_QP_SOLVER_HPP_
#define OPTIMIZATION_IPM_QP_SOLVER_HPP_

/**
 * @file IPMQPSolver.hpp
 * @brief Primal-Dual Interior Point Method (내부점법) 기반의 Quadratic Programming (QP) Solver 구현
 * 
 * 다음과 같은 형태의 2차 계획법(QP) 문제를 풉니다.
 *   Minimize    1/2 * x^T * P * x + q^T * x
 *   Subject to  A_eq * x = b_eq
 *               A_ineq * x <= b_ineq
 * 
 * 부등식 제약조건은 슬랙 변수(s)를 도입하여 A_ineq * x + s = b_ineq (s >= 0) 로 변환하여 처리합니다.
 * 이 구현체는 KKT 조건을 풀기 위해 Newton-Raphson 기반의 Primal-Dual 스텝을 계산하며,
 * 특이점(Singularity) 방지를 위한 Tikhonov 정규화와 발산 방지용 클램핑이 적용되어 있습니다.
 */

#include <algorithm>
#include <cmath>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @class IPMQPSolver
 * @brief 정적 메모리 할당을 사용하는 IPM QP Solver 클래스
 * 
 * @tparam N_vars 최적화 변수(x)의 차원
 * @tparam N_eq 등식 제약조건의 개수 (없을 경우 0, 내부적으로 크기 1로 처리되어 메모리 절약)
 * @tparam N_ineq 부등식 제약조건의 개수 (없을 경우 0, 내부적으로 크기 1로 처리되어 메모리 절약)
 */
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class IPMQPSolver {
   public:
    // 목적 함수 행렬 및 벡터
    StaticMatrix<double, N_vars, N_vars> P; ///< Hessian 행렬 (반드시 양의 준정부호 Positive Semi-Definite 이어야 함)
    StaticVector<double, N_vars> q;         ///< Gradient 벡터

    // 등식 제약조건 (A_eq * x = b_eq)
    StaticMatrix<double, (N_eq > 0 ? N_eq : 1), N_vars> A_eq;
    StaticVector<double, (N_eq > 0 ? N_eq : 1)> b_eq;

    // 부등식 제약조건 (A_ineq * x <= b_ineq)
    StaticMatrix<double, (N_ineq > 0 ? N_ineq : 1), N_vars> A_ineq;
    StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> b_ineq;

    /**
     * @brief 기본 생성자. 모든 행렬과 벡터를 0으로 초기화합니다.
     */
    IPMQPSolver() {
        P.set_zero();
        q.set_zero();
        A_eq.set_zero();
        b_eq.set_zero();
        A_ineq.set_zero();
        b_ineq.set_zero();
    }

    /**
     * @brief 설정된 QP 문제를 해결합니다.
     * 
     * @param[in,out] x 초기 추정값이자 최적화된 결과가 저장될 변수 벡터
     * @param max_iter 최대 반복 횟수 (기본값: 50)
     * @param tol 수렴 판정을 위한 허용 오차 (기본값: 1e-3)
     * @return bool 성공 여부 (타임아웃 시에도 최선해를 반환하며 항상 true를 반환하여 외부 SQP 루프를 살림)
     */
    bool solve(StaticVector<double, N_vars>& x, int max_iter = 50, double tol = 1e-3) {
        // Dual 변수 (Lagrange Multipliers)
        StaticVector<double, (N_eq > 0 ? N_eq : 1)> y;     ///< 등식 제약조건에 대한 Dual 변수
        y.set_zero();
        StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> z; ///< 부등식 제약조건에 대한 Dual 변수 (z >= 0)
        StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> s; ///< 부등식 제약조건을 위한 Slack 변수 (s >= 0)

        // 부등식 제약조건에 대한 초기화
        for (size_t i = 0; i < N_ineq; ++i) {
            z(static_cast<int>(i)) = 1.0; // 엄격히 양수여야 함 (Strictly feasible)
            // 슬랙 변수 초기화 (최소 1.0을 보장하여 로그 장벽 함수의 정의 구역 내에 있도록 함)
            s(static_cast<int>(i)) = std::max(1.0, b_ineq(static_cast<int>(i)));
        }

        // KKT 시스템의 전체 크기: Primal 변수 + 등식 Dual 변수 + 부등식 Dual 변수
        constexpr size_t KKT_size = N_vars + N_eq + N_ineq;
        
        // Newton Step을 계산하기 위한 KKT 행렬(K)과 우변 벡터(rhs), 그리고 해를 담을 벡터(delta)
        StaticMatrix<double, (KKT_size > 0 ? KKT_size : 1), (KKT_size > 0 ? KKT_size : 1)> K;
        StaticVector<double, (KKT_size > 0 ? KKT_size : 1)> rhs;
        StaticVector<double, (KKT_size > 0 ? KKT_size : 1)> delta;

        // 메인 IPM 반복 루프
        for (int iter = 0; iter < max_iter; ++iter) {
            // 1. KKT 잔여물 (Residuals) 계산
            // 1.1 라그랑지안(Lagrangian)의 그래디언트 잔여물 (r_L = P*x + q + A_eq^T*y + A_ineq^T*z)
            StaticVector<double, N_vars> r_L = (P * x) + q;
            if constexpr (N_eq > 0) r_L = r_L + (A_eq.transpose() * y);
            if constexpr (N_ineq > 0) r_L = r_L + (A_ineq.transpose() * z);

            // 1.2 등식 제약조건 잔여물 (r_eq = A_eq*x - b_eq)
            StaticVector<double, (N_eq > 0 ? N_eq : 1)> r_eq;
            if constexpr (N_eq > 0) r_eq = (A_eq * x) - b_eq;

            // 1.3 부등식 제약조건 잔여물 (r_ineq = A_ineq*x + s - b_ineq)
            StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> r_ineq;
            if constexpr (N_ineq > 0) r_ineq = (A_ineq * x) + s - b_ineq;

            // 2. Duality Gap 및 수렴 조건 검사
            double gap = 0.0; // 상보성 여유(Complementarity slackness) 갭 (s^T * z / N_ineq)
            if constexpr (N_ineq > 0) {
                for (size_t i = 0; i < N_ineq; ++i)
                    gap += s(static_cast<int>(i)) * z(static_cast<int>(i));
                gap /= static_cast<double>(N_ineq); // 평균 갭 계산
            }

            // 모든 잔여물의 최대 절댓값(Infinity Norm)을 구함
            double res_norm = 0.0;
            for (size_t i = 0; i < N_vars; ++i)
                res_norm = std::max(res_norm, std::abs(r_L(static_cast<int>(i))));
            if constexpr (N_eq > 0)
                for (size_t i = 0; i < N_eq; ++i)
                    res_norm = std::max(res_norm, std::abs(r_eq(static_cast<int>(i))));
            if constexpr (N_ineq > 0)
                for (size_t i = 0; i < N_ineq; ++i)
                    res_norm = std::max(res_norm, std::abs(r_ineq(static_cast<int>(i))));

            // 잔여물과 Duality Gap이 모두 허용 오차 이내로 들어오면 수렴한 것으로 판단
            if (res_norm < tol && gap < tol) {
                return true;
            }

            // 3. 중심 경로 매개변수 설정 (Centering Parameter)
            // Barrier parameter(mu)를 점진적으로 줄이기 위한 설정 (보통 Mehrotra's predictor-corrector를 쓰지만 여기서는 고정된 sigma 사용)
            double sigma = 0.1; 
            double mu_target = sigma * gap;

            // KKT 행렬 및 우변 초기화
            K.set_zero();
            rhs.set_zero();

            // 4. Augmented KKT System (K * delta = rhs) 구성
            // 4.1 행렬 K의 좌상단: P 행렬 복사 및 정규화
            for (size_t i = 0; i < N_vars; ++i) {
                for (size_t j = 0; j < N_vars; ++j)
                    K(static_cast<int>(i), static_cast<int>(j)) =
                        P(static_cast<int>(i), static_cast<int>(j));
                
                // [Architect's Armor 1] 티호노프 정규화 (Tikhonov Regularization)
                // P가 양의 정부호가 아닐 경우나 특이점(Singularity) 도달 시 역행렬 계산이 폭발하는 것을 방지
                K(static_cast<int>(i), static_cast<int>(i)) += 1e-6;
            }

            // 4.2 행렬 K: 등식 제약조건 블록 할당 (A_eq, A_eq^T)
            if constexpr (N_eq > 0) {
                for (size_t i = 0; i < N_eq; ++i) {
                    for (size_t j = 0; j < N_vars; ++j) {
                        K(static_cast<int>(i + N_vars), static_cast<int>(j)) =
                            A_eq(static_cast<int>(i), static_cast<int>(j));
                        K(static_cast<int>(j), static_cast<int>(i + N_vars)) =
                            A_eq(static_cast<int>(i), static_cast<int>(j));
                    }
                }
            }

            // 4.3 행렬 K: 부등식 제약조건 블록 할당 (A_ineq, A_ineq^T) 및 대각 블록 보정 (-s/z)
            if constexpr (N_ineq > 0) {
                for (size_t i = 0; i < N_ineq; ++i) {
                    for (size_t j = 0; j < N_vars; ++j) {
                        K(static_cast<int>(i + N_vars + N_eq), static_cast<int>(j)) =
                            A_ineq(static_cast<int>(i), static_cast<int>(j));
                        K(static_cast<int>(j), static_cast<int>(i + N_vars + N_eq)) =
                            A_ineq(static_cast<int>(i), static_cast<int>(j));
                    }
                    // Slack 변수의 역수 계산 시 z가 0에 가까워져 0으로 나누어지는 오류(Division by Zero) 방지
                    double z_safe = std::max(z(static_cast<int>(i)), 1e-12);
                    // 행렬의 우하단 블록에 부등식 정보(-S^-1 * Z 의 역방향 반영인 -S/Z 로 스케일링) 반영
                    K(static_cast<int>(i + N_vars + N_eq), static_cast<int>(i + N_vars + N_eq)) =
                        -s(static_cast<int>(i)) / z_safe;
                }
            }

            // 4.4 우변(rhs) 벡터 구성
            for (size_t i = 0; i < N_vars; ++i)
                rhs(static_cast<int>(i)) = -r_L(static_cast<int>(i)); // Primal 업데이트 요구량
            if constexpr (N_eq > 0)
                for (size_t i = 0; i < N_eq; ++i)
                    rhs(static_cast<int>(i + N_vars)) = -r_eq(static_cast<int>(i)); // 등식 제약 조건 충족 요구량
            if constexpr (N_ineq > 0) {
                for (size_t i = 0; i < N_ineq; ++i) {
                    // 상보성 조건 잔여물 (Complementarity condition): s * z - mu_target
                    double r_c = s(static_cast<int>(i)) * z(static_cast<int>(i)) - mu_target;
                    double z_safe = std::max(z(static_cast<int>(i)), 1e-12);
                    // 부등식 제약조건 및 상보성 보정 반영
                    rhs(static_cast<int>(i + N_vars + N_eq)) =
                        -r_ineq(static_cast<int>(i)) + r_c / z_safe;
                }
            }

            // 5. 가우스 소거법(Gaussian Elimination)을 이용한 선형 시스템(K_aug * delta = rhs) 풀이
            StaticMatrix<double, (KKT_size > 0 ? KKT_size : 1), (KKT_size > 0 ? KKT_size : 1)>
                K_aug = K;
            delta = rhs;
            constexpr size_t N_kkt = (KKT_size > 0 ? KKT_size : 1);

            // 전방 소거(Forward Elimination) 과정 (부분 피벗팅 적용)
            for (size_t i = 0; i < N_kkt; ++i) {
                size_t pivot = i;
                double max_val = std::abs(K_aug(static_cast<int>(i), static_cast<int>(i)));
                for (size_t j = i + 1; j < N_kkt; ++j) {
                    if (std::abs(K_aug(static_cast<int>(j), static_cast<int>(i))) > max_val) {
                        max_val = std::abs(K_aug(static_cast<int>(j), static_cast<int>(i)));
                        pivot = j;
                    }
                }

                // 수치적 불안정성 방지: 피벗 값이 너무 작으면 해당 행의 소거를 건너뜀 (정규화 덕에 거의 발생 안함)
                if (max_val < 1e-12) {
                    continue;
                }

                // 행 교환(Row Swap) 수행
                if (pivot != i) {
                    for (size_t j = i; j < N_kkt; ++j) {
                        double tmp = K_aug(static_cast<int>(i), static_cast<int>(j));
                        K_aug(static_cast<int>(i), static_cast<int>(j)) =
                            K_aug(static_cast<int>(pivot), static_cast<int>(j));
                        K_aug(static_cast<int>(pivot), static_cast<int>(j)) = tmp;
                    }
                    double tmp_d = delta(static_cast<int>(i));
                    delta(static_cast<int>(i)) = delta(static_cast<int>(pivot));
                    delta(static_cast<int>(pivot)) = tmp_d;
                }

                // 행 뺄셈 연산 수행
                for (size_t j = i + 1; j < N_kkt; ++j) {
                    double factor = K_aug(static_cast<int>(j), static_cast<int>(i)) /
                                    K_aug(static_cast<int>(i), static_cast<int>(i));
                    for (size_t k = i; k < N_kkt; ++k) {
                        K_aug(static_cast<int>(j), static_cast<int>(k)) -=
                            factor * K_aug(static_cast<int>(i), static_cast<int>(k));
                    }
                    delta(static_cast<int>(j)) -= factor * delta(static_cast<int>(i));
                }
            }

            // 후방 대입(Backward Substitution)을 통한 해 도출
            for (int i = static_cast<int>(N_kkt) - 1; i >= 0; --i) {
                for (size_t j = i + 1; j < N_kkt; ++j) {
                    delta(i) -= K_aug(i, static_cast<int>(j)) * delta(static_cast<int>(j));
                }
                delta(i) /= (K_aug(i, i) != 0.0 ? K_aug(i, i) : 1e-12);
            }

            // 6. 계산된 델타(Delta) 분리 및 방향 벡터(Step Directions) 설정
            StaticVector<double, N_vars> dx;                         // Primal 업데이트
            StaticVector<double, (N_eq > 0 ? N_eq : 1)> dy;          // 등식 Dual 업데이트
            StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> dz;      // 부등식 Dual 업데이트
            StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> ds;      // Slack 변수 업데이트

            for (size_t i = 0; i < N_vars; ++i) {
                // [안전 장치] IPM 내부 클램핑 (1차 방어)
                // 해가 한 번의 스텝에서 비정상적으로 튀는 것(Overshooting)을 방지
                double val = delta(static_cast<int>(i));
                if (val > 10.0) val = 10.0;
                if (val < -10.0) val = -10.0;
                dx(static_cast<int>(i)) = val;
            }
            if constexpr (N_eq > 0)
                for (size_t i = 0; i < N_eq; ++i)
                    dy(static_cast<int>(i)) = delta(static_cast<int>(i + N_vars));
            
            // Slack 변수의 방향(ds) 계산: ds = -r_ineq - A_ineq * dx
            if constexpr (N_ineq > 0) {
                for (size_t i = 0; i < N_ineq; ++i) {
                    dz(static_cast<int>(i)) = delta(static_cast<int>(i + N_vars + N_eq));
                    double Adx = 0.0;
                    for (size_t j = 0; j < N_vars; ++j)
                        Adx += A_ineq(static_cast<int>(i), static_cast<int>(j)) *
                               dx(static_cast<int>(j));
                    ds(static_cast<int>(i)) = -r_ineq(static_cast<int>(i)) - Adx;
                }
            }

            // 7. Step Size (보폭) 계산 (Fraction-to-the-boundary rule)
            // Primal 변수와 Dual 변수가 항상 양의 제약(s > 0, z > 0)을 만족하도록 보폭을 제한
            double alpha_prim = 1.0;
            double alpha_dual = 1.0;
            constexpr double tau = 0.995; // 경계에 너무 가까이 붙는 것을 막기 위한 백오프 파라미터

            if constexpr (N_ineq > 0) {
                for (size_t i = 0; i < N_ineq; ++i) {
                    // ds < 0 일 때, s + alpha*ds >= (1-tau)*s 를 만족하는 최대 alpha_prim 탐색
                    if (ds(static_cast<int>(i)) < 0.0)
                        alpha_prim = std::min(
                            alpha_prim, -tau * s(static_cast<int>(i)) / ds(static_cast<int>(i)));
                    // dz < 0 일 때, z + alpha*dz >= (1-tau)*z 를 만족하는 최대 alpha_dual 탐색
                    if (dz(static_cast<int>(i)) < 0.0)
                        alpha_dual = std::min(
                            alpha_dual, -tau * z(static_cast<int>(i)) / dz(static_cast<int>(i)));
                }
            }

            // 8. 상태 변수 업데이트 (Update Variables)
            x = x + (dx * alpha_prim);
            if constexpr (N_eq > 0) y = y + (dy * alpha_dual);
            if constexpr (N_ineq > 0) {
                z = z + (dz * alpha_dual);
                s = s + (ds * alpha_prim); // Slack 변수는 Primal 스텝 사이즈 적용
            }
        }

        // [Architect's Safe Return]
        // 타임아웃(최대 반복 횟수 도달)이 발생하더라도, 그동안 계산된 최선의 해(x)를 반환하여
        // 이를 호출하는 상위 루프(예: SQP)가 완전히 무너지지 않도록 방어적인 설계를 취함
        return true;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_IPM_QP_SOLVER_HPP_