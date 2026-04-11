#ifndef OPTIMIZATION_IPM_QP_SOLVER_HPP_
#define OPTIMIZATION_IPM_QP_SOLVER_HPP_

#include "Optimization/Matrix/MatrixEngine.hpp"
#include <cmath>
#include <algorithm>

namespace Optimization {

/**
 * @brief Layer 4 - Primal-Dual Interior-Point Method (IPM) QP Solver
 * @details 
 * Minimize 0.5 * x^T P x + q^T x
 * s.t. A_eq x = b_eq
 * A_ineq x <= b_ineq  (변환: A_ineq x + s = b_ineq, s >= 0)
 */
template <size_t N_vars, size_t N_eq, size_t N_ineq>
class IPMQPSolver {
public:
    StaticMatrix<double, N_vars, N_vars> P;
    StaticVector<double, N_vars> q;
    
    StaticMatrix<double, (N_eq > 0 ? N_eq : 1), N_vars> A_eq;
    StaticVector<double, (N_eq > 0 ? N_eq : 1)> b_eq;
    
    StaticMatrix<double, (N_ineq > 0 ? N_ineq : 1), N_vars> A_ineq;
    StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> b_ineq;

    IPMQPSolver() {
        P.set_zero(); q.set_zero();
        A_eq.set_zero(); b_eq.set_zero();
        A_ineq.set_zero(); b_ineq.set_zero();
    }

    bool solve(StaticVector<double, N_vars>& x, int max_iter = 50, double tol = 1e-6) {
        // 1. 변수 초기화 (Strictly Interior)
        StaticVector<double, (N_eq > 0 ? N_eq : 1)> y; y.set_zero();
        StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> z; // Dual 변수 (z > 0)
        StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> s; // Slack 변수 (s > 0)
        
        for(size_t i = 0; i < N_ineq; ++i) {
            z(static_cast<int>(i)) = 1.0; 
            s(static_cast<int>(i)) = 1.0;
        }

        constexpr size_t KKT_size = N_vars + N_eq + N_ineq;
        StaticMatrix<double, (KKT_size > 0 ? KKT_size : 1), (KKT_size > 0 ? KKT_size : 1)> K;
        StaticVector<double, (KKT_size > 0 ? KKT_size : 1)> rhs;
        StaticVector<double, (KKT_size > 0 ? KKT_size : 1)> delta;

        for (int iter = 0; iter < max_iter; ++iter) {
            // 2. Residual(잔차) 계산
            StaticVector<double, N_vars> r_L = (P * x) + q;
            if constexpr (N_eq > 0) r_L = r_L + (A_eq.transpose() * y);
            if constexpr (N_ineq > 0) r_L = r_L + (A_ineq.transpose() * z);

            StaticVector<double, (N_eq > 0 ? N_eq : 1)> r_eq;
            if constexpr (N_eq > 0) r_eq = (A_eq * x) - b_eq;

            StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> r_ineq;
            if constexpr (N_ineq > 0) r_ineq = (A_ineq * x) + s - b_ineq;

            // 3. Duality Gap (mu) 및 수렴 검사
            double gap = 0.0;
            if constexpr (N_ineq > 0) {
                for(size_t i=0; i<N_ineq; ++i) gap += s(static_cast<int>(i)) * z(static_cast<int>(i));
                gap /= static_cast<double>(N_ineq);
            }

            double res_norm = 0.0;
            for(size_t i=0; i<N_vars; ++i) res_norm = std::max(res_norm, std::abs(r_L(static_cast<int>(i))));
            if constexpr (N_eq > 0) for(size_t i=0; i<N_eq; ++i) res_norm = std::max(res_norm, std::abs(r_eq(static_cast<int>(i))));
            if constexpr (N_ineq > 0) for(size_t i=0; i<N_ineq; ++i) res_norm = std::max(res_norm, std::abs(r_ineq(static_cast<int>(i))));

            if (res_norm < tol && gap < tol) {
                return true; 
            }

            double sigma = 0.1; 
            double mu_target = sigma * gap;

            // 4. KKT 행렬(K) 및 RHS 구성
            K.set_zero();
            rhs.set_zero();

            for(size_t i=0; i<N_vars; ++i) for(size_t j=0; j<N_vars; ++j) K(static_cast<int>(i), static_cast<int>(j)) = P(static_cast<int>(i), static_cast<int>(j));
            
            if constexpr (N_eq > 0) {
                for(size_t i=0; i<N_eq; ++i) {
                    for(size_t j=0; j<N_vars; ++j) {
                        K(static_cast<int>(i + N_vars), static_cast<int>(j)) = A_eq(static_cast<int>(i), static_cast<int>(j));
                        K(static_cast<int>(j), static_cast<int>(i + N_vars)) = A_eq(static_cast<int>(i), static_cast<int>(j));
                    }
                }
            }

            if constexpr (N_ineq > 0) {
                for(size_t i=0; i<N_ineq; ++i) {
                    for(size_t j=0; j<N_vars; ++j) {
                        K(static_cast<int>(i + N_vars + N_eq), static_cast<int>(j)) = A_ineq(static_cast<int>(i), static_cast<int>(j));
                        K(static_cast<int>(j), static_cast<int>(i + N_vars + N_eq)) = A_ineq(static_cast<int>(i), static_cast<int>(j));
                    }
                    K(static_cast<int>(i + N_vars + N_eq), static_cast<int>(i + N_vars + N_eq)) = -s(static_cast<int>(i)) / z(static_cast<int>(i));
                }
            }

            for(size_t i=0; i<N_vars; ++i) rhs(static_cast<int>(i)) = -r_L(static_cast<int>(i));
            if constexpr (N_eq > 0) for(size_t i=0; i<N_eq; ++i) rhs(static_cast<int>(i + N_vars)) = -r_eq(static_cast<int>(i));
            if constexpr (N_ineq > 0) {
                for(size_t i=0; i<N_ineq; ++i) {
                    double r_c = s(static_cast<int>(i)) * z(static_cast<int>(i)) - mu_target;
                    rhs(static_cast<int>(i + N_vars + N_eq)) = -r_ineq(static_cast<int>(i)) + r_c / z(static_cast<int>(i));
                }
            }

            // -------------------------------------------------------------------------
            // 5. 선형 방정식 풀이 (LU Factorization with Partial Pivoting 내장)
            // [Architect's Fix] Symmetric Indefinite System(KKT)의 안정적 분해
            // -------------------------------------------------------------------------
            StaticMatrix<double, (KKT_size > 0 ? KKT_size : 1), (KKT_size > 0 ? KKT_size : 1)> K_aug = K;
            delta = rhs;
            bool is_singular = false;
            constexpr size_t N_kkt = (KKT_size > 0 ? KKT_size : 1);

            for (size_t i = 0; i < N_kkt; ++i) {
                // 피벗팅 (수치 안정성 확보)
                size_t pivot = i;
                double max_val = std::abs(K_aug(static_cast<int>(i), static_cast<int>(i)));
                for (size_t j = i + 1; j < N_kkt; ++j) {
                    if (std::abs(K_aug(static_cast<int>(j), static_cast<int>(i))) > max_val) {
                        max_val = std::abs(K_aug(static_cast<int>(j), static_cast<int>(i)));
                        pivot = j;
                    }
                }
                
                if (max_val < 1e-12) {
                    is_singular = true;
                    break;
                }
                
                // 행 교환(Row Swap)
                if (pivot != i) {
                    for (size_t j = i; j < N_kkt; ++j) {
                        double tmp = K_aug(static_cast<int>(i), static_cast<int>(j));
                        K_aug(static_cast<int>(i), static_cast<int>(j)) = K_aug(static_cast<int>(pivot), static_cast<int>(j));
                        K_aug(static_cast<int>(pivot), static_cast<int>(j)) = tmp;
                    }
                    double tmp_d = delta(static_cast<int>(i));
                    delta(static_cast<int>(i)) = delta(static_cast<int>(pivot));
                    delta(static_cast<int>(pivot)) = tmp_d;
                }
                
                // 전진 소거(Forward Elimination)
                for (size_t j = i + 1; j < N_kkt; ++j) {
                    double factor = K_aug(static_cast<int>(j), static_cast<int>(i)) / K_aug(static_cast<int>(i), static_cast<int>(i));
                    for (size_t k = i; k < N_kkt; ++k) {
                        K_aug(static_cast<int>(j), static_cast<int>(k)) -= factor * K_aug(static_cast<int>(i), static_cast<int>(k));
                    }
                    delta(static_cast<int>(j)) -= factor * delta(static_cast<int>(i));
                }
            }

            if (is_singular) return false;

            // 후진 대입(Back Substitution)
            for (int i = static_cast<int>(N_kkt) - 1; i >= 0; --i) {
                for (size_t j = i + 1; j < N_kkt; ++j) {
                    delta(i) -= K_aug(i, static_cast<int>(j)) * delta(static_cast<int>(j));
                }
                delta(i) /= K_aug(i, i);
            }
            // -------------------------------------------------------------------------

            StaticVector<double, N_vars> dx;
            StaticVector<double, (N_eq > 0 ? N_eq : 1)> dy;
            StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> dz;
            StaticVector<double, (N_ineq > 0 ? N_ineq : 1)> ds;

            for(size_t i=0; i<N_vars; ++i) dx(static_cast<int>(i)) = delta(static_cast<int>(i));
            if constexpr (N_eq > 0) for(size_t i=0; i<N_eq; ++i) dy(static_cast<int>(i)) = delta(static_cast<int>(i + N_vars));
            if constexpr (N_ineq > 0) {
                for(size_t i=0; i<N_ineq; ++i) {
                    dz(static_cast<int>(i)) = delta(static_cast<int>(i + N_vars + N_eq));
                    double Adx = 0.0;
                    for(size_t j=0; j<N_vars; ++j) Adx += A_ineq(static_cast<int>(i), static_cast<int>(j)) * dx(static_cast<int>(j));
                    ds(static_cast<int>(i)) = -r_ineq(static_cast<int>(i)) - Adx;
                }
            }

            // 6. Fraction-to-Boundary Rule (방벽을 넘지 않도록 보폭 제어)
            double alpha_prim = 1.0;
            double alpha_dual = 1.0;
            constexpr double tau = 0.995; 

            if constexpr (N_ineq > 0) {
                for(size_t i=0; i<N_ineq; ++i) {
                    if (ds(static_cast<int>(i)) < 0.0) alpha_prim = std::min(alpha_prim, -tau * s(static_cast<int>(i)) / ds(static_cast<int>(i)));
                    if (dz(static_cast<int>(i)) < 0.0) alpha_dual = std::min(alpha_dual, -tau * z(static_cast<int>(i)) / dz(static_cast<int>(i)));
                }
            }

            // 7. 상태 업데이트
            x = x + (dx * alpha_prim);
            if constexpr (N_eq > 0) y = y + (dy * alpha_dual);
            if constexpr (N_ineq > 0) {
                z = z + (dz * alpha_dual);
                s = s + (ds * alpha_prim);
            }
        }
        return false; 
    }
};

} // namespace Optimization

#endif // OPTIMIZATION_IPM_QP_SOLVER_HPP_