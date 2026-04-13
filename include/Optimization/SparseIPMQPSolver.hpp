#ifndef OPTIMIZATION_SPARSE_IPM_QP_SOLVER_HPP_
#define OPTIMIZATION_SPARSE_IPM_QP_SOLVER_HPP_

#include "Optimization/Matrix/SparseMatrixEngine.hpp"
#include <cmath>
#include <algorithm>

namespace Optimization {

template <size_t N_vars, size_t N_ineq, size_t MaxNNZ_P, size_t MaxNNZ_A>
class SparseIPMQPSolver {
public:
    StaticSparseMatrix<double, N_vars, N_vars, MaxNNZ_P> P;
    StaticVector<double, N_vars> q;
    StaticSparseMatrix<double, N_ineq, N_vars, MaxNNZ_A> A_ineq;
    StaticVector<double, N_ineq> b_ineq;

    SparseIPMQPSolver() {
        q.set_zero();
        b_ineq.set_zero();
    }

    // 1. 암시적 헤시안-벡터 곱 연산: Hp = (P + A^T * W * A) * p
    void apply_H_sys(const StaticVector<double, N_vars>& p, const StaticVector<double, N_ineq>& W, StaticVector<double, N_vars>& Hp) const {
        StaticVector<double, N_vars> Pp;
        P.multiply(p, Pp);

        StaticVector<double, N_ineq> Ap;
        A_ineq.multiply(p, Ap);

        StaticVector<double, N_ineq> WAp;
        for (size_t i = 0; i < N_ineq; ++i) {
            WAp(i) = W(i) * Ap(i);
        }

        StaticVector<double, N_vars> AtWAp;
        A_ineq.multiply_transpose(WAp, AtWAp);

        for (size_t i = 0; i < N_vars; ++i) {
            Hp(i) = Pp(i) + AtWAp(i);
        }
    }

    // 2. Matrix-Free Conjugate Gradient (켤레 기울기법) 솔버
    bool solve_implicit_cg(const StaticVector<double, N_vars>& rhs, const StaticVector<double, N_ineq>& W, StaticVector<double, N_vars>& dx) const {
        dx.set_zero();
        StaticVector<double, N_vars> r = rhs;
        StaticVector<double, N_vars> p_cg = r;
        double rsold = 0.0;
        
        for (size_t i = 0; i < N_vars; ++i) {
            rsold += r(i) * r(i);
        }

        if (std::sqrt(rsold) < 1e-9) return true;

        for (int iter = 0; iter < static_cast<int>(N_vars) * 2; ++iter) {
            StaticVector<double, N_vars> Ap_cg;
            apply_H_sys(p_cg, W, Ap_cg);

            double pAp = 0.0;
            for (size_t i = 0; i < N_vars; ++i) {
                pAp += p_cg(i) * Ap_cg(i);
            }

            if (std::abs(pAp) < 1e-12) break;

            double alpha = rsold / pAp;
            double rsnew = 0.0;

            for (size_t i = 0; i < N_vars; ++i) {
                dx(i) += alpha * p_cg(i);
                r(i) -= alpha * Ap_cg(i);
                rsnew += r(i) * r(i);
            }

            if (std::sqrt(rsnew) < 1e-6) break;

            double beta = rsnew / rsold;
            for (size_t i = 0; i < N_vars; ++i) {
                p_cg(i) = r(i) + beta * p_cg(i);
            }
            rsold = rsnew;
        }
        return true;
    }

    // 3. 메인 내점법 (Primal-Dual Interior Point Method) 솔버
    bool solve(StaticVector<double, N_vars>& u_opt, int max_iter = 10, double tol = 1e-3) {
        StaticVector<double, N_vars> x = u_opt;
        StaticVector<double, N_ineq> s;
        StaticVector<double, N_ineq> z;
        
        for (size_t i = 0; i < N_ineq; ++i) {
            s(i) = 1.0;
            z(i) = 1.0;
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            StaticVector<double, N_vars> Px;
            P.multiply(x, Px);
            
            StaticVector<double, N_vars> At_z;
            A_ineq.multiply_transpose(z, At_z);
            
            StaticVector<double, N_vars> r_d;
            for (size_t i = 0; i < N_vars; ++i) {
                r_d(i) = Px(i) + q(i) + At_z(i);
            }

            StaticVector<double, N_ineq> Ax;
            A_ineq.multiply(x, Ax);
            
            StaticVector<double, N_ineq> r_p;
            for (size_t i = 0; i < N_ineq; ++i) {
                r_p(i) = Ax(i) - b_ineq(i) + s(i);
            }

            StaticVector<double, N_ineq> r_c;
            double mu = 0.0;
            for (size_t i = 0; i < N_ineq; ++i) {
                r_c(i) = s(i) * z(i);
                mu += r_c(i);
            }
            mu /= N_ineq;

            double r_norm = 0.0;
            for(size_t i = 0; i < N_vars; ++i) r_norm += r_d(i) * r_d(i);
            for(size_t i = 0; i < N_ineq; ++i) r_norm += r_p(i) * r_p(i);
            r_norm = std::sqrt(r_norm);

            if (r_norm < tol && mu < tol) {
                u_opt = x;
                return true;
            }

            double sigma = 0.1;
            for (size_t i = 0; i < N_ineq; ++i) {
                r_c(i) = r_c(i) - sigma * mu;
            }

            StaticVector<double, N_ineq> W;
            StaticVector<double, N_ineq> inv_S;
            for (size_t i = 0; i < N_ineq; ++i) {
                inv_S(i) = 1.0 / s(i);
                W(i) = z(i) * inv_S(i);
            }

            StaticVector<double, N_vars> rhs;
            StaticVector<double, N_ineq> temp_vec;
            for (size_t i = 0; i < N_ineq; ++i) {
                temp_vec(i) = inv_S(i) * (z(i) * r_p(i) - r_c(i));
            }

            StaticVector<double, N_vars> At_temp;
            A_ineq.multiply_transpose(temp_vec, At_temp);

            for (size_t i = 0; i < N_vars; ++i) {
                rhs(i) = -r_d(i) - At_temp(i);
            }

            StaticVector<double, N_vars> dx;
            solve_implicit_cg(rhs, W, dx);

            StaticVector<double, N_ineq> Adx;
            A_ineq.multiply(dx, Adx);

            StaticVector<double, N_ineq> ds;
            StaticVector<double, N_ineq> dz;
            for (size_t i = 0; i < N_ineq; ++i) {
                ds(i) = -r_p(i) - Adx(i);
                dz(i) = inv_S(i) * (-r_c(i) - z(i) * ds(i));
            }

            double alpha_p = 1.0;
            double alpha_d = 1.0;
            for (size_t i = 0; i < N_ineq; ++i) {
                if (ds(i) < 0.0) {
                    alpha_p = std::min(alpha_p, -0.99 * s(i) / ds(i));
                }
                if (dz(i) < 0.0) {
                    alpha_d = std::min(alpha_d, -0.99 * z(i) / dz(i));
                }
            }

            for (size_t i = 0; i < N_vars; ++i) {
                x(i) += alpha_p * dx(i);
            }
            for (size_t i = 0; i < N_ineq; ++i) {
                s(i) += alpha_p * ds(i);
                z(i) += alpha_d * dz(i);
            }
        }
        
        u_opt = x;
        return false;
    }
};

} // namespace Optimization

#endif // OPTIMIZATION_SPARSE_IPM_QP_SOLVER_HPP_