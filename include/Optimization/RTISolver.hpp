#ifndef OPTIMIZATION_RTI_SOLVER_HPP_
#define OPTIMIZATION_RTI_SOLVER_HPP_

#include <chrono>
#include <cmath>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/IPMQPSolver.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

// Sparse 관련 MaxNNZ 템플릿 파라미터 소거됨
template <size_t N_vars, size_t N_eq, size_t N_ineq, size_t N_res>
class RTISolver {
   public:
    // 오직 Dense IPM 인스턴스만 유지
    IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver_dense;

    struct ProfileData {
        double last_exec_time_ms;
        double wcet_ms;
        const char* current_engine;
    } profile;

    RTISolver() {
        profile.last_exec_time_ms = 0.0;
        profile.wcet_ms = 0.0;
        profile.current_engine = "None";
    }

    // ====================================================================
    // Dense Solver (가우스-뉴턴 기반 조밀 행렬 해법 및 KKT 모니터링 연동)
    // ====================================================================
    template <typename ResidualFunc, typename EqFunc, typename IneqFunc>
    bool solve_dense(StaticVector<double, N_vars>& u, ResidualFunc res_f, EqFunc eq_f,
                     IneqFunc ineq_f) {
        auto start = std::chrono::high_resolution_clock::now();
        profile.current_engine = "Dense (Gaussian IPM)";

        // 1. Gauss-Newton Hessian Approximation (P) & Gradient (q)
        StaticVector<double, N_res> r_val = res_f(u);
        StaticMatrix<double, N_res, N_vars> J_res = AutoDiff::jacobian<N_res, N_vars>(res_f, u);
        StaticMatrix<double, N_vars, N_vars> H_GN;
        StaticVector<double, N_vars> grad_f;

        H_GN.set_zero();
        grad_f.set_zero();

        for (size_t i = 0; i < N_vars; ++i) {
            for (size_t j = 0; j < N_vars; ++j) {
                double sum_h = 0.0;
                for (size_t k = 0; k < N_res; ++k) {
                    sum_h += J_res(k, i) * J_res(k, j);
                }
                H_GN(i, j) = sum_h;
            }
            double sum_g = 0.0;
            for (size_t k = 0; k < N_res; ++k) {
                sum_g += J_res(k, i) * r_val(k);
            }
            grad_f(i) = sum_g;
            H_GN(i, i) += 5.0;  // Levenberg-Marquardt damping
        }

        // 2. Constraints Setup
        StaticVector<double, N_ineq> ineq_val = ineq_f(u);
        StaticMatrix<double, N_ineq, N_vars> J_ineq = AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);

        for (size_t i = 0; i < N_ineq; ++i) {
            qp_solver_dense.b_ineq(i) = -ineq_val(i);
            for (size_t j = 0; j < N_vars; ++j) {
                qp_solver_dense.A_ineq(i, j) = J_ineq(i, j);
            }
        }

        qp_solver_dense.P = H_GN;
        qp_solver_dense.q = grad_f;

        // 3. Solve QP using Dense IPM
        StaticVector<double, N_vars> p;
        p.set_zero();
        // 직전에 주입한 IPMQPSolver의 KKT Monitor가 여기서 작동합니다
        bool success = qp_solver_dense.solve(p, 15, 1e-3);

        if (success) {
            for (size_t i = 0; i < N_vars; ++i) {
                u(i) += p(i);
            }
        }

        // 4. Profiling
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        profile.last_exec_time_ms = elapsed.count();
        if (profile.last_exec_time_ms > profile.wcet_ms) {
            profile.wcet_ms = profile.last_exec_time_ms;
        }

        return success;
    }
};

}  // namespace Optimization
#endif  // OPTIMIZATION_RTI_SOLVER_HPP_