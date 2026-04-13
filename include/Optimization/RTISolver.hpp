#ifndef OPTIMIZATION_RTI_SOLVER_HPP_
#define OPTIMIZATION_RTI_SOLVER_HPP_

#include "Optimization/Matrix/SparseMatrixEngine.hpp"
#include "Optimization/SparseIPMQPSolver.hpp"
#include "Optimization/AutoDiff.hpp"
#include "Optimization/IPMQPSolver.hpp"
#include <cmath>
#include <chrono>

namespace Optimization {
    template <size_t N_vars, size_t N_eq, size_t N_ineq, size_t N_res, size_t MaxNNZ = N_vars * N_vars>

    class RTISolver {
        public:
            IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver_dense;
            // Sparse IPM 인스턴스 (제약조건 포함 KKT용)
            SparseIPMQPSolver<N_vars, N_ineq, MaxNNZ, N_vars * 8> qp_solver_sparse;

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

            template <typename ResidualFunc, typename EqFunc, typename IneqFunc>
            bool solve_dense(StaticVector<double, N_vars>& u, ResidualFunc res_f, EqFunc eq_f, IneqFunc ineq_f) {
                auto start = std::chrono::high_resolution_clock::now();
                profile.current_engine = "Dense (Gaussian IPM)";
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
                    H_GN(i, i) += 5.0; // Levenberg-Marquardt damping
                }

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

                StaticVector<double, N_vars> p;
                p.set_zero();
                bool success = qp_solver_dense.solve(p, 15, 1e-3);
                if (success) {
                    for (size_t i = 0; i < N_vars; ++i) {
                        u(i) += p(i);                    
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                profile.last_exec_time_ms = elapsed.count();
                if (profile.last_exec_time_ms > profile.wcet_ms) {
                    profile.wcet_ms = profile.last_exec_time_ms;
                }
                return success;
            }

            template <typename ResidualFunc, typename EqFunc, typename IneqFunc>
            bool solve_sparse(StaticVector<double, N_vars>& u, ResidualFunc res_f, EqFunc eq_f, IneqFunc ineq_f) {
                auto start = std::chrono::high_resolution_clock::now();
                profile.current_engine = "Sparse (Matrix-Free IPM)";

                // 1. Hessian 행렬과 Gradient 벡터 계산
                StaticVector<double, N_res> r_val = res_f(u);
                StaticMatrix<double, N_res, N_vars> J_res = AutoDiff::jacobian<N_res, N_vars>(res_f, u);
                
                qp_solver_sparse.P.nnz_count = 0;
                qp_solver_sparse.P.row_ptr.set_zero();

                for (size_t i = 0; i < N_vars; ++i) {
                    for (size_t j = 0; j < N_vars; ++j) {
                        double val = 0.0;
                        for (size_t k = 0; k < N_res; ++k) {
                            val += J_res(k, i) * J_res(k, j);
                        }
                        if (i == j) {
                            val += 5.0; // Levenberg-Marquardt damping
                        }
                        if (std::abs(val) > 1e-9) {
                            qp_solver_sparse.P.add_value(i, j, val);
                        }
                    } // j 루프 종료
                    
                    double sum_g = 0.0;
                    for (size_t k = 0; k < N_res; ++k) {
                        sum_g += J_res(k, i) * r_val(k);
                    }
                    qp_solver_sparse.q(i) = sum_g;
                }

                qp_solver_sparse.P.finalize();

                // 2. Constraint 조립
                StaticVector<double, N_ineq> ineq_val = ineq_f(u);
                StaticMatrix<double, N_ineq, N_vars> J_ineq = AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);

                qp_solver_sparse.A_ineq.nnz_count = 0;
                qp_solver_sparse.A_ineq.row_ptr.set_zero();

                for (size_t i = 0; i < N_ineq; ++i) {
                    qp_solver_sparse.b_ineq(i) = -ineq_val(i);
                    for (size_t j = 0; j < N_vars; ++j) {
                        if (std::abs(J_ineq(i, j)) > 1e-9) {
                            qp_solver_sparse.A_ineq.add_value(i, j, J_ineq(i, j));
                        }
                    }
                }
                qp_solver_sparse.A_ineq.finalize();

                // 3. Sparse IPM Solve (Warm-start u)
                StaticVector<double, N_vars> p;
                p.set_zero();
                bool success = qp_solver_sparse.solve(p, 10, 1e-3);
                
                if (success) {
                    for (size_t i = 0; i < N_vars; ++i) {
                        u(i) += p(i);
                    }
                }
                
                // 4. 타이밍 프로파일링
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                profile.last_exec_time_ms = elapsed.count();
                if (profile.last_exec_time_ms > profile.wcet_ms) {
                    profile.wcet_ms = profile.last_exec_time_ms;
                }

                return success; 
            }

    };
} // namespace Optimization
#endif // OPTIMIZATION_RTI_SOLVER_HPP_