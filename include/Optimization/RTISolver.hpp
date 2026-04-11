#ifndef OPTIMIZATION_RTI_SOLVER_HPP_
#define OPTIMIZATION_RTI_SOLVER_HPP_

#include "Optimization/AutoDiff.hpp"
#include "Optimization/IPMQPSolver.hpp" 
#include <cmath>
#include <algorithm>

namespace Optimization {

// [Architect's True Gauss-Newton] N_res(잔차 벡터 크기) 템플릿 추가
template <size_t N_vars, size_t N_eq, size_t N_ineq, size_t N_res>
class RTISolver {
public:
    IPMQPSolver<N_vars, N_eq, N_ineq> qp_solver;

    RTISolver() {}

    template <typename ResidualFunc, typename EqFunc, typename IneqFunc>
    bool solve(StaticVector<double, N_vars>& u, ResidualFunc res_f, EqFunc eq_f, IneqFunc ineq_f) {
        
        // ---------------------------------------------------------------------
        // 1. Preparation Phase: True Gauss-Newton Hessian (J^T * J)
        // ---------------------------------------------------------------------
        
        // 잔차 벡터(r)와 야코비안(J_res) 추출
        StaticVector<double, N_res> r_val = res_f(u);
        StaticMatrix<double, N_res, N_vars> J_res = AutoDiff::jacobian<N_res, N_vars>(res_f, u);

        StaticMatrix<double, N_vars, N_vars> H_GN;
        StaticVector<double, N_vars> grad_f;
        H_GN.set_zero();
        grad_f.set_zero();

        // H = J^T * J  /  g = J^T * r 계산
        for (size_t i = 0; i < N_vars; ++i) {
            for (size_t j = 0; j < N_vars; ++j) {
                double sum_h = 0.0;
                for (size_t k = 0; k < N_res; ++k) {
                    sum_h += J_res(static_cast<int>(k), static_cast<int>(i)) * J_res(static_cast<int>(k), static_cast<int>(j));
                }
                H_GN(static_cast<int>(i), static_cast<int>(j)) = sum_h;
            }
            
            double sum_g = 0.0;
            for (size_t k = 0; k < N_res; ++k) {
                sum_g += J_res(static_cast<int>(k), static_cast<int>(i)) * r_val(static_cast<int>(k));
            }
            grad_f(static_cast<int>(i)) = sum_g;

            // [Architect's Armor] Levenberg-Marquardt Damping
            // Line Search가 없는 RTI 환경에서, 장애물 회피 시 발생하는 폭발적인 Gradient를
            // 부드럽게 억제하고 스텝 보폭을 안정화하는 'Trust-Region' 가상 브레이크 역할
            H_GN(static_cast<int>(i), static_cast<int>(i)) += 10.0; 
        }

        // 제약 조건 야코비안 선형화
        if constexpr (N_eq > 0) {
            StaticVector<double, N_eq> eq_val = eq_f(u);
            StaticMatrix<double, N_eq, N_vars> J_eq = AutoDiff::jacobian<N_eq, N_vars>(eq_f, u);
            for(size_t i=0; i<N_eq; ++i) {
                qp_solver.b_eq(static_cast<int>(i)) = -eq_val(static_cast<int>(i));
                for(size_t j=0; j<N_vars; ++j) {
                    qp_solver.A_eq(static_cast<int>(i), static_cast<int>(j)) = J_eq(static_cast<int>(i), static_cast<int>(j));
                }
            }
        }

        if constexpr (N_ineq > 0) {
            StaticVector<double, N_ineq> ineq_val = ineq_f(u);
            StaticMatrix<double, N_ineq, N_vars> J_ineq = AutoDiff::jacobian<N_ineq, N_vars>(ineq_f, u);
            for(size_t i=0; i<N_ineq; ++i) {
                qp_solver.b_ineq(static_cast<int>(i)) = -ineq_val(static_cast<int>(i));
                for(size_t j=0; j<N_vars; ++j) {
                    qp_solver.A_ineq(static_cast<int>(i), static_cast<int>(j)) = J_ineq(static_cast<int>(i), static_cast<int>(j));
                }
            }
        }

        qp_solver.P = H_GN;
        qp_solver.q = grad_f;
        
        StaticVector<double, N_vars> p;
        p.set_zero();

        // ---------------------------------------------------------------------
        // 2. Feedback Phase: Fixed Iteration IPM (WCET 보장)
        // ---------------------------------------------------------------------
        bool qp_success = qp_solver.solve(p, 15, 1e-3); 

        // ---------------------------------------------------------------------
        // 3. Update Phase: Full Step & Hard Clamping
        // ---------------------------------------------------------------------
        if (qp_success) {
            for(size_t i=0; i<N_vars; ++i) {
                // NaN/Inf 방탄조끼
                if (std::isnan(p(static_cast<int>(i))) || std::isinf(p(static_cast<int>(i)))) {
                    p(static_cast<int>(i)) = 0.0;
                }
                
                // 제어 입력 업데이트 (Line Search 없음)
                u(static_cast<int>(i)) += p(static_cast<int>(i));
                
                // [Architect's Safe Bound] 제어 입력 자체를 물리적 한계로 최종 클램핑
                if (i % 2 == 0) { // 가속도 [-3.0, 3.0]
                    if (u(static_cast<int>(i)) > 3.0) u(static_cast<int>(i)) = 3.0;
                    if (u(static_cast<int>(i)) < -3.0) u(static_cast<int>(i)) = -3.0;
                } else { // 조향각 [-0.5, 0.5] rad
                    if (u(static_cast<int>(i)) > 0.5) u(static_cast<int>(i)) = 0.5;
                    if (u(static_cast<int>(i)) < -0.5) u(static_cast<int>(i)) = -0.5;
                }
            }
            return true;
        } 
        
        // [Fail-safe] IPM 실패 시 쓰레기 값(p)을 더하지 않고, 이전 안전 상태(u) 유지
        return false; 
    }
};

} // namespace Optimization

#endif  // OPTIMIZATION_RTI_SOLVER_HPP_