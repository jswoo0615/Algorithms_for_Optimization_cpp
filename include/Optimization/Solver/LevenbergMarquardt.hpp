#ifndef OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_
#define OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
namespace solver {
/**
 * @brief Levenberg-Marquardt 비선형 최소자승 솔버 (Nonlinear Least Squares)
 * @details min 0.5 * ||F(x)||^2 를 수행합니다
 * @tparam N 최적화할 변수 갯수 (State/Control dimension)
 * @tparam M 잔차 (Residual) 방정식의 갯수 (M >= N 이어야 함)
 */
template <size_t N, size_t M, typename Functor>
SolverStatus solve_LM(StaticVector<double, N>& x_opt, const Functor& calc_residuals,
                      int max_iter = 50, double tol = 1e-6, double initial_lambda = 1e-3) {
    using ADVar = DualVec<double, N>;
    double lambda = initial_lambda;

    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. DualVec 변수 초기화 (시드 주입)
        StaticVector<ADVar, N> x_dual;
        for (size_t i = 0; i < N; ++i) {
            x_dual(static_cast<int>(i)) = ADVar::make_variable(x_opt(static_cast<int>(i)), i);
        }

        // 2. 잔차 벡터 F(x) 및 자코비안 J 동시 평가
        StaticVector<ADVar, M> residuals_dual = calc_residuals(x_dual);

        // 3. H = J^T * J와 g = J^T * F 조립 (명시적 루프로 캐시 최적화)
        StaticMatrix<double, N, N> H;
        StaticVector<double, N> g;
        double current_cost = 0.0;

        // H, g 0으로 초기화
        for (size_t i = 0; i < N; ++i) {
            g(static_cast<int>(i)) = 0.0;
            for (size_t j = 0; j < N; ++j) {
                H(static_cast<int>(i), static_cast<int>(j)) = 0.0;
            }
        }

        for (size_t k = 0; k < M; ++k) {
            double r_val = residuals_dual(static_cast<int>(k)).v;
            current_cost += 0.5 * r_val * r_val;

            for (size_t i = 0; i < N; ++i) {
                double J_ki = residuals_dual(static_cast<int>(k)).g[i];
                g(static_cast<int>(i)) += J_ki * r_val;  // g = J^T * F

                for (size_t j = 0; j < N; ++j) {
                    double J_kj = residuals_dual(static_cast<int>(k)).g[j];
                    H(static_cast<int>(i), static_cast<int>(j)) += J_ki * J_kj;  // H = J^T * J
                }
            }
        }

        // 4. 수렴 판정 (Gradient의 최대 크기가 허용 오차 이내인가?)
        double max_grad = 0.0;
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(g(static_cast<int>(i))) > max_grad) {
                max_grad = std::abs(g(static_cast<int>(i)));
            }
        }
        if (max_grad < tol) {
            return SolverStatus::SUCCESS;
        }

        // 5. LM Damping 추가 : H_lm = H + lambda * I (또는 대각성분 비례)
        StaticMatrix<double, N, N> H_lm = H;
        for (size_t i = 0; i < N; ++i) {
            // H_lm(i, i) += lambda;        // 기본적인 Identity Damping
            H_lm(static_cast<int>(i), static_cast<int>(i)) +=
                lambda * (H(static_cast<int>(i), static_cast<int>(i)) + 1e-6);
        }

        // 6. 선형 시스템 해 찾기 : H_lm * dx = -g
        // H_lm은 대칭 양의 정부호 (SPD)이므로 LDLT 솔버를 사용
        for (size_t i = 0; i < N; ++i) {
            g(static_cast<int>(i)) = -g(static_cast<int>(i));  // -g
        }
        MathStatus m_status = linalg::LDLT_decompose(H_lm);
        if (m_status != MathStatus::SUCCESS) {
            // LDLT가 실패하면 Damping을 늘려 행렬의 조건수 (Condition number)를 개선
            lambda *= 10.0;
            continue;
        }

        StaticVector<double, N> dx = linalg::LDLT_solve(H_lm, g);

        // 7. 업데이트 평가 (스텝을 밟았을 때 Cost가 줄어드는가?)
        StaticVector<double, N> x_new;
        for (size_t i = 0; i < N; ++i) {
            x_new(static_cast<int>(i)) = x_opt(static_cast<int>(i)) + dx(static_cast<int>(i));
        }

        // 새 위치에서의 Cost 계산 (Value만 필요하므로 double 오버로딩 또는 Dual 그대로 사용)
        StaticVector<ADVar, N> x_new_dual;
        for (size_t i = 0; i < N; ++i) {
            x_new_dual(static_cast<int>(i)) = ADVar(x_new(static_cast<int>(i)));
        }

        StaticVector<ADVar, M> new_residuals = calc_residuals(x_new_dual);
        double new_cost = 0.0;
        for (size_t k = 0; k < M; ++k) {
            new_cost +=
                0.5 * new_residuals(static_cast<int>(k)).v * new_residuals(static_cast<int>(k)).v;
        }

        if (new_cost < current_cost) {
            // 성공적인 스텝 : x 업데이트 및 Damping 감소 (가우스-뉴턴에 가깝게)
            for (size_t i = 0; i < N; ++i) {
                x_opt(static_cast<int>(i)) = x_new(static_cast<int>(i));
            }
            lambda = std::max(1e-7, lambda / 10.0);
        } else {
            // 실패한 스텝 : x 유지, Damping 증가 (경사하강법에 가깝게)
            lambda *= 10.0;
        }
    }
    return SolverStatus::MAX_ITERATION_REACHED;
}
}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_