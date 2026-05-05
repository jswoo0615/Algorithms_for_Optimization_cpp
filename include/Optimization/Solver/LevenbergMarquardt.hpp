#ifndef OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_
#define OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_

#include "Optimization/Dual.hpp"
#include "Optimization/Matrix/MathTraits.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"
#include "Optimization/Solver/SolverStatus.hpp"

namespace Optimization {
namespace solver {

/**
 * @brief 고속 Levenberg-Marquardt 비선형 최소자승 솔버 (Zero-Allocation & SIMD)
 * @details 
 * min 0.5 * ||F(x)||^2 를 수행합니다.
 * 내부 루프(Inner Retry)를 통해 실패한 스텝에서 자코비안(AD) 연산을 생략하고,
 * SIMD 행렬 엔진을 이용하여 H = J^T J 연산을 극대화했습니다.
 * 
 * @tparam N 최적화할 변수 갯수 (State/Control dimension)
 * @tparam M 잔차 (Residual) 방정식의 갯수 (M >= N)
 */
template <size_t N, size_t M, typename Functor>
inline SolverStatus solve_LM(StaticVector<double, N>& x_opt, const Functor& calc_residuals,
                             int max_iter = 50, double tol = 1e-6, double initial_lambda = 1e-3) {
    static_assert(M >= N, "Residuals M must be greater than or equal to variables N");
    
    using ADVar = DualVec<double, N>;
    double lambda = initial_lambda;

    // [Architect's Update] 메모리 호이스팅 (루프 외부로 메모리 블록을 통째로 이전)
    StaticVector<ADVar, N> x_dual;
    StaticMatrix<double, M, N> J_mat;
    StaticVector<double, M> F_vec;
    StaticMatrix<double, N, N> H;
    StaticVector<double, N> g;
    StaticMatrix<double, N, N> H_lm;
    StaticVector<double, N> neg_g;
    StaticVector<double, N> dx;
    StaticVector<double, N> x_new;
    StaticVector<ADVar, N> x_new_dual;

    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. DualVec 변수 초기화 (시드 주입)
        for (size_t i = 0; i < N; ++i) {
            x_dual(i) = ADVar::make_variable(x_opt(i), i);
        }

        // 2. 잔차 벡터 F(x) 및 자코비안 J 동시 평가 (가장 무거운 연산)
        StaticVector<ADVar, M> residuals_dual = calc_residuals(x_dual);

        // 3. F 벡터 및 J 행렬 추출, 현재 Cost 계산
        double current_cost = 0.0;
        for (size_t k = 0; k < M; ++k) {
            double r_val = Optimization::get_value(residuals_dual(k));
            F_vec(k) = r_val;
            current_cost += 0.5 * r_val * r_val;

            for (size_t i = 0; i < N; ++i) {
                J_mat(k, i) = residuals_dual(k).g[i];
            }
        }

        // 4. H = J^T * J 와 g = J^T * F 조립 (SIMD 가속 엔진 타격)
        linalg::multiply_AT_B(J_mat, J_mat, H);
        linalg::multiply_AT_B(J_mat, F_vec, g);

        // 5. 수렴 판정 (Gradient의 최대 크기가 허용 오차 이내인가?)
        double max_grad = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double abs_g = MathTraits<double>::abs(g(i));
            if (abs_g > max_grad) max_grad = abs_g;
        }
        if (max_grad < tol) {
            return SolverStatus::SUCCESS; // 성공적 수렴
        }

        // 6. [Architect's Update] Inner Retry Loop (자코비안 재계산 방어)
        bool step_accepted = false;
        int inner_retry = 0;
        
        while (!step_accepted && inner_retry < 10) {
            // LM Damping 추가 : H_lm = H + lambda * diag(H)
            H_lm = H; // 복사
            for (size_t i = 0; i < N; ++i) {
                H_lm(i, i) += lambda * (H(i, i) + 1e-6);
                neg_g(i) = -g(i); // 우변 벡터 부호 반전
            }

            // H_lm은 대칭 양의 정부호(SPD) 보장 -> LDLT 분해
            MathStatus m_status = linalg::LDLT_decompose(H_lm);
            if (m_status != MathStatus::SUCCESS) {
                // 수치적 불안정: Damping을 늘려 조건수(Condition Number) 개선 후 재시도
                lambda *= 10.0;
                inner_retry++;
                continue;
            }

            // Zero-Allocation LDLT Solve
            linalg::LDLT_solve(H_lm, neg_g, dx);

            // Step Size(Stagnation) 검증
            double max_dx = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double abs_dx = MathTraits<double>::abs(dx(i));
                if (abs_dx > max_dx) max_dx = abs_dx;
            }
            if (max_dx < std::numeric_limits<double>::epsilon() * 10.0) {
                return SolverStatus::STEP_SIZE_TOO_SMALL; // 탐색 불가
            }

            // x_new = x_opt + dx (SIMD 가속)
            x_new = x_opt;
            x_new += dx;

            // 7. 새로운 위치의 Cost 평가 (Gradient는 필요 없으므로 상수 듀얼 값 주입)
            for (size_t i = 0; i < N; ++i) {
                x_new_dual(i) = ADVar(x_new(i)); 
            }
            
            StaticVector<ADVar, M> new_residuals = calc_residuals(x_new_dual);
            double new_cost = 0.0;
            for (size_t k = 0; k < M; ++k) {
                double r_val = Optimization::get_value(new_residuals(k));
                new_cost += 0.5 * r_val * r_val;
            }

            // 8. 스텝 승인(Accept) 및 거절(Reject) 판단
            if (new_cost < current_cost) {
                // 성공적인 스텝 : x 업데이트 및 Damping 감소 (가우스-뉴턴에 근접)
                x_opt = x_new;
                lambda = MathTraits<double>::max(1e-7, lambda / 10.0);
                step_accepted = true; // 내부 루프 탈출
            } else {
                // 실패한 스텝 : 위치는 유지하되 Damping 증가 (경사하강법에 근접)
                lambda *= 10.0;
                inner_retry++; // J를 다시 구하지 않고 while문 재진입
            }
        }

        if (!step_accepted) {
            // 댐핑을 10번이나 올렸는데도 Cost를 줄이는 스텝을 찾지 못함 -> 평탄 지형(Stagnation)
            return SolverStatus::STEP_SIZE_TOO_SMALL;
        }
    }
    
    return SolverStatus::MAX_ITERATION_REACHED;
}

}  // namespace solver
}  // namespace Optimization

#endif  // OPTIMIZATION_LEVENBERG_MARQUARDT_HPP_