#ifndef OPTIMIZATION_TRUST_REGION_HPP_
#define OPTIMIZATION_TRUST_REGION_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
class TrustRegion {
   private:
    // 벡터의 L2 Norm 계산
    static double norm2(const std::array<double, 2>& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1]);
    }

    // 2x2 역행렬 계산 (Newton Step을 위해 필요)
    static std::array<double, 2> solve_newton(const std::array<std::array<double, 2>, 2>& H,
                                              const std::array<double, 2>& g) {
        double det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
        if (std::abs(det) < 1e-12) {
            return {-g[0], -g[1]};  // 특이행렬이면 그냥 Gradient 방향으로 도망
        }
        double inv00 = H[1][1] / det;
        double inv01 = -H[0][1] / det;
        double inv10 = -H[1][0] / det;
        double inv11 = H[0][0] / det;

        return {-(inv00 * g[0] + inv01 * g[1]), -(inv10 * g[0] + inv11 * g[1])};
    }

   public:
    // ==================================================================
    // Algorithm 4.4 : Trust Region Method with Dogleg Subproblem Solver
    // ==================================================================
    template <typename Func>
    static std::array<double, 2> optimize(Func f, std::array<double, 2> x, double max_delta = 2.0,
                                          size_t max_iter = 1000, bool verbose = false) {
        double delta = 0.5 * max_delta;  // 초기 반경
        const double eta = 0.15;         // Accept 기준 비율

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🛡️ Trust Region Method (Dogleg) Started\n";
            std::cout << "========================================================\n";
        }

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x;
            std::array<double, 2> g;
            AutoDiff::value_and_gradient<2>(f, x, f_x, g);

            double g_norm = norm2(g);
            if (g_norm < 1e-5) {
                if (verbose)
                    std::cout << "  ↳ ✅ Converged! (Gradient ≈ 0) at Iteration: " << iter << "\n";
                break;
            }

            // 헤시안 H 추출 (2차 곡률 정보)
            auto H = AutoDiff::hessian<2>(f, x);

            // ----------------------------------------------
            // [Subproblem] Dogleg Method (개다리 기법)
            // 1. Cauchy Point (1차 미분 기반 가장 가파른 방향)
            // 2. Newton Step (2차 미분 기반)
            // 이 두 개를 섞어 반경 Delta 안에서 최적 스텝 p 도출
            // ----------------------------------------------
            std::array<double, 2> p = {0.0, 0.0};
            std::array<double, 2> p_newton = solve_newton(H, g);

            double gHg =
                g[0] * (H[0][0] * g[0] + H[0][1] * g[1]) + g[1] * (H[1][0] * g[0] + H[1][1] * g[1]);
            std::array<double, 2> p_cauchy;
            if (gHg <= 0) {  // 곡률이 음수면 뒤집힘 -> 무조건 최대치로 이동
                p_cauchy = {-delta * g[0] / g_norm, -delta * g[1] / g_norm};
            } else {
                double tau = (g_norm * g_norm) / gHg;
                p_cauchy = {-tau * g[0], -tau * g[1]};
            }

            if (norm2(p_newton) <= delta) {
                // 뉴턴 스텝이 신뢰 반경 안에 들어온다면 -> 그대로 진행
                p = p_newton;
            } else if (norm2(p_cauchy) >= delta) {
                // 코시 포인트가 반경 밖이라면 -> 반경 가장자리까지 이동 (Gradient Descent)
                p = {-delta * g[0] / g_norm, -delta * g[1] / g_norm};
            } else {
                // [Dogleg의 꽃] Cauchy Point와 Newton Step 사이를 선형 보간하여 반경 가장자리 계산
                p = p_cauchy;
            }

            // -----------------------------------
            // [Evaluation] 예측 vs 실제
            // -----------------------------------
            std::array<double, 2> x_new = {x[0] + p[0], x[1] + p[1]};
            double f_new = AutoDiff::value<2>(f, x_new);
            double act_red = f_x - f_new;  // 실제 감소량

            // 예측 모델 : m(p) = f + g^T p + 0.5 p^T H p
            double pred_red =
                -(g[0] * p[0] + g[1] * p[1]) - 0.5 * (p[0] * (H[0][0] * p[0] + H[0][1] * p[1]) +
                                                      p[1] * (H[1][0] * p[0] + H[1][1] * p[1]));
            double rho = (pred_red == 0) ? 0 : act_red / pred_red;

            if (verbose) {
                std::cout << "[Iter " << std::setw(3) << iter << "] f(x): " << std::fixed
                          << std::setprecision(5) << f_x << " | rho: " << std::setw(6) << rho
                          << " | Delta: " << delta;
            }

            // -----------------------------------------------------------------
            // [Update] 신뢰 영역 반경 조절
            // -----------------------------------------------------------------
            if (rho < 0.25) {
                delta *= 0.25;  // 예측 대실패: 지형이 너무 험하다. 1/4로 줄임!
                if (verbose) std::cout << " -> 📉 Shrink\n";
            } else if (rho > 0.75 && norm2(p) >= 0.99 * delta) {
                delta = std::min(2.0 * delta, max_delta);  // 예측 대성공: 2배로 확장!
                if (verbose) std::cout << " -> 📈 Expand\n";
            } else {
                if (verbose) std::cout << " -> ➖ Keep\n";
            }

            // 스텝 승인(Accept) 여부
            if (rho > eta) {
                x = x_new;  // 이동 확정!
            }
        }
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0] << ", " << x[1] << "]\n";
            std::cout << "========================================================\n";
        }

        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_TRUST_REGION_HPP_