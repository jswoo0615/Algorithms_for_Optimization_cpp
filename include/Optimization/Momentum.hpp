#ifndef OPTIMIZATION_MOMENTUM_HPP_
#define OPTIMIZATION_MOMENTUM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @class Momentum
 * @brief 모멘텀(Momentum) 최적화 알고리즘 (Heavy Ball Method)
 *
 * @details 모멘텀 방식은 경사 하강법(Gradient Descent)의 진동(Oscillation) 문제를 줄이고
 * 지역 최솟값(Local Minima)을 탈출하거나 평탄한 지역(Plateau)을 빠르게 지나가기 위해
 * 고안되었습니다. 물리적인 '관성(Momentum)'의 개념을 차용하여, 과거에 이동했던 방향(속도,
 * Velocity)을 기억해두고 현재의 기울기(Gradient) 업데이트에 일정 비율(beta)만큼 더해주는
 * 방식입니다.
 *
 * 수식:
 * v_{t+1} = beta * v_t - alpha * g_t
 * x_{t+1} = x_t + v_{t+1}
 * (여기서 v는 속도, alpha는 학습률, beta는 모멘텀 계수, g_t는 기울기)
 */
class Momentum {
   public:
    // =======================================================================
    // Algorithm 5.2 : Momentum (Heavy Ball Method)
    // =======================================================================
    /**
     * @brief 모멘텀 알고리즘을 이용해 함수의 최솟값을 찾습니다.
     *
     * @param f 최적화할 목적 함수 (AutoDiff 호환 가능해야 함)
     * @param x 탐색을 시작할 초기 위치 (N차원 벡터)
     * @param alpha 학습률 (보폭, Learning Rate). 현재 기울기를 얼마나 반영할지 결정합니다. (기본값:
     * 0.001)
     * @param beta 관성 계수 (Momentum). 과거 속도를 얼마나 유지할지 결정합니다. (일반적으로 0.9
     * 근처의 값을 사용)
     * @param max_iter 최대 반복 횟수
     * @param tol 허용 오차. 기울기의 노름(Norm)이 이 값보다 작아지면 수렴한 것으로 판단합니다.
     * @param verbose 진행 과정 콘솔 출력 여부
     * @return 최적화된 위치 벡터(x_opt) 반환
     */
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.001,
                                          double beta = 0.9, size_t max_iter = 15000,
                                          double tol = 1e-4, bool verbose = false) {
        std::array<double, N> v = {
            0.0};  // 속도(Velocity) 누적 변수. 초기 상태는 정지(0.0) 상태로 시작.

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🌪️ Momentum Optimizer Started (alpha=" << alpha << ", beta=" << beta
                      << ")\n";
            std::cout << "========================================================\n";
        }

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x;
            std::array<double, N> g;

            // 1. 현재 위치(x)에서의 목적 함수값(f_x)과 기울기 벡터(g)를 동시에 평가합니다.
            // AutoDiff(자동 미분)를 이용하므로 수치적 오차가 없고 정확한 기울기를 얻습니다.
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 2. 기울기 노름 (Gradient Norm) 계산
            // 기울기 벡터의 크기(L2-Norm)를 계산하여 현재 위치가 평탄한지(기울기가 0에 가까운지)
            // 확인합니다.
            double g_norm = 0.0;
            for (size_t i = 0; i < N; ++i) g_norm += g[i] * g[i];
            g_norm = std::sqrt(g_norm);

            // 기울기 노름이 허용 오차(tol)보다 작으면 최적점(임계점)에 도달한 것으로 판단하고 조기
            // 종료합니다.
            if (g_norm < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 3. 모멘텀 업데이트
            // 모든 차원에 대해 독립적으로 속도(Velocity)와 위치(Position)를 업데이트합니다.
            for (size_t i = 0; i < N; ++i) {
                // 이전 속도(v)에 관성 계수(beta)를 곱하여 이전 이동 방향을 유지하려는 성질을
                // 부여하고, 현재 기울기(g)에 학습률(alpha)을 곱한 값을 빼주어(경사 하강) 새로운
                // 이동 속도(방향 및 크기)를 결정합니다.
                v[i] = beta * v[i] - alpha * g[i];

                // 계산된 속도만큼 현재 위치(x)를 이동시킵니다.
                x[i] = x[i] + v[i];
            }

            // 지정된 주기(1000번)마다 현재 상태(함수값, 기울기 노름)를 출력합니다.
            if (verbose && iter % 1000 == 0) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << g_norm << "\n";
            }
        }

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            // 1차원 이상인 경우 나머지 차원의 결과값도 안전하게 이어서 출력합니다.
            for (size_t i = 1; i < N; ++i) {
                std::cout << ", " << x[i];
            }
            std::cout << "]\n========================================================\n";
        }

        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_MOMENTUM_HPP_