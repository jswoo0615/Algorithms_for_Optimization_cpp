#ifndef OPTIMIZATION_GRADIENT_DESCENT_HPP_
#define OPTIMIZATION_GRADIENT_DESCENT_HPP_

/**
 * @file GradientDescent.hpp
 * @brief Gradient Descent (경사하강법) 최적화 알고리즘 구현
 * 
 * Gradient Descent 알고리즘은 1차 미분(Gradient) 정보를 활용하여
 * 주어진 목적 함수(Objective Function)의 최솟값을 찾는 가장 기본적인 최적화 기법입니다.
 * 이 구현은 방향(Direction)을 Gradient의 반대 방향으로 설정하고,
 * 이동 거리(Step Size, alpha)는 Strong Wolfe 조건을 만족하는 Backtracking Line Search를 통해 
 * 적응적으로 결정하여 최적화 과정의 안정성과 수렴성을 보장합니다.
 */

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/StrongBacktrackingLineSearch.hpp"

namespace Optimization {

/**
 * @class GradientDescent
 * @brief Gradient Descent 알고리즘을 수행하는 정적 클래스
 */
class GradientDescent {
   public:
    /**
     * @brief Gradient Descent 알고리즘을 사용하여 주어진 함수의 최솟값을 탐색합니다.
     * 
     * @tparam N 최적화할 변수의 차원 (크기)
     * @tparam Func 목적 함수(Objective Function)의 타입. (AutoDiff 적용을 위해 템플릿 처리)
     * 
     * @param f 최적화하려는 목적 함수 (AutoDiff 호환 함수)
     * @param x 초기 시작 위치 (Initial guess)
     * @param tol 수렴 판정을 위한 허용 오차 (Gradient의 L2 Norm이 이 값보다 작아지면 종료). 기본값: 1e-6
     * @param max_iter 최대 반복 횟수. 이 횟수에 도달하면 알고리즘이 강제 종료됩니다. 기본값: 1000
     * @param verbose 진행 과정 출력 여부. true일 경우 매 반복마다 상태를 콘솔에 출력합니다. 기본값: false
     * 
     * @return std::array<double, N> 최적화가 완료된 후의 변수 값 (최적해)
     */
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, double tol = 1e-6,
                                          int max_iter = 1000, bool verbose = false) {
        // verbose 플래그가 참일 경우 최적화 시작을 알림
        if (verbose) std::cout << "🚀 Gradient Descent Started...\n";

        for (int i = 0; i < max_iter; ++i) {
            // 1. 현재 위치의 함수값(f_val)과 기울기(Gradient, grad) 계산
            // AutoDiff::value_and_gradient를 사용하여 자동 미분 기반으로 정확한 함수값과 기울기를 얻습니다.
            double f_val;
            std::array<double, N> grad;
            AutoDiff::value_and_gradient<N>(f, x, f_val, grad);

            // 2. 기울기의 L2 Norm (유클리디안 노름) 계산
            // 기울기의 벡터 크기(|grad|)를 통해 현재 위치가 충분히 평탄한지(극솟값에 도달했는지) 확인합니다.
            double g_norm = 0.0;
            for (double g : grad) {
                g_norm += g * g;
            }
            g_norm = std::sqrt(g_norm);

            // 현재 상태 출력 (반복 횟수, 함수값, 기울기의 크기)
            if (verbose) {
                std::cout << "[Iter " << i << "] f(x) = " << f_val << " | |grad| = " << g_norm
                          << "\n";
            }

            // 수렴 확인: 기울기의 크기가 지정된 허용 오차(tol)보다 작으면 
            // 최적점(극솟값)에 충분히 도달했다고 판단하여 탐색을 종료합니다.
            if (g_norm < tol) break;

            // 3. 하강 방향 설정 (기울기의 반대 방향)
            // Gradient는 함수값이 가장 가파르게 '증가'하는 방향이므로,
            // 최솟값을 찾기 위해 그 반대 방향(-grad)으로 탐색 방향(direction)을 설정합니다.
            std::array<double, N> direction;
            for (size_t j = 0; j < N; ++j) direction[j] = -grad[j];

            // 4. 보폭 (Step Size, alpha) 결정
            // 고정된 보폭(Learning Rate) 대신 Strong Wolfe 조건을 만족하는 Backtracking Line Search를 사용하여
            // 충분한 함수값 감소를 보장하는 최적의 이동 거리(alpha)를 탐색합니다.
            double alpha = StrongBacktrackingLineSearch::search<N>(f, x, direction);

            // 5. 위치 업데이트 : x_{k+1} = x_k + alpha * d_k
            // 계산된 보폭과 방향을 곱하여 현재 위치를 갱신합니다.
            for (size_t j = 0; j < N; ++j) {
                x[j] += alpha * direction[j];
            }
        }
        
        // 탐색이 끝난 후, 찾아낸 최적의 파라미터(최적해) 반환
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_GRADIENT_DESCENT_HPP_