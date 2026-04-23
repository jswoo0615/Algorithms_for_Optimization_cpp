#ifndef OPTIMIZATION_NESTROV_MOMENTUM_HPP_
#define OPTIMIZATION_NESTROV_MOMENTUM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @class NestrovMomentum
 * @brief 네스테로프 가속 경사(Nesterov Accelerated Gradient, NAG) 최적화 알고리즘
 * 
 * @details 일반적인 모멘텀(Momentum) 방식은 현재 위치에서 기울기를 계산한 뒤 속도(Velocity)에 더하여 이동합니다.
 * 반면, 네스테로프 모멘텀(NAG) 방식은 현재 지니고 있는 관성(Velocity)을 이용해 미리 미래 위치(Look-ahead)를 예상하고,
 * 그 "예상된 미래 위치"에서의 기울기를 계산하여 현재 속도를 업데이트합니다.
 * 
 * 이를 통해 알고리즘이 내리막길의 끝(최적점)을 미리 보고 관성을 제어할 수 있어, 일반 모멘텀에 비해 
 * 극심한 오버슈팅(Overshooting)이나 진동(Oscillation)을 크게 줄이면서 수렴 속도를 높일 수 있습니다.
 * 
 * 수식:
 * x_lookahead = x_t + beta * v_t
 * v_{t+1} = beta * v_t - alpha * g(x_lookahead)
 * x_{t+1} = x_t + v_{t+1}
 */
class NestrovMomentum {
   public:
    // ==============================================================
    // Algorithm 5.3 : Nesterov Momentum
    // 미래의 위치(Look-ahead)를 먼저 예측하고 그곳의 기울기를 이용해 업데이트합니다.
    // ==============================================================
    /**
     * @brief 네스테로프 모멘텀을 이용해 함수의 최솟값을 찾습니다.
     * 
     * @param f 최적화할 목적 함수 (AutoDiff를 통한 기울기 평가가 가능해야 함)
     * @param x 탐색을 시작할 초기 위치 (N차원 벡터)
     * @param alpha 학습률 (Learning Rate). 기울기를 반영하는 스텝 크기 (기본값: 0.001)
     * @param beta 모멘텀 계수. 과거의 속도를 얼마나 유지할지 결정 (기본값: 0.9)
     * @param max_iter 최대 반복 횟수 (기본값: 15000)
     * @param tol 허용 오차. 예상된 미래 위치의 기울기 노름(Norm)이 이 값 미만이면 수렴으로 판단
     * @param verbose 콘솔에 진행 과정을 출력할지 여부
     * @return 최적화된 위치 벡터(x_opt) 반환
     */
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.001,
                                          double beta = 0.9, size_t max_iter = 15000,
                                          double tol = 1e-4, bool verbose = false) {
        std::array<double, N> v = {0.0};  // 속도(Velocity) 누적 변수 초기화

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🔮 Nesterov Momentum Started (alpha=" << alpha << ", beta=" << beta
                      << ")\n";
            std::cout << "========================================================\n";
        }
        
        for (size_t iter = 1; iter <= max_iter; ++iter) {
            // -----------------------------------------------------------
            // 1. 미래의 위치 (Look-ahead point) 계산
            // 일반 모멘텀과 달리, 현재 위치(x)에서 현재 관성(beta * v)만큼 먼저 이동해봅니다.
            // -----------------------------------------------------------
            std::array<double, N> x_lookahead;
            for (size_t i = 0; i < N; ++i) {
                x_lookahead[i] = x[i] + beta * v[i];
            }

            // -----------------------------------------------------------
            // 2. 미래 위치에서의 함수값 및 기울기(g_lookahead) 획득
            // AutoDiff를 통해 미래 위치(x_lookahead)에서의 정확한 기울기를 계산합니다.
            // 이 기울기는 "앞으로 내리막이 계속될지, 오르막으로 변할지"에 대한 정보를 담고 있습니다.
            // -----------------------------------------------------------
            double f_lookahead;
            std::array<double, N> g_lookahead;
            AutoDiff::value_and_gradient<N>(f, x_lookahead, f_lookahead, g_lookahead);

            // 종료 조건 검사: "미래 위치"의 기울기 노름(Norm) 계산
            double g_norm = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm += g_lookahead[i] * g_lookahead[i];
            }
            g_norm = std::sqrt(g_norm);

            // 기울기 노름이 허용 오차보다 작으면 최적점(임계점)에 도달한 것으로 판단하고 종료
            if (g_norm < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // -----------------------------------------------------------
            // 3. 네스테로프 업데이트 (파라미터 갱신)
            // 미래 위치에서 구한 기울기를 현재 속도 업데이트에 반영하여 실제 이동할 위치를 계산합니다.
            // -----------------------------------------------------------
            for (size_t i = 0; i < N; ++i) {
                // 속도 업데이트: 기존 관성 유지 + (미래 위치에서의 기울기 기반 하강)
                v[i] = beta * v[i] - alpha * g_lookahead[i];
                
                // 위치 업데이트: 계산된 새로운 속도를 이용해 현재 위치를 이동시킵니다.
                x[i] = x[i] + v[i];  
            }

            // (선택) 진행 과정 출력 로직을 추가하려면 이곳에 배치할 수 있습니다.
        }

        if (verbose) {
            std::cout << "========================================================\n";
            // 차원 N과 상관없이 유연하게 출력할 수 있도록 루프 처리
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            for (size_t i = 1; i < N; ++i) {
                std::cout << ", " << x[i];
            }
            std::cout << "]\n========================================================\n";
        }
        
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_NESTROV_MOMENTUM_HPP_