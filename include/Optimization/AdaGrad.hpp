#ifndef OPTIMIZATION_ADAGRAD_HPP_
#define OPTIMIZATION_ADAGRAD_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief AdaGrad (Adaptive Gradient) 최적화 알고리즘을 구현한 정적 클래스
 * 
 * AdaGrad는 경사 하강법(Gradient Descent)의 변형으로, 모든 매개변수(최적화 변수)에 대해 
 * 동일한 학습률(Learning Rate)을 적용하는 대신, **각 매개변수마다 맞춤형 학습률을 계산**하여 적용하는 알고리즘입니다.
 * 
 * 핵심 아이디어:
 * 과거에 많이 변화했던(기울기가 컸던) 변수는 학습률을 작게 하여 세밀하게 조정하고,
 * 과거에 거의 변화하지 않았던(기울기가 작았던) 변수는 학습률을 크게 하여 빠르게 학습되도록 유도합니다.
 * 이는 희소한(Sparse) 특징을 가진 데이터나 함수에서 특히 효과적입니다.
 * 
 * 단점:
 * 과거 기울기의 제곱이 계속 누적만 되기 때문에($G_i$ 가 단조 증가), 
 * 반복 횟수가 많아질수록 학습률이 $0$에 수렴하여 결국 학습이 멈추는(Premature Convergence) 문제가 발생할 수 있습니다. 
 * (이를 해결하기 위해 등장한 것이 RMSProp, AdaDelta 등입니다.)
 * 
 * @note MISRA C++ 및 실시간 제어(RT) 코딩 표준 준수:
 *       동적 메모리 할당(힙 할당)을 완전히 배제하고 O(1) 정적 배열 및 In-place 연산을 사용하여 속도와 안정성을 극대화했습니다.
 */
class AdaGrad {
   public:
    // 상태를 저장하지 않는 정적(Static) 유틸리티 클래스로만 사용되도록 인스턴스화를 방지합니다.
    AdaGrad() = delete; 

    // ==============================================================================
    // Algorithm 5.4 : AdaGrad
    // 차원별로 과거 기울기의 제곱을 누적하여 맞춤형 학습률 제공
    // ==============================================================================
    /**
     * @brief AdaGrad 메인 최적화 함수
     * 
     * @tparam N 최적화 변수의 차원
     * @tparam Func 목적 함수 타입 (람다식, 펑터 등)
     * @param f 최소화할 목적 함수
     * @param x 최적화 시작 지점 (초기 추정값)
     * @param alpha 초기 글로벌 학습률 (기본값: 0.1)
     * @param epsilon 0으로 나누어지는 것을 방지하기 위한 수치적 안정성 상수 (작은 양수, 기본값: 1e-8)
     * @param max_iter 무한 루프를 방지하기 위한 최대 허용 반복 횟수
     * @param tol 수렴 판정을 위한 허용 오차 (현재는 수렴 판정에 epsilon 파라미터를 그대로 재사용하고 있습니다)
     * @param verbose 콘솔에 최적화 진행 상황을 출력할지 여부
     * @return std::array<double, N> 최적화가 완료된 해(Optimal Point)
     */
    template <size_t N, typename Func>
    [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x,
                                                        double alpha = 0.1, double epsilon = 1e-8,
                                                        size_t max_iter = 15000, double tol = 1e-4,
                                                        bool verbose = false) {
        // 0차원 배열이 주입되어 발생하는 런타임/컴파일 타임 오류를 원천 차단합니다.
        static_assert(N > 0, "Dimension N must be greater than 0.");
        
        // G: 각 변수(차원)별로 과거 기울기 제곱을 누적하여 저장하는 배열 ($G_i = \sum_{t=1}^k g_{i,t}^2$)
        // 동적 할당이 없는 스택(Stack) 메모리인 std::array를 사용하여 실시간성을 보장합니다.
        // 제어 이론의 관례에 따라 속도(Velocity, v) 모멘텀과 혼동되지 않도록 변수명 G를 사용합니다.
        std::array<double, N> G = {0.0};

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🎯 AdaGrad Optimizer Started (alpha=" << alpha << ", eps=" << epsilon
                      << ")\n";
            std::cout << "========================================================\n";
        }

        // 최적화 반복 루프
        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x = 0.0; // 현재 위치에서의 목적 함수 값
            std::array<double, N> g = {0.0}; // 현재 위치에서의 목적 함수 기울기 벡터 (Gradient)

            // 1. Auto Diff (자동 미분) 호출
            // 해석적으로 정확한 미분값을 O(1) 정적 메모리 할당만으로 초고속으로 계산합니다.
            AutoDiff::value_and_gradient<N>(f, x, f_x, g);

            // 2. 기울기 L2-Norm 계산 및 조기 종료(수렴) 검증
            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g[i] * g[i];
            }

            const double g_norm = std::sqrt(g_norm_sq);
            
            // 기울기의 크기(Norm)가 허용 오차(여기서는 epsilon 파라미터 활용)보다 작아지면,
            // 1차 필요 조건(평고점)에 도달했다고 판단하여 최적화를 종료합니다.
            if (g_norm < epsilon) {
                if (verbose) {
                    std::cout << "✅ Convergence achieved at iteration " << iter
                              << " with gradient norm " << g_norm << ".\n";
                }
                break;
            }

            // 3. AdaGrad 파라미터 업데이트 
            for (size_t i = 0; i < N; ++i) {
                // [Step 3-1] 차원별로 과거 기울기의 제곱을 계속 누적합니다.
                G[i] += g[i] * g[i];  
                
                // [Step 3-2] 차원별 맞춤형 학습률을 적용하여 변수를 업데이트합니다.
                // 공식: x_{i} = x_{i} - \frac{\alpha}{\sqrt{G_i} + \epsilon} \cdot g_i
                // [수치적 안정성] epsilon을 \sqrt{G_i} 바깥에 더하여 분모가 0이 되어 발산(Divergence)하는 것을 방지합니다.
                // [실시간성/최적화] '-=' In-place 연산자를 사용하여 캐시 메모리 히트율을 극대화합니다.
                x[i] -= (alpha / (std::sqrt(G[i]) + epsilon)) * g[i];
            }

            // 4. 진행 상태 로깅
            // 런타임 분기 예측(Branch Prediction) 최적화를 위해 % 연산을 최소화하고자, 
            // 반복 1000회당 1번만 조건문에 진입하도록 설계되었습니다.
            if (verbose && (iter % 1000 == 0)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << g_norm << "\n";
            }
        }
        
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            // C++17 constexpr if를 활용하여 N차원(1차원, 2차원, 3차원 이상)에 따른 
            // 안전하고 유연한 출력 포맷을 컴파일 타임에 보장합니다.
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }
        
        // 최종적으로 업데이트된 최적해 배열 반환
        return x;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_ADAGRAD_HPP_