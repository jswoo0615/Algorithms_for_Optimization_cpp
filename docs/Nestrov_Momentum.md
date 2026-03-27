# Algorithm 5.4 : 네스테로프 모멘텀 (Nestrov Momentum)
## 1. 수학적 원리 (Mathematical Formulation)
일반 모멘텀 (Algorithm 5.3)의 가장 큰 문제는 관성이 너무 크게 붙었을 때 계곡의 바닥 (최솟값)에 도달해도 제 때 멈추지 못하고 반대편 언덕으로 올라가는 **오버슈팅 (Overshooting)** 현상입니다.  
**네스테로프 모멘텀 (Nestrov Momentum)**은 이 문제를 해결하기 위해, 현재 위치가 아닌 **현재 관성대로 흘러갔을 때 도달하게 될 미래의 위치 (Look-ahead point)**에서 기울기를 미리 계산합니다. 만약 미래의 위치가 오르막길이라면 기울기 (Gradient)가 반대 방향을 가리키게 되므로, 알고리즘이 바닥에 도달하기 전에 미리 **브레이크 (Brake)**를 밟아 감속합니다.

### 네스테로프 업데이트 공식
$$v^{(k + 1)} = \beta v^{(k)} - \alpha \nabla f(x^{(k)} + \beta v^{(k)})$$
$$x^{(k + 1)} = x^{(k)} + v^{(k + 1)}$$
* $x^{(k)} + \beta v^{(k)}$ : 미래 예측 위치 (Look-ahead point)

---
## 2. C++ 설계 (Code Implementation)
미래 위치 (`x_lookahead`)를 먼저 도출하고, 해당 지점에서의 1차 미분 정보 (`g_lookahead`)를 평가하여 브레이크를 밟는 메커니즘을 $O(N)$의 1차원 정적 배열 (`std::array`) 만으로 구현했습니다.

```C++
#ifndef OPTIMIZATION_NESTROV_MOMENTUM_HPP_
#define OPTIMIZATION_NESTROV_MOMENTUM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"
namespace Optimization {
    class NestrovMomentum {
        public:
            template <size_t N, typename Func>
            static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.001, double beta = 0.9, size_t max_iter = 15000, double tol = 1e-4, bool verbose = false) {
                // 초기 속도 (Velocity) 벡터를 0으로 초기화
                std::array<double, N> v = {0.0};

                if (verbose) { /* ... 로그 출력 ... */ }

                for (size_t iter = 1; iter <= max_iter; ++iter) {
                    // 1. 미래의 위치 (Look-ahead point) 계산 : 관성 (beta * v)대로 먼저 가본다
                    std::array<double, N> x_lookahead;
                    for (size_t i = 0; i < N; ++i) {
                        x_lookahead[i] = x[i] + beta * v[i];
                    }

                    // 2. 미래 위치에서의 함수값 및 기울기 (g_lookahead) 획득
                    double f_lookahead;
                    std::array<double, N> g_lookahead;
                    AutoDiff::value_and_gradient<N>(f, x_lookahead, f_lookahead, g_lookhead);

                    // 종료 조건 검사 (미래 위치의 기우릭 노름 기준)
                    double g_norm = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        g_norm += g_lookahead[i] * g_lookahead[i];
                    }
                    g_norm = std::sqrt(g_norm);

                    if (g_norm < tol) {
                        if (verbose) 
                            std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                        break;
                    }

                    // 3. 네스테로프 업데이트 (미래의 기울기를 반영하여 속도 갱신 및 실제 위치 이동)
                    for (size_t i = 0; i < N; ++i) {
                        // 선제적 브레이킹이 포함된 새로운 속도
                        v[i] = beta * v[i] - alpha * g_lookahead[i];

                        // 실제 위치 업데이트
                        x[i] = x[i] + v[i];
                    }
                }

                if (verbose) { /* ... 로그 출력 ... */ }
                return x;
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_NESTROV_MOMENTUM_HPP_
```

---
## 3. Technical Review
1. **제어공학의 D-제어기 (Derivative)와 유사한 '예측 방어막'** :  
차량 제어 (예: 스마트 크루즈 컨트롤, 차선 유지 보조)에서 일반 모멘텀은 오차를 계속 누적하는 적분기 (Integral, I-Control)와 같아서 타겟 속도나 차선을 넘어가는 오버슈팅 (Overshooting)을 유발합니다. 반면 네스테로프 모멘텀은 **미래 오차 (Look-ahead gradient)를 미래 내다보고 감속**하기 때문에 PID 제어기의 미분기 (Derivative, D-Control)와 매우 유사한 감쇠 (Damping) 역할을 합니다. 실차 테스트 시 승차감을 저해하는 Jert (울컥거림) 현상을 줄여줄 수 있습니다

2. **비용 대비 극강의 성능** :  
이 알고리즘은 여전히 무거운 선탐색 (Line Search)이나 헤시안 (Hessian) 행렬 연산 없이, 매 Iteration 마다 **단 1회의 기울기 평가 (`AutoDiff`)와 1차원 배열 덧셈/곱셈만 수행**합니다. 연산 자원이 극도로 적은 저사용 마이크로컨토를러 (MCU)에서 NMCP 최적화를 수행할 때, 가장 빠르면서도 진동 없이 수렴하는 최고의 1차 (First-Order) 타협책입니다.

3. **한계 및 확장 (Learning Rate Tuning)** :  
네스테로프 모멘텀 역시 고정된 학습률 (Learning Rate, $\alpha$)에 크게 의존합니다. 환경 변화에 따라 곡률이 달라지면 $\alpha$를 튜닝해야 하는 부담이 여전히 존재합니다. 따라서 딥러닝이나 복잡한 궤적 생성 문제에서는 이 네스테로프의 장점 (관성 및 미래 예측)에 **각 차원 (Diemsion)별로 보폭을 자동 조절**하는 기능까지 합친 Adam 또는 Nadam (Nestrov-Accelerated Adaptive Moment Estimation)으로 발전하게 됩니다.