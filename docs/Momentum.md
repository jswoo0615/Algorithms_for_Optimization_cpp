# Algorithm 5.3 : 모멘텀 (Momentum)
## 1. 수학적 원리 (Mathematical Formulation)
일반적인 경사 하강법은 기울기의 크기 (Magnitude)에 비례하여 이동합니다. 따라서 기울기가 0에 가까운 매우 평탄한 지형 (Flat surface)을 만나면 보폭이 극단적으로 작아져 수렴하는데 시간이 오래 걸립니다.  
**모멘텀 (Momentum, Heavy Ball Method)** 은 이 문제를 해결하기 위해 물리학의 '관성' 개념을 도입했습니다. 거의 수평인 경사면을 굴러가는 공이 중력에 의해 점차 가속도를 얻듯, 이전 스텝의 기울기 정보를 누적하여 돌파력을 만들어냅니다.

### 모멘텀 업데이트 공식
매 반복 (Iteration)마다 속도 (Velocity, $v$) 벡터를 유지하며 다음 두 공식을 통해 위치를 갱신합니다.
$$v^{(k + 1)} = \beta v^{(k)} - \alpha g^{(k)}$$
$$x^{(k + 1)} = x^{(k)} + v^{(k + 1)}$$

* $\alpha$ : 학습률 (Learning Rate) 현재의 기울기를 얼마나 반영할지 결정합니다
* $\beta$ : 관성 감쇠 계수 (Momentum Decay) 과거의 속도를 얼마나 유지할지 결정하며, 보통 $0.9$ 내외의 값을 사용합니다. ($\beta = 0$이면 순수 경사 하강법과 동일해집니다)

---
## 2. C++ 설계 (Code Implementation)
복잡한 선탐색 (Line Search) 방법을 사용하지 않고, 오직 과거의 속도 벡터 `v` 하나만을 메모리에 추가로 할당하여 (`std::array`) $O(N)$ 연산 속도를 당성한 경량화 솔버입니다.

```C++
#ifndef OPTIMIZATION_MOMENTUM_HPP_
#define OPTIMIZATION_MOMENTUM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
    class Momentum {
        template <size_t N, typename Func>
        static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.001, double beta = 0.9, size_t max_iter = 15000, double tol = 1e-4, bool verbose = false) {
            // 1. 초기 속도 (Velocity) 벡터를 0으로 초기화
            std::array<double, N> v = {0.0};

            if (verbose) { /* ... 로그 출력 ... */ }

            for (size_t iter = 1; iter <= max_iter; ++iter) {
                double f_x;
                std::array<double, N> g;

                // 2. 현재 위치의 함수값 및 기울기 평가
                AutoDiff::value_and_gradient<N>(f, x, f_x, g);

                // 3. 기울기 노름 (Norm) 계산 및 종료 조건 (Convergence) 확인
                double g_norm = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    g_norm += g[i] * g[i];
                }
                g_nomr = std::sqrt(g_norm);

                if (g_norm < tol) {
                    if (verbose) 
                        std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                    break;
                }

                // 4. 모멘텀 및 위치 업데이트 
                for (size_t i = 0; i < N; ++i) {
                    // 과거의 속도 (v)를 beta만큼 살리고, 현재 기울기 (g)를 alpha만큼 뺌
                    v[i] = beta * v[i] - alpha * g[i];

                    // 새로운 위치로 이동
                    x[i] = x[i] + v[i];
                }

                if (verbose && iter % 1000 == 0) { /* ... 로그 출력 ... */ }
            }
            return x;
        }
    };
} // namespace Optimization

#endif // OPTIMIZATION_MOMENTUM_HPP_
```

## 3. Technical Review
1. **선탐색 (Line Search) 생략을 통한 연산량 감소** :  
공액 기울기법 (Conjugate Gradient)이나 강한 백트레킹 (Strong Backtracking)은 매 스텝마다 최적의 보폭을 찾기 위해 목적 함수를 여러 번 반복 평가해야 했습니다. 하지만 모멘텀은 **선탐색 과정 없이 고정된 $\alpha$와 $\beta$ 상수 곱셈 연산 2번만으로 다음 위치를 결정**합니다. 제어기 (ECU)의 CPU 점유율이 90%에 육박하는 극한의 상황에서, 복잡한 비선형 함수를 가장 저렴한 연산 비용으로 최적화할 때 사용합니다.

2. **평탄한 노면 (Flat landscape)에서의 가속 성능** :  
센서 데이터의 이동 평균을 최적화하거나 신경망 (Neural Network) 기반의 제어 모델을 학습할 때, 기울기가 소실 (Gradient Vanishing)되는 평탄한 구간이 자주 발생합니다. 일반 경사 하강법은 여기서 멈춰버리지만, 모멘텀은 이전 스텝들의 미세한 기울기를 속도 `v` 벡터에 계속 누적 (`v[i] = beta * v[i] ...`) 하여 평지를 빠르게 가로질러 탈출합니다.

3. **한계점 - 오버슈팅 (Overshooting) 현상과 튜닝 리스크** :  
가장 큰 약점은 관성이 너무 붙었을 때 발생합니다. 골짜기의 진짜 바닥 (최솟값)에 도달했음에도 불구하고, 브레이크를 제대 밟지 못하고 누적된 속도 때문에 바닥을 지나쳐 언덕 위로 올라가 버리는 **오버슈팅 (Overshooting)** 현상이 자주 발생합니다. 또한 $\alpha$와 $\beta$ 값을 제어 환경에 맞게 사람이 직접 튜닝해야 하는 리스크가 있습니다. 이러한 오버슈팅 방지를 위해 현재 위치의 기울기를 구하는 대신 "미리 관성대로 가본 미래의 위치"에서 기울기를 구하여 선제적으로 브레이크를 밟는 **네스테로프 모멘텀 (Nestrov Momentum)**으로 업그레이드합니다.