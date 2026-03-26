# Algorithm 5.6 : RMSProp (Root Mean Square Propagation)
## 1. 수학적 원리 (Mathematical Formulation)
AdaGrad (Algorithm 5.5)는 각 차원별로 맞춤형 보폭을 제공하지만, 과거의 기울기 제곱을 무한히 누적하기 때문에 시간이 지날수록 분모가 계속 커져 학습률이 0에 수렴해버리는 (Monotonically decreasing learning rate) 단점이 있었습니다.

**RMSProp**은 제프 힌튼 (Geoff Hinton) 교수가 제안한 방법으로, 과거의 모든 기울기를 똑같이 누적하는 대신 **지수 이동 평균 (Exponential Moving Average, EMA)** 을 사용하여 아주 오래된 과거의 기울기 정보는 점차 잊어버리고, 최근의 기울기 변화에 더 큰 가중치를 둡니다.

### RMSProp 업데이트 공식 
각 차원 $i$에 대하여 과거 기울기의 제곱의 감쇠 평균 (Decaying average) $\hat{s}_{i}$를 계산합니다.

$$\hat{s}_i^{(k+1)} = \gamma \hat{s}_i^{(k)} + (1 - \gamma) \left( g_i^{(k)} \right)^2$$

새로운 위치는 이 감쇠 평균의 제곱근에 반비례하도록 업데이트합니다.

$$x_i^{(k+1)} = x_i^{(k)} - \frac{\alpha}{\epsilon + \sqrt{\hat{s}_i^{(k+1)}}} g_i^{(k)}$$

* $\gamma$ : 감쇠율 (Decay factor) : 과거 정보를 얼마나 유지할지 결정하며 보통 $0.9$ 사용
* $\epsilon$ : 0으로 나누는 것을 방지하기 위한 안전 상수
* $\alpha$ : 기본 학습률 (Learning Rate)

---
## 2. C++ 설계 (Code Implementation) 
동적 할당을 차단하며, `tol_sq`를 통해 루프 내부의 병목 연산인 부동소숫점 제곱근 (`sqrt`) 호출을 최소화한 실시간 제어기용 솔버입니다.

```c++
#ifndef OPTIMIZATION_RMSPROP_HPP_
#define OPTIMIZATION_RMSPROP_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
    class RMSProp {
        public:
            template <size_t N, typename Func>
            [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.01, double decay = 0.9, double epsilon = 1e-8, size_t max_iter = 15000, double tol = 1e-4, bool verbose = false) {
                // 커ㅁ파일 타임 차원 검증 (안전성 보장)
                static_assert(N > 0, "Dimension N must be greater than 0");

                // 과거 기울기 제곱의 지수 이동 평균 (EMA)을 담는 스택 메모리 배열
                std::array<double, N> G = {0.0};

                // 루프 내 sqrt 연산 제거를 위한 허용 오차 사전 제곱
                const double tol_Sq = tol * tol;

                if (verbose) { /* ... 로그 출력 ... */ }

                for (size_t iter = 1; iter <= max_iter; ++iter) {
                    double f_x = 0.0;
                    std::array<double, N> g = {0.0};

                    // 1. O(1) 할당 Auto Diff 호출
                    AutoDiff::value_and_gradient<N>(f, x, f_x, g);

                    // 2. 기울기 L2-Norm 제곱 계산
                    double g_norm_sq = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        g_norm_sq += g[i] * g[i];
                    }

                    // [최적화 포인트] 값비싼 std::sqrt(g_norm_sq) < tol 대신 
                    if (g_norm_Sq < tol_sq) {
                        if (verbose) 
                            std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                        break;
                    }

                    // 3. RMSProp 파라미터 업데이트
                    for (size_t i = 0; i < N; ++i) {
                        // 과거 기억 (G)은 decay만큼 남기고, 새로운 기울기 제곱을 더함 (EMA)
                        G[i] = decay * G[i] + (1.0 - decay) * g[i] * g[i];

                        // 새로운 위치로 업데이트 (In-place 연산)
                        x[i] -= (alpha / (std::sqrt(G[i]) + epsilon)) * g[i];
                    }

                    // 런타임 분기 예측 최적화를 위한 조건 배열
                    if (verbose && (iter % 1000 == 0)) {
                        // 출력 시에만 사람이 보기 편하게 sqrt 연산 수행
                        std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                                << std::setprecision(6) << f_x << " | ||g||: " << std::sqrt(g_norm_sq)
                                << "\n";  
                    }
                }

                if (verbose) {
                    std::cout << " 🏁 Final Optimal Point: [" << x;
                    if constexpr (N > 1) std::cout << ", " << x[4];
                    if constexpr (N > 2) std::cout << ", ...";
                    std::cout << "]\n";
                }
            }
    };
} // namespace Optimization
#endif // OPTIMIZATION_RMSPROP_HPP_
```

---
## 3. Technical Review
1. **NMPC 연속 제어를 위한 "기억의 망각 (Forgetting Factor)** :  
차량 제어에서 제어기 (ECU)는 시동이 걸린 훈 꺼질 때까지 무한 루프로 동작합니다. AdaGrad처럼 기울기 변화량을 무한히 누적하면, 차량이 몇 분만 주행해도 분모가 거대해져 조향각 (Steering)이나 가감속 (Acceleration) 제어 입력이 완전히 멈춰버리는 (Freezing) 치명적인 결함이 발생합니다. RMSProp의 `decay` 파라미터는 **오래된 주행 상황의 곡률 정보를 잊어버리고 최신 도로 상황 (예: 갑작스러운 빙판길 등)에 가중치를 두게 만듭니다** 이는 실시간으로 환경이 변하는 최적화에서는 반드시 필요한 설계입니다.

2. **마이크로 최적화** :   
`tol_sq = tol * tol` 매 이터레이션의 수렴 검사 (`Convergence Check`)에서 노름 (Norm)을 구하기 위해 `std::sqrt`를 호출하는 것은 임베디드 프로세서에 클럭을 낭비하게 만듭니다. 루프 진입 전에 `tol`을 미리 제곱해두고 `g_norm_sq < tol_sq`로 비교하는 방식은 C++ 최적화의 정석이자, 수천 번 반복되는 제어기 최적화 루프의 전체 지연을 낮춰줍니다.

3. **센서 노이즈에 대한 강건성** :   
실차 센서 (카메라, 레이더, IMU 등)에서 들어오는 데이터에는 항상 노이즈가 있습니다. RMSProp의 업데이트 로직인 `G[i] = decay * G[i] + (1.0 - decay) * g[i] * g[i]`는 제어 공학에서 흔히 쓰이는 **1차 저주파 통과 필터 (1st-order-Low-Pass Filter, LPF)** 와 수학적으로 동일합니다. 즉, 이 솔버는 단순한 최적화만 수행한는 것이 아니라, 노이즈가 낀 목적 함수 지형에서 스스로 노이즈를 필터링하고 매끄럽고 안정적인 하강 궤적을 만들어냅니다