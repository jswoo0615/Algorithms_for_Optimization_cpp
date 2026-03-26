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
    class RMSProp {};
} // namespace Optimization

#endif // OPTIMIZATION_RMSPROP_HPP_
```