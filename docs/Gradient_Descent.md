# Algorithm 5.1 : 경사 하강법 (Gradient Descent with Line Search)
## 1. 수학적 원리 (Mathematical Formulation)
목적 함수가 부드럽고 (Smooth) 미분 가능하다면, 함수가 가장 가파르게 감소하는 방향 (Direction of steepest descent)은 기울기 (Gradient, $nabla f(x)$)의 정확히 반대 방향입니다. 경사 하강법은 이 1차 미분 (First-order) 정보를 활용하여 국소 최솟값 (Local minimum)을 향해 한 걸음씩 나아가는 직관적이고 강력한 방법입니다.

### 탐색 방향
현재 위치 $x^{(k)}$에서의 기울기를 $g^{(k)}$라 할 때, 하강 방향은 역방향 기울기입니다.

$$g^{(k)} = \nabla f(x^{(k)})$$
$$d^{(k)} = -g^{(k)}$$

### 위치 업데이트 (Position Update)
$$x^{(k)} = x^{(k)} + \alpha^{(k)} d^{(k)}$$

여기서 $\alpha^{(k)}$는 보폭 (Step size 또는 Learning rate)입니다. 보폭이 너무 크면 최솟값을 지나쳐 발산할 위험이 있고, 수렴이 극도로 느려집니다. 따라서 이 구현에는 고정된 $\alpha$ 대신, 4장에서 설계한 **선탐색 (Line Search)을 통해 매 반복마다 최적의 보폭을 수학적으로 찾아냅니다**

---
## 2. C++ 설계 (Code Implementation)
`AutoDiff`를 통한 $O(1)$ 해석적 미분과 `StrongBacktrackingLineSearch`를 결합하여 안정성을 극대화한 실차 제어기용 솔버입니다.

```c++
#ifndef OPTIMIZATOIN_GRADIENT_DESCENT_HPP_
#define OPTIMIZATOIN_GRADIENT_DESCENT_HPP_

#include <array>
#include <functional>
#include <cmath>
#include <iostream>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/StrongBacktrackingLineSearch.hpp"

namespace Optimization {
    class GradientDescent {
        public:
            template <size_t N, tyename Func>
            static std::array<double, N> optimize(Func f, std::array<double, N> x, double tol = 1e-6, int max_iter = 1000, bool verbose = false) {
                if (verbose)
                    std::cout << "🚀 Gradient Descent Started...\n";
                for (int i = 0; i < max_iter; ++i) {
                    // 1. 현재 위치의 함수값 (f_val)과 기울기 (grad)를 AutoDiff로 정확히 계산
                    double f_val;
                    std::array<double, N> grad;
                    AutoDiff::value_and_gradient<N>(f, x, f_x, grad);

                    // 2. 종료 조건 (Convergence Check) : 기울기의 L2 Norm이 0에 수렴했는지 확인
                    double g_norm = 0.0;
                    for (double g : grad) {
                        g_norm += g * g;
                    }
                    g_norm = std::sqrt(g_norm);

                    if (verbose) { /* ... 로그 출력 ... */ }

                    // 기울기가 허용 오차 (tol)보다 작으면 바닥에 도착한 것으로 간주하고 탈출
                    if (g_norm < tol)
                        break;
                    
                    // 3. 하강 방향 설정 : 기울기의 정확히 반대 방향 (Steepest Descent)
                    std::array<double, N> direction;
                    for (size_t j = 0; j < N; ++j) {
                        direction[j] = -grad[j];
                    }

                    // 4. [핵심] 보폭 (alpha) 자동 결정 : 강한 울프 조건 (Strong Wolfe)을 만족하는 안전한 스텝
                    double alpha = StrongBacktrackingLineSearch::search<N>(f, x, direction);

                    // 5. 위치 업데이트 (x_new = x + alpha * d)
                    for (size_t j = 0; j < N; ++j) {
                        x[j] += alpha * direction[j];
                    }
                }
                return x;
            }
    };
} // namespace Optimization

#endif // OPTIMIZATOIN_GRADIENT_DESCENT_HPP_
```

## 3. Technical Review
1. **하이퍼파라미터 리스크 제거 (Safe Execution)** :   
일반적인 머신러닝/딥러닝 환경의 경사 하강법은 사용자가 학습률 $\alpha$를 임의로 지정해야 합니다 (예 : 0.1, 0.01). 하지만 NMPC (Model Predictive Control)가 구동되는 주행 환경에서는 타이어 슬립이나 노면 마찰 계수에 따라 목적 함수의 곡률이 매초 수십 번씩 변합니다. 고정된 $\alpha$를 사용하면 어느 순간 제어기가 크게 발산하여 차량이 불안정해질 수 있습니다. `StrongBacktrackingLineSearch`를 호출하여 **"함수값이 충분히 줄어들면서도 기울기가 평탄해지는"** 완벽한 보폭을 매번 찾아내므로, 어떠한 악조건의 지형 (Obejctive function landscape)이 들어와도 제어기가 발산하지 않는 강력한 유지력을 보장합니다.

2. **$O(N)$ 메모리 효율성** :   
BFGS나 뉴턴 방법은 최소 $O(N^{2})$ 크기의 행렬을 연산해야 합니다. 그러나 경사 하강법 코드는 상태 공간이 아무리 커져도 (예 : $N = 100$) 길이 $N$짜리 1차원 배열 (`std::array`) 2 ~ 3개로만 연산을 수행합니다. RAM이 수백 KB에 불과한 하위 제어기 (MCU)에 최적화 모듈을 올려야 할 때 가장 먼저 투입할 수 있는 "가볍고 빠른" 코드입니다.

3. **한겨점 : 좁고 긴 골짜기에서의 지그재그 현상** :   
가장 큰 약점은 교재에서도 지적해듯, 목적 함수가 길고 좁은 계곡 (Narrow Valleys) 형태일 때 나타납니다. 선탐색을 통해 최적의 $\alpha$를 찾고 나면, 다음 하강 방향은 이전 방향과 직교 (Orthogonal)하게 되어버려, 계곡 바닥을 지그재그로 오가며 수렴 속도가 매우 느려집니다.  
그러므로 $H$ 행렬의 조건수 (Condition number)가 좋을 때 (둥근 밥그릇 모양) 쓰거나, 혹은 BFGS나 공액 기울기법 (Conjugate Gradient)이 수치적 오류로 실패했을 때 **가장 안전하게 동작은 이어받는 Fallback (백업) 솔버**로 구성하는 것이 최적입니다.