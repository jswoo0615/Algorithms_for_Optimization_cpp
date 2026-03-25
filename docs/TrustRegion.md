# Algorithm 4.3 : 신뢰 영역 기법 (Trust Region Method w/ Simplified Dogleg)
## 1. 수학적 원리 (Mathematical Formulation)
선탐색 (Line Search)이 '방향'을 먼저 정하고 '보폭'을 찾았다면, 신뢰 영역 방법 **현재의 2차 테일러 근사 모델을 믿을 수 있는 최대 반경 $(\delta)$** 을 먼저 설정하고, 그 반경 안에서 최적의 이동 벡터 $p$를 찾습니다.

매 스텝마다 풀어야 하는 2차 부분 문제 (Quadratic Subproblem)는 다음과 같습니다

$$\min_{\mathbf{p}} m(\mathbf{p}) = f(\mathbf{x}) + \mathbf{g}^T \mathbf{p} + \frac{1}{2} \mathbf{p}^T \mathbf{H} \mathbf{p} \quad \text{subject to} \quad \|\mathbf{p}\| \le \delta$$

이 코드는 위 최적화 문제를 풀기 위해 **코시 점 (Cauchy Point)** 과 **뉴턴 스텝 (Newton Step)** 을 결합한 Dogleg 방식의 근사 해봅을 사용합니다.
1. 뉴턴 스텝 $(p_{newton})$ : 제약이 없을 때의 최적점 $p_{newton} = -H^{-1} g$
2. 코시 점 $(p_{cauchy})$ : 가장 가파른 내리막 방향 $(-g)$으로 모델 $m(p)$를 최소화하는 점. 이동거리 = $\tau == \frac{g^{T}g}{g^{T}Hg}$

### 신뢰도 (Ratio, $\rho$) 평가 및 반경 업데이트
예측된 모델의 감소량과 실제 목적 함수의 감소량 비율을 통해 반경 $\delta$를 조절합니다.

$$\rho = \frac{\text{Actual Reduction}}{\text{Predicted Reduction}} = \frac{f(x) - f(x + p)}{m(0) - m(p)}$$

* $\rho < 0.25$ : 예측이 틀림 $\rightarrow$ 반경 축소 $(\delta \leftarrow 0.25\delta)$
* $\rho > 0.75$ 및 경계 도달 : 예측이 매우 정확함 $\rightarrow$ 반경 확장 $(\delta \leftarrow min(2\delta, \delta_{max}))$
* $\rho > 0.15$ : 이동 수용 $(x \leftarrow x + p)$

---
## 2. C++ 설계 (Code Implementation)
루프 내부에서 동적 할당이나 무거운 행렬 라이브러리 (Eigen 등)의 호출 없이, 2차원 제어 공간에 대해 해석적 (Analytical) 역행렬을 계산하여 $O(1)$의 결정론적 (Deterministic) 속도를 보장하는 실차용 솔버입니다.

```c++
namespace Optimization {
    class TrustRegion {
        private:
            static double norm2(const std::array<double, 2>& v) {
                return std::sqrt(v[0] * v[0] + v[1] * v[1]);
            }

            static std::array<double, 2> solve_newton(const std::array<std::array<double, 2>, 2>& H, const std::array<double, 2>& g) {
                double det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
                if (std::abs(det) < 1e-12) {
                    return {-g[0], -g[1]};
                }
                double inv00 = H[1][1] / det;
                double inv01 = -H[0][1] / det;
                double inv10 = -H[1][0] / det;
                double inv11 = H[0][0] / det;

                return {-(inv00 * g[0] + inv01 * g[1]), -(inv10 * g[0] + inv11 * g[1])};
            }
        public:
            // @TODO 작성 필요
    };
} // namespace Optimization
```

---
## 3. Technical Review
1. **음의 곡률 (Negative Curvature)에 대한 완벽한 방어** :  
일반적인 뉴턴 방법 (Newton's Method)은 헤시안 행렬 $H$가 양의 정부호 (Positive Definite)가 아닌 구간, 즉 안장점 (Saddle Point)이나 극댓값 근처에서 $g^{T}Hg \leq 0$이 발생하면 잘못된 오르막 방향으로 발산해 버립니다. 해당 코드의 `if (gHg <= 0)` 분기문은 **모델이 위로 볼록하거나 평탄할 때, 안전하게 신뢰 반경의 경계 $(\delta)$까지 경사 하강 방향 $(-g)$으로 미끄러져 내려가도록 강제**합니다. 이 로직 덕분에 비선형성이 강한 차량 동역학 환경에서도 제어기가 결코 멈추거나 발산하지 않습니다.

2. **O(1) Subproblem Solver의 파괴력** : 교재의 알고리즘은 매 스텝마다 최적화 문제 (Trust Region Subproblem) 문제를 풀기 위해 `Convex.jl`과 같은 외부 패키지를 호출합니다. 이는 동적 메모리 할당을 유발하여 실시간 (RTOS) 환경에서는 절대 사용할 수 없습니다. 이 코드는 **뉴턴 스텝이 안에 있으면 뉴턴으로 가고, 밖으로 튀어나가면 코시 점이나 반경 끝단을 잘라 쓴다**는 단순화된 Dogleg 휴리스틱을 `if-else` 분기 3개만으로 처리했습니다. 연산 시간이 단 몇 마이크로초 $(us)$로 완벽하게 고정 (Deterministic)됩니다.

3. **2D 고속 연산의 목적성 (State-space Targeting)** : NMPC 등에서 횡방향 차량 제어를 할 때, State Vector를 `(횡방향 오차, 헤딩 오차)` 형태의 2D 공간으로 단순화하여 푸는 경우가 많습니다. 크기가 큰 행렬 라이브러리를 쓰지 않고, 코드 내에서 `det`와 `inv`를 직접 하드코딩한 것은 캐시 미스 (Cache miss)를 줄이고 레지스터 단에서 연산을 끝내기 위함입니다.