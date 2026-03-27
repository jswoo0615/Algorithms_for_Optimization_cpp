# Layer 0: AutoDiff (Automatic Differentiation Engine)

NMPC(비선형 모델 예측 제어)를 구현하기 위해서는 시스템 동역학의 선형화(Linearization)와 목적 함수의 이차 근사(Quadratic Approximation)가 필수적입니다. `AutoDiff.hpp`는 이를 위해 정밀한 미분 값을 제공하는 고수준 인터페이스를 정의합니다.

## 1. Gradient (구배) 계산: $f: \mathbb{R}^n \to \mathbb{R}$

스칼라 함수 $f$에 대해 모든 입력 변수 방향의 편미분 계수를 모은 벡터를 계산합니다.

### 수학적 정의

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}^\top$$

### 코드 구현 분석

`DualVec`을 사용하여 $n$개의 변수에 대해 시드(Seed)를 설정(Forward-mode AD)하고, 단 한 번의 함수 실행으로 전체 Gradient를 획득합니다.

```cpp
template <size_t N, typename Func>
static std::array<double, N> gradient(Func f, const std::array<double, N>& x_point) {
    std::array<DualVec<double, N>, N> x_dual;
    for (size_t i = 0; i < N; ++i) {
        // i번째 성분에 대해 미분 시드(1.0) 주입 (Seeding)
        x_dual[i] = DualVec<double, N>::make_variable(x_point[i], i);
    }
    DualVec<double, N> res = f(x_dual); // Forward Pass
    return res.g; // 계산된 Gradient 벡터 반환
}

```

## 2. Jacobian (자코비안) 행렬: $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$

차량 동역학 모델(Layer 7)의 상태 천이 행렬($A$)과 입력 행렬($B$)을 구하기 위해 필수적인 연산입니다.

### 수학적 정의

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \nabla f_1(\mathbf{x})^\top \\ \vdots \\ \nabla f_m(\mathbf{x})^\top \end{bmatrix}$$


즉, $J_{ij} = \frac{\partial f_i}{\partial x_j}$ 입니다.

### 코드 구현 분석

결과값이 벡터인 함수 `f`에 대해 각 요소의 Gradient를 행렬의 행(Row)으로 쌓아 올립니다.

```cpp
template <size_t M, size_t N, typename Func>
static std::array<std::array<double, N>, M> jacobian(Func f, const std::array<double, N>& x_point) {
    // ... 시딩 과정 동일 ...
    std::array<DualVec<double, N>, M> res_vec = f(x_dual);
    std::array<std::array<double, N>, M> J;
    for (size_t i = 0; i < M; ++i) {
        J[i] = res_vec[i].g; // i번째 함수의 Gradient가 Jacobian의 i행이 됨
    }
    return J;
}

```

## 3. Hessian (헤시안) 행렬: $\nabla^2 f(\mathbf{x})$

목적 함수의 곡률(Curvature) 정보를 제공하여 SQP나 Interior-Point 방식에서 수렴 속도를 가속화합니다.

### 수식: 하이브리드 접근법 (AD + FD)

본 코드는 1차 미분은 정확한 **AD(Automatic Differentiation)**로 계산하고, 2차 미분은 **중앙 차분법(Central Difference)**을 사용하는 하이브리드 방식을 채택합니다.

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{g_j(\mathbf{x} + \epsilon \mathbf{e}_i) - g_j(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}$$


(여기서 $g_j$는 $f$의 $j$번째 Gradient 성분)

### 코드 구현 분석 및 수치 안정화

수치 오차로 인해 발생할 수 있는 비대칭성을 해결하기 위해 **대칭화(Symmetrization)** 과정을 거칩니다.

```cpp
// 수치 미분을 이용한 2차 도함수 근사
for (size_t j = 0; j < N; ++j) {
    H[i][j] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
}

// 대칭화: H = 0.5 * (H + H^T)
double sym = (H[i][j] + H[j][i]) / 2.0;
H[i][j] = H[j][i] = sym;

```

---

## Technical Review

1. **Efficiency of `value_and_gradient`** : NMPC 솔버의 매 Iteration마다 목적 함수 값과 기울기를 동시에 요구합니다. 코드는 단 한 번의 Forward Pass(`res = f(x_dual)`)로 이를 해결하여 계산 자원을 최적화했습니다. 이는 **Layer 4(Nonlinear Solver)**의 실시간성을 결정짓는 요소입니다.
2. **Hybrid Hessian의 타당성** : Full-AD로 헤시안을 구현할 경우 메모리 복잡도가 $O(N^2)$으로 급증합니다. 본 코드처럼 Gradient(AD) + FD(Central Difference) 방식을 사용하면, 메모리 사용량을 억제하면서도 순수 수치 미분보다 훨씬 높은 정밀도를 확보할 수 있습니다. 이는 **정적 메모리 설계** 철학과 일치합니다.
3. **Encapsulation** : `AutoDiff` 클래스는 복잡한 `DualVec` 연산을 캡슐화하여, 상위 레이어의 개발자가 물리 법칙(Layer 7) 구현에만 집중할 수 있도록 돕는 **'입법자'의 도구**입니다.
4. **Numerical Stability** : 헤시안 계산 시 사용된 `eps = 1e-5`는 부동 소수점 오차와 절단 오차 사이의 균형을 고려한 수치입니다. 하지만 차량의 급격한 거동 시 발생하는 불연속점에서는 이 간격에 대한 엄밀한 검토가 필요합니다.