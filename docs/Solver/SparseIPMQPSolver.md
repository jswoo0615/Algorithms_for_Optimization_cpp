## SparseIPMQPSolver (Sparse Interior-Point Method Quadratic Programming Solver)
#### 코드 목적
최적화 이론의 정수인 **원쌍대 내부점법 (Primal-Dual Interior Point Method)** 을 임베디드 환경에 맞게 경량화한 코드

---
### 1. 최적화 문제 정의 (Problem Definition)
우리가 풀고자 하는 QP (Quadratic Programming) 문제는 아래와 같습니다.

$$\min_{x} \frac{1}{2}x^{T}Px + q^{T}x$$

$$\text{subject to } Ax \leq b$$

코드에서는 이를 위해 **슬랙 변수 (Slack Variable)** $s \geq 0$를 도입하여 부등식 제약을 등식 제약으로 변환합니다.

$$Ax + s = b, \quad s \geq 0$$

---
### 2. KKT 시스템과 잔차 (Residual) 계산
내부점법은 최적 조건인 KKT (Karush-Kuhm-Tucker) 조건을 만족하는 점을 향해 나갑니다.

코드의 `solve` 루프 초입에서 계산하는 세 가지 잔차는 다음과 같습니다.

#### 1. 쌍대 잔차 (Dual Residual, $r_d$)
물리 법칙 (Hessian)과 제약 조건의 평형 상태를 측정합니다.

$$r_{d} = Px + q + A^{T}\lambda$$

#### 2. 원시 잔차 (Primal Residual, $r_{p}$)
제약 조건 $Ax \leq b$를 얼마나 위반했는지 측정합니다.

$$r_{p} = Ax - b + s$$

#### 3. 상보성 잔차 (Complementarity Residual, $r_{c}$)
제약 조건의 경계면에 얼마나 붙어있는지 측정합니다 ($\mu$는 중심화 파라미터)

$$r_c = S \Lambda e - \sigma \mu e, \quad \text{where } S = \text{diag}(s), \Lambda = \text{diag}(\lambda)$$

---
### 3. 축소된 KKT 시스템 (Schur Complement)
가장 핵심적인 부분입니다. 원래는 $(x, s, \lambda)$에 대한 거대한 시스템을 풀어야 하지만, 연산량을 줄이기 위해 $\Delta x$에 대해서만 정리합니다. 이것이 코드의 `apply_H_sys` 함수가 수행하는 수식입니다.

$$(P + A^{T}WA)\Delta x = RHS$$
$$\text{where }W = \Lambda S^{-1} \quad \text{Weight Matrix}$$

* **Matrix-Free Operator** : 코드의 `apply_H_sys`는 $(P + A^T W A)$라는 거대한 행렬을 직접 만들지 않습니다. 대신 $A\Delta x$를 먼저 구하고, 거기에 $W$를 곱한 뒤, 다시 $A^{T}$를 곱하는 **연산 순서**만으로 결과를 도출합니다.

* **복잡도** : 이 기법을 통해 $O(N^{3})$의 조밀 행렬 (Dense Matrix) 연산을 **$O(NNZ)$** 의 희소 행렬 연산으로 격하시켰습니다. (NNZ : Nnumber of Non-Zero elements)

---
### 4. Newton Step과 업데이트 (The Upgrade Rule)
`solve_implicit_cg`를 통해 $\Delta x$를 구하면, 나머지 변수들을 복원합니다.

1. **슬랙 업데이트** : $\Delta s = -r_{p} - A\Delta x$
2. **쌍대 변수 업데이트** : $\Delta \lambda = -S^{-1}(r_{c} + \Lambda \Delta s)$
3. **보폭 결정 (Fraction-to-the-boundary)** : 변수들이 항상 양수 $(s, \lambda > 0)$를 유지하도록 보폭 $\alpha$를 조절합니다.

$$x \leftarrow x + \alpha_{p} \Delta x$$

$$s \leftarrow s + \alpha_{p} \Delta s, \quad \lambda \leftarrow \lambda + \alpha_{d} \Delta \lambda$$

이 수식 체계는 아래와 같은 특성을 가집니다.
* **구조적 단순화** : 복잡한 행렬 분해 대신 벡터 곱셈의 반복 (CG)으로 문제를 정의하여 어디서 멈춰야 하는지 (WCET) 명확히 제어합니다.
* **강건성 증명** : Soft Constraints의 철학을 수식적으로 뒷받침하여, Solver가 실패하더라도 시스템이 발산하지 않습니다.