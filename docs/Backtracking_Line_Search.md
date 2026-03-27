# Algorithm 4.1 : 백트래킹 선탐색 (Backtracking Line Search)
## 1. 수학적 원리 (Mathemiatical Formulation)
근사 선탐색 (Approximate Line Search)은 적절한 보폭 $\alpha$를 찾기 위해 **충분 감소 조건 (Sufficient Decrease Condition)**을 사용합니다. 이 조건은 '아르미호 (Armijo) 조건' 또는 '제 1 Wolfe 조건'으로도 불리며, 다음과 같이 정의됩니다

$$f(x + \alpha d) \leq f(x) + \beta \alpha \nabla_{d} f(x)$$

* $\nabla_{d} f(x)$ : 방향 도함수 (Directional Derivative). 경사도와 탐색 방향의 내적 $(g^{T} d)$과 같습니다.
* $\beta \in$ : 요구되는 최소한의 감소 비율. 일반적으로 매우 작은 값인 $10^{-4}$를 사용합니다. (코드에서는 변수 `c`에 해당)
* 해석 : 단순히 $f(x)$가 줄어드는 것을 넘어, 1차 미분 (기울기)이 예측하는 감소량의 최소 $\beta$ 비율만큼은 실제로 감소해야만 그 보폭을 인정하겠다는 뜻입니다.

### 알고리즘 흐름 (Algorithm Flow)
1. 방향 도함수 $\nabla_d f(x)$를 계산하여 내리막 방향인지 검증합니다.
2. 초기 보폭 $\alpha$ (보통 1.0)부터 시작하여 새로운 지점 $x_{new} = x + \alpha d$의 함수값을 평가합니다.
3. 새로운 함수값이 목표 감소치 (Target value)보다 작거나 같으면 (즉, 아르미호 조건을 만족하면) 현재의 $\alpha$를 즉시 반환합니다
4. 만족하지 못하면 보폭 $\alpha$를 감소 비율 $p$ (보통 0.5)만큼 줄여서 다시 검사합니다.

---
## 2. C++ 설계 (Code Implementation)
**SIMD (Single Instruction Multiple Data) 벡터화 연산과 Failsafe 로직이 결합된 고성능 실시간 제어기 (RT)용 엔진**으로 구현되었습니다.

```C++
template <size_t N, typename Func>
[[nodiscard]] static double search(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double f_x, const std::array<double, N>& grad_x, double alpha = 1.0, double p = 0.5, double c = 1e-4, bool verbose = false) noexcept {
    // 1. 방향 도함수 (Directional Derivative) 계산 : \Delta f(x)^T * d
    double dir_deriv = 0.0;
#pragma omp simd    // SIMD 벡터화 명령어로 O(N) 내적 연산 병렬 가속
    for (size_t i = 0; i < N; ++i) {
        dir_deriv = std::fma(grad_x[i], d[i], dir_deriv);   // FMA로 정밀도 및 속도 향상
    }

    // 2. 하상 방향 (Descent Direction) 안전 검사
    if (dir_deriv >= 0.0 && verbose) {
        std::cout << "  [Warning] Not a descent direction! dir_deriv: " << dir_deriv << "\n";
    }

    size_t iter = 0;
    while (true) {
        iter++;

        // 3. 새로운 평가 지점 (x_new) 생성 및 함수값 (f_new) 평가
        auto x_new = ray_point<N>(x, d, alpha);
        double f_new = AutoDiff::value<N>(f, x_new);

        // 4. Armijo 조건 (제1 Wolfe 조건) 목표 감소치 계산
        // Target: f(x) + c * alpha * (g^T d)
        double target_val = f_x + c * alpha * dir_deriv;

        if (verbose) { /* ... 로그 출력 ... */ }

        // 5. 충분 감소 조건 만족 시 즉시 탈출 (Approximate Line Search)
        if (f_new <= target_val) {
            if (verbose) 
                std::cout << "  ↳ [Accepted] Armijo condition satisfied!\n";
            return alpha;
        }

        // 6. 보폭 축소 (Backtracking)
        alpha *= p;

        // 7. [OEM 핵심] 무한 루프 방지용 Failsafe 안전장치
        if (alpha < 1e-10) {
            if (verbose) 
                std::cout << "  ↳ [Failsafe] Alpha reached minimum limit.\n";
            return alpha;
        }
    }
}
```

---
## 3. Technical Review
1. **`#pragma omp simd`와 `std::fma`를 통한 초고속 벡터화** 최적화 코드의 `ray_point`와 `dir_deriv` 계산 루프에 삽입된 `#pragma omp simd`와 `std::fma`는 탁월한 설계입니다. 차량의 ECU나 자율주행 컴퓨터에서는 N차원 상태 벡터 연산 시 CPU의 벡터 레지스터 (AVX/NEON)를 활용하여 연산을 병렬 처리하는 것이 지연 시간 (Latency) 단축에 필수적입니다. 이 한 줄로 연산 속도가 비약적으로 올라갑니다.

2. **무한 루프 탈출 안전장치 (`alpha < 1e-10`)** NMPC 솔버 구동 중 모델의 비선형성이나 센서 노이즈로 인해 목적 함수 지형이 순간적으로 일그러지면, 아무리 보폭을 줄여도 아르미호 조건을 만족하지 못하는 사태가 발생할 수 있습니다. `alpha < 1e-10` 브레이크 로직은 **제어기 다운 (Watchdog Reset)을 막고 시스템이 다음 스텝에서 회복 (Recovery)할 수 있게 해주는 가장 완벽한 방어막 (Fail-safe)** 입니다.

3. **정밀 탐색을 포기하고 얻는 '실시간성 (Real-time)'** 3장의 황금 분할 탐색이나 이차 적합 탐색은 한 번의 방향 $d$에 대해 깊이 파고들어 최적 보폭 $\alpha^{\star}$ 를 찾아냅니다. 매 스텝마다 완벽한 선탐색을 하는 것은 연산 낭비입니다. SQP (Sequential Quadratic Programming)나 뉴턴 기반의 NMPC에서는 **대충 충분히 줄어들었으면 (Armijo 만족), 빨리 그 지점에서 자코비안/헤시안을 다시 계산해서 새로운 하강 방향 $d_{next}$를 잡는 것**이 전체 수렴을 압도적으로 빠르게 만듭니다. 이 코드가 바로 그 역할을 완벽히 수행합니다.

4. **하강 방향 (Descent Direction) 검증 로그** `dir_deriv >= 0.0`에 대한 경고 출력 로직은 디버깅에 큰 도움을 줍니다. 헤시안 근사가 잘못되어 솔버가 "오르막"을 가리킬 때, 이 라인이 실차 테스트에서 문제의 원인을 즉각적으로 알려줄 것입니다.