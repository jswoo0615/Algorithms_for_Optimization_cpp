# Algorithm 4.2 (Variant) : 강한 울프 조건 선탐색 (Strong Backtracking Line Search)
## 1. 수학적 원리 (Mathematical Formulation)
단순한 아르미호 (Armijo) 조건만 사용하면, 보폭 $\alpha$가 너무 작아져서 (Permature convergence) 최적화 속도가 느려질 수 있습니다. 이를 방지하고 진정한 최솟값 근처에 도달했음을 보장하기 위해 **두 가지 조건 (Strong Wolfe Conditions)**을 모두 만족하는 보폭을 찾습니다.
### 1. 충분 감소 조건 (Sufficient Decrease, 제 1 Wolfe 조건)
$$f(x + \alpha d) \leq f(x) + c_{1} \alpha \nabla_{d} f(x)$$
* 해석 : 보폭 $\alpha$만큼 이동했을 때, 목적 함수가 최소한 요구되는 비율 ($c_1$, 교재의 $\beta$)만큼은 감소해야 합니다 (너무 멀리 가는 것 방지)

### 2. 강한 곡률 조건 (Strong Curvature, 제 2 Wolfe 조건)
$$|\nabla_{d} f(x + \alpha d)| \leq c_{2}|\nabla_{d}f(x)|$$
* 해석 : 새로운 지점에서의 기울기 크기가 시작점 기울기 크기의 $c_{2}$ (교재의 $\sigma$) 비율 이하로 평탄해져야 합니다 (가파른 내리막 중간에 멈추는 것을 방지)

### 3. 알고리즘 흐름 (Interval Management)
교재의 알고리즘은 Bracket 단계와 Zoom 단계로 나눠지지만, 본 C++ 구현은 이를 **단일 상태 머신 (Single State Machine)** 으로 통합하여 하한선 (`alpha_lo`)과 상한선 (`alpha_hi`)을 갱신합니다.
* **후퇴 (Shrink)** : 함수값이 충분히 감소하지 않았거나, 오르막으로 꺾였다면 (방향 도함수 > 0), 상한선을 당겨오고 보폭을 절반으로 줄입니다 (Bisection)
* **전진 (Expand)** : 아직 내리막이 너무 가파르다면 (곡률 조건 실패), 하한선을 끌어올립니다. 만약 상한선을 아직 못 찾았다면 보폭을 2배 늘리고, 찾았다면 절반 위치로 전진합니다.

---
## 2. C++ 설계 (Code Implementation)
코드는 다차원 공간 $x$에서의 자동 미분 (`AutoDiff`)를 활용하여, **단일 `while(true)` 루프 안에서 구간 확장 (Expansion)과 축소 (Bisection)를 처리하는 고성능 C++ 솔버**입니다.
```C++
template <size_t N, typename Func>
static double search(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double alpha_init = 1.0, double c1 = 1e-4, double c2 = 0.9, bool verbose = false) {
    // 1. 출발점의 함수값 및 방향 도함수 (∇f(x)^T * d) 획득
    double f_x;
    std::array<double, N> grad_x;
    AutoDiff::value_and_gradient<N>(f, x, f_x, grad_x);
    
    double dir_deriv = 0.0;
    for (size_t i = 0; i < N; ++i) {
        dir_deriv += grad_x[i] * d[i];
    }

    // [안전장치] 하강 방향이 아닌 경우 경고
    if (dir_deriv >= 0.0 && verbose) 
        std::cout << "  [Warning] Not a descent direction!\n";
    
    // 2. 탐색 구간 (Bracket) 초기화
    double alpha_lo = 0.0;
    double alpha_hi = 1e9;  // 초기 상한선
    double alpha = alpha_init;

    size_t iter = 0;
    while (true) {
        iter++;
        auto x_new = ray_point<N>(x, d, alpha);

        // 3. 새로운 지점의 함수값 및 방향 도함수 획득
        double f_new;
        std::array<double, N> grad_new;
        AutoDiff::value_and_gradient<N>(f, x_new, f_new, grad_new);

        double dir_deriv_new = 0.0;
        for (size_t i = 0; i < N; ++i) {
            dir_deriv_new += grad_new[i] * d[i];
        }

        if (verbose) { /* ... 로그 출력 ... */ }

        // [조건 1] Armijo 검사 (충분 감소 실패) 또는 골짜기를 지나쳐 오르막 (dir_deriv_new > 0)
        if (f_new > f_x + c1 * alpha * dir_deriv || dir_deriv_new > 0.0) {
            alpha_hi = alpha;   // 상한선을 현재 위치로 당김 (Bounded 됨)
            alpha = (alpha_lo + alpha_hi) / 2.0;    // 구간의 절반으로 후퇴 (Bisection)
        }
        // [조건 2] Curvature 검사 (기울기가 여전히 가파름 = 더 갈 수 있음)
        else if (std::abs(dir_deriv_new) > c2 * std::abs(dir_deriv)) {
            alpha_lo = alpha;       // 하한선을 현재 위치로 올림
            if (alpha_hi >= 1e8) {
                alpha *= 2.0;       // 상한선이 없다면 보폭을 2배로 확장 (Expansion)
            } else {
                alpha = (alpha_lo + alpha_hi) / 2.0;        // 상한선이 있다면 그 사이로 전진 (Bisection)
            }
        }

        // [조건 3] 두 조건 모두 통과
        else {
            if (verbose) 
                std::cout << "  ↳ [Accepted] Strong Wolfe conditions satisfied!\n";
            return alpha;
        }

        // 무한 루프 방지 (정밀도 한계 도달 시 강제 종료)
        if (std::abs(alpha_hi - alpha_lo) < 1e-10) {
            if (verbose) 
                std::cout << "  ↳ [Failsafe] Bracket too small, returning best alpha.\n";
            return alpha;
        }
    }
    
}
```

---
## 3. Technical Review
1. **BFGS (Quasi-Newton) 제어기의 생명줄** :  
실차 NMPC에서 가장 많이 쓰이는 최적화 기법은 **BFGS 알고리즘**입니다. BFGS는 1차 미분 (Gradient) 기록을 바탕으로 2차 미분 (Hessian) 행렬을 근사하는데, 이때 행렬이 '양의 정부호 (Positive Definite)'를 유지하려면 $\delta^{T} \gamma > 0$ (Secant Equation Condition) 조건을 반드시 만족해야 합니다. **오직 이 '강한 울프 조건 (Strong Wolfe Conditions)' 만이 BFGS의 양의 정부호성을 수학적으로 보장**합니다. 단순 백트래킹을 BFGS에 물리면 얼마 안 가 제어기가 발산하게 됩니다.

2. **Bracket과 Zoom의 단일 루프 (Single-loop) 최적화** :   
구간을 찾는 1단계 (Bracket)와 그 구간 안에서 이분법으로 좁혀 들어가는 2단계 (Zoom)를 별도의 `while` 루프로 구현합니다. 하지만 `alpha_hi = 1e9`라는 초깃값을 이용해, **상한선이 무한대일 때는 2배씩 확장 (Expansion)하고, 상한선이 닫히면 절반씩 축소 (Bisection)하는 방식**으로 두 단계를 하나의 루프로 압축했습니다. 이는 코드 사이즈를 줄이고 분기 예측 (Branch prediction) 성능을 높여 임베디드 (ECU) 환경에 훨씬 적합합니다.

3. **확실한 Fail-safe (`alpha_hi - alpha_lo < 1e-10`)** :   
NMPC 솔버 구동 중 타이어 마찰 모델의 비선형성이나 센서 노이즈가 유입되면, 수학적인 울프 조건 교집합 영역이 사라질 수 있습니다. 이때 `alpha_hi`와 `alpha_lo`가 무한대에 가까워지며 갇혀버리는 문제가 발생할 수 있습니다. 이를 방지하기 위해 이중 안전장치 (`1e-10`)를 둔 것은 실차 교정에서 시스템 다운을 막아주는 역할을 합니다.