# Algorithm 3.7 : 이분법 (Bisection Method)
## 1. 수학적 원리 (Mathematical Formulation)
이분법은 목적 함수 $f(x)$ 자체의 값이 아닌, **도함수 $f'(x)$의 근 (Root), 즉 $f'(x) = 0$ 이 되는 지점을 찾는 알고리즘**입니다.

이 알고리즘은 **중간값 정리 (Intermediate Value Theorem)**를 기반으로 합니다. 만약 어떤 연속 함수 $f'(x)$가 구간 $[a, b]$에서 $f'(a)$와 $f'(b)$의 부호가 서로 다르다면 $(f'(a)f'(b) \leq 0)$, 구간 내부에 $f'(x) = 0$을 만족하는 점이 최소 하나 이상 반드시 존재합니다.

### 알고리즘 흐름 (Algorithm Flow)
1. **중간점 평가** : 현재 구간 $[a, b]$를 정확히 반으로 가르는 중간점 $x_{mid} = \frac{a + b}{2}$를 구하고, 해당 지점의 도함수 $f'(x_{mid})$를 평가합니다.
2. **정확한 근 발견** : 만약 $f'(x_{mid}) = 0$이라면 탐색을 즉시 종료합니다.
3. **구간 축소**
* $f'(x_{mid})$의 부호가 시작점 $f'(a)$의 부호가 같다면, 근은 중간점과 끝점 사이에 있으므로 구간을 $[x_{mid}, b]$로 축소합니다 $(a \leftarrow x_{mid})$
* 반대라면, 근은 시작점과 중간점 사이에 있으므로 구간을 $[a, x_{mid}]$로 축소합니다 $(b \leftarrow x_{mid})$
4. **수렴성 (Convergence)** : 구간의 길이 $|b - a|$가 허용 오차 $\epsilon$ 이하가 될 때까지 반복합니다. 이 방정식은 매 반복마다 구간이 정확히 절반으로 줄어드므로, 최대 $log_{2}(\frac{|b - a|}{2})$ 번 반복 내에 수렴함이 수학적으로 보장됩니다.

---
## 2. C++ 설계 (Code Implementation)
다차원 공간에서 임의의 방향 $d$로 향하는 **방향 도함수 (Directional Derivative)**를 평가하여, 다변수 NMPC의 선탐색 (Line Search)에 즉각 활용될 수 있는 고성능 구조입니다.
```c++
template <size_t N, typename Func>
static double bisection(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double a, double b, double tol = 1e-5, bool verbose = false) {
    // 1. 방향 일관성 보장 (항상 a < b가 되도록)
    if (a > b)
        std::swap(a, b);

    // 2. 방향 도함수 평가 람다 (x 지점에서 d 방향으로 alpha만큼 이동했을 때의 기울기)
    auto eval_deriv = [&](double alpha) {
        return directional_derivative<N>(f, x, d, alpha);
    };
    auto ya = eval_deriv(a);
    auto yb = eval_deriv(b);

    // [안전장치] 양 끝점이 이미 기울기 0 (최적점)에 도달한 경우 즉시 반환
    if (std::swap(ya) <= 1e-9)
        return a;
    if (std::swap(yb) <= 1e-9)
        return b;

    size_t iter = 0;

    // 3. 종료 조건 : 구간의 너비가 허용 오차 (tol) 이하가 될 때까지
    while ((b - a) > tol) {
        iter++;
        
        // 4. 중간점 (mid) 도함수 평가
        double mid = (a + b) / 2.0;
        double y_mid = eval_deriv(mid);

        if (verbose) { /* ... 로그 출력 ... */ }

        // [안전장치] 중간점의 기울기가 0에 극도로 가까우면 즉시 반환
        if (std::abs(y_mid) <= 1e-9) {
            if (verbose) 
                std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
            return mid;
        }

        // 5. 도함수의 부호가 같은 쪽의 구간을 축소
        // y_mid와 ya의 부호가 같다면 바닥은 mid의 우측에 있음 -> a를 mid로 당겨옴
        if ((y_mid > 0.0 && ya > 0.0) || (y_mid < 0.0 && ya < 0.0)) {
            a = mid;
            ya = y_mid;
        } else {
            // 부호가 다르다면 바닥은 mid의 좌측에 있음 -> b를 mid로 당겨옴
            b = mid;
            yb = y_mid;
        }
    }

    if (verbose) 
        std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
    
    // 6. 허용 오차 이내로 좁혀진 구간의 중앙값을 최종 보폭(alpha)으로 반환
    return (a + b) / 2.0;
}
```

---
## 3. Technical Review
1. **완벽한 시간 결정성 (Deterministic Execution Time)** :  
이분법의 가장 큰 무기는 **최악 실행 시간 (Worst-case Execution Time)을 완벽하게 예측할 수 있다**는 점입니다. 루프의 조건이 오직 `(b - a) > tol`이고 매번 구간이 정확히 1/2로 줄어드므로, $10^{-5}$의 오차를 요구하더라도 최대 약 17 ~ 20번 내외의 루프만 돌고 확정적으로 종료됩니다. 하드 리얼타임 (RTOS) 환경에 가장 적합한 특성입니다.

2. **연산 부하 트에이드오프 (Function Evaluation vs Derivative Evaluation)** :  
이분법은 $f(x)$가 아닌 도함수 $f'(x)$를 평가해야 합니다. `directional_derivative`가 만약 내부적으로 유한 차분법 (Finite Difference)을 사용한다면, 매 루프마다 함수를 2번씩 (`f(x+h)`, `f(x-h)`) 평가해야 하므로 0차 방법 (Zero-order methods)인 황금 분할 탐색보다 1 Iteration 당 연산량이 두 배 무겁습니다. 따라서 AutoDiff (자동 미분)를 통한 해석적 미분 결과가 $O(1)$로 제공될 때 진정한 위력을 발휘합니다.

3. **수치적 안정성을 위한 탁월한 방어 로직 (`1e-9`)** :  
코드에 삽입된 `std::abs(y_mid) <= 1e-9` 조건은 교재의 순수 알고리즘 (`y == 0`)을 실무적으로 완벽하게 보완한 형태입니다. 부동소수점 환경에 정확히 `0.0`이 나오는 경우는 드물기 때문에, 극소값 한계 (Tolerance)를 두어 불필요한 루프 낭비를 막고 안전하게 탈출하도록 한 점은 실차 제어 관점에서 매우 잘 된 설계입니다.