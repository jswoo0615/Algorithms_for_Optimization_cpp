# Algorithm 3.4 : 이차 적합 탐색 (Quadratic Fit Search)
## 1. 수학적 원리 (Mathematical Formulation)
대부분의 부드러운 (Smooth) 목적 함수는 국소 최솟값 (Local Minimum) 부근으로 충분히 확대해 보면 2차 함수 (Quadratic Function) 형태에 근사합니다.

이차 적합 탐색은 브래킷으로 묶인 세 점 $a < b < c$와 그 함수값 $y_{a}, y_{b}, y_{c}$를 완벽하게 지나는 가상의 2차 함수 $q(x) = p_{1} + p_{2}x + p_{3}x^2$를 적합 (Fit) 시킵니다. 

행렬식으로 표현하면 아래와 같습니다.

$$\begin{bmatrix}
y_{a} \\
y_{b} \\
y_{c}
\end{bmatrix} = \begin{bmatrix} 1 & a & a^{2} \\ 1 & b & b^{2} \\ 1 & c & c^{2} \end{bmatrix} \begin{bmatrix} p_{1} \\ p_{2} \\ p_{3} \end{bmatrix}$$

이 가상의 2차 함수가 최솟값을 갖는 지점 (즉, 미분값이 0이 되는 지점 $q'(x) = 0$)인 $x^{\star}$는 역행렬 연산을 통해 해석적으로 다음과 같이 단번에 도출됩니다.

$$x^{star} = \frac{y_{a}(b^2 - a^2) + y+{b}(c^2 - a^2) + y_{c}(a^2 - b^2)}{2(y_{a}(b - c) + y_{b}(c - a) + y_{c}(a - b))}$$

### 알고리즘 흐름 (Algorithm Flow)
1. 세 점 $a, b, c$로 2차 함수를 적합하여 최솟값 $x^{star}$를 계산합니다.
2. 새로운 점 $x^{star}$의 함수값 $y^{star}$를 평가합니다.
3. 기존의 세 점 중 하나를 버리고, $x^{star}$를 포함하여 **항상 중간점 $(b)$이 가장 낮은 함수값을 갖도록** 새로운 $a < b < c$ 브래킷을 재구성합니다.
4. $x^{star}$와 $b$의 거리가 허용 오차 (`tol`) 이내가 될 때까지 반복합니다.

---
## 2. C++ 설계 (Code Implementation)
$$x^{star} = \frac{y_{a}(b^2 - a^2) + y+{b}(c^2 - a^2) + y_{c}(a^2 - b^2)}{2(y_{a}(b - c) + y_{b}(c - a) + y_{c}(a - b))}$$
이 수식을 그대로 쓰지 않고, 부동소수점 오차 (Floating-point error)를 방지하기 위해 중간점 $b$를 기준으로 식을 재정리한 **수치해석학적 최적화 수식**을 사용했습니다.

```c++
template <size_t N, typename Func>
static double quadratic_fit_search(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double a, double b, double c, size_t max_iter = 50, double tol = 1e-5, bool verbose = false) {
    // 다차원 공간의 방향 d에 대한 1차원 선탐색 함수 (eval) 생성
    auto eval = [&](double alpha) {
        return AutoDiff::value<N>(f, ray_point<N>(x, d, alpha));
    };
    double ya = eval(a), yb = eval(b), yc = eval(c);

    for (size_t iter = 1; iter <= max_iter; ++iter) {
        // 1. x* 계산
        // 분자와 분모를 b를 기준으로 한 오프셋으로 재정리하여 Catastrophic Cancellation 방지
        double num = (b - a) * (b - a) * (yb - yc) - (b - c) * (b - c) * (yb - ya);
        double den = (b - a) * (ybb - yc) - (b - c) * (yb - ya);

        // [안전 장치] 세 점이 일직선상에 있어 2차 함수 적합이 불가능할 경우 (분모가 0) 루프 탈출
        if (std::abs(den) < 1e-16) {
            break;
        }

        double x_star = b - 0.5 * num / den;
        double y_star = eval(x_star);

        if (verbose) { /* ... 로그 출력 ... */ }

        // 2. 종료 조건 : 새로운 최솟값 x*와 기존 최솟값 b의 위치 변화가 거의 없을 때
        if (std::abs(x_star - b) < tol) {
            if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
            return x_star;
        }

        // 3. 브래킷 업데이트 (a < b < c 유지 및 최소점 b 보장 로직)
        if (x_star > b) {   // x*가 우측에 생성됨
            if (y_star > yb) {
                c = x_star; yc = y_star;    // [a, b, x*] 로 구간 축소
            } else {
                a = b; ya = yb;             // [b, x*, c] 로 구간 이동
                b = x_star; yb = y_star;
            }
        } else {            // x*가 좌측에 생성
            if (y_star > yb) {
                a = x_star; ya = y_star;    // [x*, b, c]로 구간 축소
            } else {
                c = b; yc = yb;
                b = x_star; yb = y_star;
            }
        }
    }
    return b;
}
```

---
## 3. Technical Review
1. **빠른 수렴 속도** : 황금 분할 탐색 (Algorithm 3.3)은 항상 구간을 약 0.618 배로 줄이는 선형적 (Linear) 수렴 속도를 가집니다. 하지만 NMPC (비선형 모델 예측 제어)에서 최적점 부근에 도달했을 때 이 이차 적합 탐색을 사용하면, 수렴 속도가 기하급수적으로 빨라집니다. 실시간 제어에서 **적은 함수 평가 (Evaluation) 횟수로 정확한 보폭 $(\alpha)$을 찾는 데 압도적으로 유리**합니다.
2. **수치적 안정화 (Numerical Stability) 처리** `x_star = b - 0.5 * num / den` 로직은 교재 수식을 그대로 구현하지 않고 $b$에서 이동량 (Shift)으로 재정의한 것입니다. 이는 제어 변수 값 $(x)$이 커졌을 때 $a^{2}, b^{2}, c^{2}$ 연산에서 발생하는 부동소수점 정보 손실을 막아줍니다.
3. **Fail-safe 방어 로직 (`den < 1e-16`)** 교재에서도 **새로 생성된 점이 기존 점과 너무 가까울 때를 대비한 안정장치 (Safeguard)가 필요하다**고 언급합니다. 함수의 곡률이 없어 세 점이 직선에 가깝게 배치된 경우, 2차 함수의 분모가 0이 되어 $(p_{3} \approx 0)$ $x^{star}$가 발산해 제어기가 터지게 됩니다. 이럴 경우를 대비해 발산하기 직전까지 가장 좋은 값 `b`를 반환하도록 되어 있스빈다
4. **고정 메모리와 실행 시간 제어** `max_iter = 50` 기본값을 부여를 통해, 어떤 악조건의 목적 함수가 들어오더라도 최대 50번의 루프 후에는 탈출하도록 설계되었습니다. 이는 **최악 실행 시간 (Worst-Case Execution Time) 제한 조건**을 완벽하게 만족합니다.