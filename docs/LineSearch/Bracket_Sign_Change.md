# Algorithm 3.6 : 부호 변경 구간 탐색 (Bracket Sign Change)
## 1. 수학적 원리 (Mathematical Formulation)

목적 함수 $f(x)$의 최솟값 (Local Minima)이 존재하는 지점에서는 기울기가 0이 됩니다. $(f'(x) = 0)$. 따라서 최적점을 찾는 문제는 **도함수** $f'(x)$ **의 근 (Root)을 찾는 문제**로 치환할 수 있습니다.

중간값 정리 (Intermediate Value Theorem)에 따르면, 연속 함수 $f'(x)$에 대해 어떤 구간 $[a, b]$의 양 끝점 기울기 부호가 다르다면, 즉 $f'(a)f'(b) \leq 0$ 이라면, 그 구간 내에는 반드시 $f'(x) = 0$이 되는 지점이 최소 하나 이상 존재합니다.

### 알고리즘 흐름 (Algorithm Flow)
1. **초기화** : 임의의 초기 구간 $[a, b]$를 잡고, 중심점 (center)과 반폭 (half_width)을 계산합니다.
2. **부호 검사** : 양 끝점에서의 방향 도함수 값 $f'(a)$와 $f'(b)$를 곱하여 양수 $(> 0)$인지 확인합니다. 양수라면 두 지점의 기울기 방향이 같으므로 (둘 다 오르막이거나 둘 다 나리막), 사이에 바닥이 존재한다는 보장이 없습니다.
3. **구간 확장 (Expansion)** : 부호가 같다면 중심점을 고정한 상태로 구간의 너비를 $k$배 (기본 2배)로 늘립니다.
4. **종료 조건** : $f'(a) f'(b) \leq 0$이 되는 순간, 즉 **탐색 구간 양 끝의 기울기 부호가 엇갈리는 순간** 확장을 멈추고 구간을 반환합니다.

---
## 2. C++ 설계 (C++ Implementation)
단순히 1차원 스칼라 변수에 대한 구현을 넘어, $N$차원 공간에서 임의의 탐색 방향 $d$를 따라가는 **방향 도함수 (Directional Derivative) 평가 람다 (`eval_deriv`)** 를 통해 다변수 NMPC 환경에 완벽하게 맞춤 설계되었습니다.

```c++
template <size_t N, typename Func>
static std::pair<double, double> bracket_sign_change(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double a, double b, double k = 2.0, bool verbose = false) {
    // 1. 방향 일관성 보장 : a가 항상 b보다 작도록 정렬
    if (a > b) {
        std::swap(a, b);
    }

    // 2. 중심점 및 반폭 초기화
    double center = (a + b) / 2.0;
    double half_width = (b - a) / 2.0;

    // 3. 방향 도함수 평가 람다 : 다차원 공간 x에서 방향 d로 alpha만큼 이동한 지점의 기울기 반환
    auto eval_deriv = [&](double alpha) {
        return directional_derivative<N>(f, x, d, alpha);
    };

    double fp_a = eval_deriv(a);
    double fp_b = eval_deriv(b);

    size_t iter = 0;

    // 4. 종료 조건 : 두 기울기의 부호가 다르면 (음수 또는 0) 루프 탈출
    while (fp_a * fp_b > 0.0) {
        iter++;

        // 5. 구간 확장 : 중심을 유지한 채 반폭을 k배 늘려 새로운 a, b 갱신
        half_width *= k;
        a = center - half_width;
        b = center + half_width;

        // 6. 새로운 경계에서의 도함수 재평가
        fp_a = eval_deriv(a);
        fp_b = eval_deriv(b);

        if (verbose) { /* ... 로그 출력 ... */ }
    }

    return std::make_pair(a, b);
}
```

---
## 3. Technical Overview
수학적으로 훌륭하게 구성되었으나, 실차 ECU (Real-time 제어기)에 탑재하기 위해서는 **두 가지 치명적인 예외 상황 (Corner cases)** 에 대한 방어 로직이 반드시 추가되어야 합니다.

1. **무한 루프 ** :  
가장 심각한 문제는 `while (fp_a * fp_b > 0.0)` 루프에 **반복 횟수 제한 (`max_iter`)이 없다**는 점입니다. 목적 함수가 $f(x) = exp(x)$처럼 기울기의 부호가 영원히 바뀌지 않는 단조 증가/감소 함수이거나, 타이어 슬립 모델의 노이즈로 인해 도함수 계산이 꼬일 경우, 이 코드는 무한 루프에 빠져 제어기를 멈추게 (Watchdog Reset) 만듭니다. **반드시 `iter < MAX_ITER` 조건을 추가해야 합니다**
2. **근 누락 (Missing Roots) 현상** :  
이 방식의 치명적인 단점은 **가까이 붙어있는 두 개의 근 (Roots)을 동시에 뛰어넘어 버릴 수 있다**는 것입니다.
* 구간을 2배로 넓히는 과정에서, 새로 넓힌 구간 안에 최솟값과 최댓값이 모두 포함되어 버리면, 양 끝점의 기울기 부호는 여전히 같게 나옵니다.  
* 이 경우 알고리즘은 "아직 바닥을 찾지 못했다"라고 착각하고 구간을 영원히 확장하며 (`infinitely increase without termination`) 실제 정답이 있는 계곡을 완전히 무시해버리게 됩니다.

### [실차 적용을 위한 해결책]
NMPC 솔버에서 이 함수를 안전하게 사용하려면, 위에서 언급한 `max_iter`를 추가하고, 만약 `max_iter` 도달 시까지 부호가 바뀌지 않는다면
* 이전의 안전했던 상태로 제어 입력을 유지 (Fallback)하거나
* 가장 기울기 크기가 작았던 (가장 바닥에 가까웠던) 지점을 강제로 반환

위 둘을 예외 처리 아키텍처가 결합되어야 합니다.