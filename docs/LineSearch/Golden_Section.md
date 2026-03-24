# Algorithm 3.3 : 황금 분할 탐색 (Golden Section Search)
## 1. 수학적 원리 (Mathematical Formulation)
황금 분할 탐색은 제한된 함수 평가 횟수 내에서 탐색 구간을 최대로 축소하는 **피보나치 탐색 (Fibonacci Search)의 극한 (Limit) 개념을 활용한 근사 방법**

탐색 횟수 $n$이 무한대로 커질 때, 피보나치 수열의 연속된 두 항의 비율은 **황금비 (Golden Ratio, $\phi \approx 1.61803$)** 에 수렴합니다.
$$lim_{n \rightarrow \inf}\frac{F_{n}}{F_{n-1}} = \phi$$

이 비율을 이용하여 전체 구간 $[a, b]$ 내부에 두 개의 평가 지점 $c$와 $d$를 대칭적으로 배치합니다. 교재에 따르면 이 분할 비율 $\rho$는 $\phi - 1 \approx 0.618$ 이며, 양 끝에서부터 $1 - \rho \approx 0.382$ 만큼 떨어진 곳을 탐색 지점으로 잡습니다.

### 알고리즘 흐름 (Algorithm Flow)
1. **내부 지점 설정 (Algorithm Flow)** : 
* 구간 $[a, b]$에 대해 좌측에 가까운 점 $c$와 우측에 가까운 점 $d$를 황금비로 분할하여 잡고 함수값 $f(c)$, $f(d)$를 평가합니다
2. **구간 축소 (Interval Shrinking)** : 
* 만약 $f(c) < f(d)$라면, 최솟값은 무조건 $d$의 좌측에 존재하므로 새로운 구간을 $[a, d]$로 축소합니다.
3. **평가 지점 재활용 (O(1) Evaluation)** : 
* 황금비의 마밥에 의해, 축소된 새로운 구간 내에서 기존의 $c$ 또는 $d$ 중 하나가 **새로운 내부 평가 지점과 정확히 일치**하게 됩니다.
* 따라서 매 루프마다 함수 평가는 **단 1번만 추가로 수행**하면 됩니다

---
## 2. C++ 설계 (Code Implementation)
1차원 거리를 다차운 공간과 $x$와 방향 $d$로 매핑해주는 `ray_point`를 활용하여 NPMC의 선 탐색에 즉시 쓸 수 있도록 설계되었습니다. 코드의 변수 `phi`는 교재의 $1 - \rho$ 비율인 $\frac{3-\sqrt{5}}{2} \approx 0.38196$을 직접 계산하여 효율을 높였습니다

```C++
template <size_t N, typename Func>
static double golden_section_search(Func f, const std::array<double, N>& x,
                                    cnost std::array<double, N>& d, double a, double b, double tol = 1e-5, bool verbose = false) {
    // 1. 황금 분할 비율 설정 (3 - sqrt(5)) / 2 = 0.38196
    const double phi = (3.0 - std::sqrt(5.0)) / 2.0;

    double d_step = b - a;

    // 2. 초기 내부 지점 c (좌측), d_val (우측) 설정 및 함수 평가
    double c = a + phi * d_step;
    double d_val = b - phi * d_step;

    double yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));
    double yd = AutoDiff::value<N>(f, ray_point<N>(x, d, d_val));

    size_t iter = 0;

    // 3. 종료 조건 : 구간의 길이 (Gap)가 허용 오차 (tol) 이하가 될 때까지
    while (std::abs(d_step) > tol) {
        iter++;
        if (verbose) { /* ... 로그 출력 ... */ }

        // 4. 골짜기가 좌측에 치우져 있는 경우 (구간을 [a, d_Val]로 축소)
        if (yc < yd) {
            b = d_val;      // 우측 끝점 (b)을 d_val로 당겨옴
            d_val = c;      // 기존의 c는 새로운 구간의 d_val 역할을 수행
            yd = yc;        // 함수값 재활용 (평가 생략)
            d_step = b - a;
            c = a + phi * d_step;   // 새로운 c 지점만 추가 계산
            yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));
        }

        // 5. 골짜기가 우측에 치우져 있는 경우 (구간을 [c, b]로 축소)
        else {
            a = c;          // 좌측 끝점 (a)을 c로 당겨옴
            c = d_val;      // 기존의 d_val은 새로운 구간의 c 역할을 수행
            yc = yd;        // 함수값 재활용 (평가 생략)
            d_step = b - a;
            d_val = b - phi * d_step;   // 새로운 d_val 지점만 추가 계산
            yd = AutoDiff::value<N>(f, ray_point<N>(x, d, d_val));
        }
    }
    if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
    
    // 6. 최종 수렴된 구간의 중앙값을 최적 보폭(alpha)으로 반환
    return (a + b) / 2.0;
}
```

---
## 3. Technical Overview
1. **완벽한 실시간성 (Deterministic Execution Time)** : 교재에 명시되어 있듯 매 반복마다 구간은 약 $0.618$배로 일정하게 줄어들기 때문에, 허용 오차 $\epsilon$이 주어지면 **루프의 최대 반복 횟수가 실행 전 컴파일 타임에 완벽히 결정**됩니다. 반복횟수 $\approx \frac{ln(b-a) - ln(tol)}{ln(1.618)}$.
2. **연산 비용의 극단적 최소화 (Resuing Evaluations)** : 제어기에서 Cost Function `f(x)`를 한 번 평가하는 것은 시뮬렐이션 모델을 구동하거나 복잡한 행렬을 연산하는 무거운 작업입니다. 황금 분할 알고리즘은 구간을 버릴 때, 남은 구간에 위치한 이전 평가점 (`c` 또는 `d_val`)을 **다음 차에 그대로 재활용**합니다. 즉, 매 iteration 마다 무거운 `AutoDiff::value` 함수는 단 1회만 호출됩니다.
3. **한계점 (Limitation)** : 이 코드는 교재 4장의 NMPC 방향 탐색 $(d)$에 완벽히 부합하지만, 단봉형 (Unimodal)이라는 전제 조건이 필수적입니다. 만약 지형이 울퉁불퉁하여 여러 개의 함정이 있다면 (Non-Unimodal) 잘못된 계곡에 빠질 수 있으므로, 반드시 Bracket Minimum이 보장한 안전한 닫힌 구간 내에서만 호출되어야 합니다.