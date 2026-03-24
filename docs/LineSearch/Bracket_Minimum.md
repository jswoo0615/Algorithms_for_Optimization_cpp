# Algorithm 3.1 : 초기 구간 한정 (Bracket Minimum)
## 1. 수학적 원리 (Mathematical Formulation)
선탐색 (Line Search)을 통해 최적의 보폭 $\alpha$를 착지 위해서는, 먼저 **최솟값이 반드시 존재할 수밖에 없는 닫힌 구간 (Bounded Interval)** 을 찾아야 합니다.

이 알고리즘은 탐색 방향의 지형이 하나의 골짜기만 가진 **단봉형 (Unimodal)** 함수라 가정합니다. 어떤 세 점 $a, b, c (a < b < c)$ 에 대해 다음 조건이 성립한다면, 구간 $[a, c]$ 내부에 반드시 지역 최솟값 (Local minimum)이 존재합니다.

$$f(a) > f(b) \quad \text{and} \quad f(b) < f(c)$$

즉, 함수가 감소하다가 $(a \rightarrow b)$ 다시 증가하는 $(b \rightarrow c)$ 지점 $c$를 찾는 것이 핵심입니다.

### 알고리즘 흐름 (Algorithm Flow)
1. **초기화 및 방향 설정** : 초기 위치 $a$에서 미세한 보폭 $s$만큼 이동한 $b = a + s$를 평가합니다. 만약 $f(b) > f(a)$라면 오르막길로 잘못 진입한 것이므로, $a$와 $b$를 맞바꾸고 탐색 방향을 반대로 뒤집습니다 $(s \leftarrow -s)$ 
2. **지수적 확장 (Exponential Expansion)** : 내리막길을 따라 점진적으로 보폭을 넓히며 $(s \leftarrow k \cdot s)$ 다음 점 $c = b + s$를 평가합니다. ($k$는 보통 2.0 사용)
3. **종료 조건 (Termination)** : 새로운 지점 $c$의 함수값 $f(c)$가 이전 지점 $b$의 함수값 $f(b)$보다 커지는 순간 $(f(c) > f(b))$, 골짜기의 바닥을 지나친 것이므로 탐색을 종료하고 구간 $[a, c]$를 반환합니다.

## 2. C++ 설계 (Code Implementation)
제공해주신 코드는 1차원 스칼라 변수 x에 대한 교재의 구현을 넘어, 다차원 상태 공간 $x$와 탐색 방향 벡터 $d$를 결합한 `ray_point`를 통해 NMPC 실무에 즉시 적용 가능한 형태로 확장되었습니다

```c++
// 1. 초기 지점 (a) 및 함수값 (ya) 평가
double a = 0.0;
double ya = AutoDiff::value<N>(f, ray_point<N>(x, d, a));

// 2. 다음 지점 (b) 및 함수값 (yb) 평가
double b = s;
double yb = AutoDiff::value<N>(f, ray_point<N>(x, d, b));

// 3. 방향 검증 : f(a) < f(b) 이면 오르막이므로 탐색 방향을 반대로 뒤집음
if (yb > ya) {
    std::swap(a, b);
    std::swap(ya, yb);
    s = -s;
}

size_t iter = 0;
while (true) {
    iter++;

    // 4. 새로운 지점 (c) 탐색
    double c = b + s;
    double yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));

    if (verbose) {
        std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] a: " << std::fixed
                    << std::setprecision(5) << a << " | b: " << b << " | c: " << c
                    << " | f(c): " << yc << "\n";
    }

    // 5. 종료 조건 : f(c) > f(b)이면 어떤 함수가 다시 오르막으로 꺾인 것 (구간 발견)
    if (yc > yb) {
        return (a < c) ? std::make_pair(a, c) : std::make_pair(c, a);
    }

    // 6. 상태 전진 및 보폭 확장 (가속)
    a = b;
    ya = yb;
    b = c;
    yb = yc;
    s *= k;
}
```

## 3. Technical Overview
**안전성 (Worst-case Safety) 이슈 분석**: 수학적으로 완벽한 이 코드에도 실차 ECU에 올리기 전 반드시 수정해야 할 치명적인 약점이 하나 있습니다.

교재 41페이지에 명시되어 있듯, 목적 함수가 $f(x)=exp(x)$ 처럼 최솟값이 존재하지 않고 무한히 감소하기만 하는 지형(No local minima)에 빠지면 이 알고리즘은 영원히 구간을 찾지 못합니다.

현재 C++ 코드의 `while (true)` 루프는 이 상황에서 제어기의 **Worst-case execution time**을 무한대로 발산시켜 Watchdog 리셋(시스템 다운)을 유발할 수 있습니다.

**[실차 적용을 위한 해결책]**   
반드시 `while (iter < MAX_ITER)` 형태로 하드 리미트를 걸어주어야 합니다. 만약 `MAX_ITER` 내에 브래킷을 찾지 못했다면, 예외를 던지거나 현재까지 확장된 넓은 구간을 강제로 반환하여 시스템의 생존(Robustness)을 보장하는 로직이 추가되어야 완벽한 합격선(OEM Level)의 코드가 됩니다.