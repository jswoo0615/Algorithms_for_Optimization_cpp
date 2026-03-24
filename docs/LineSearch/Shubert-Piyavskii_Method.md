# Algorithm 3.5 : 슈베르트 - 피야브스키 방법 (Shubert-Piyavskii Method)
## 1. 수학적 원리 (Mathematical Formulation)
이 알고리즘은 목적 함수가 단봉형일 필요는 없으나, **립시츠 연속 (Lipschitz continuous)** 이어야 한다는 강력한 전제 조건이 필요합니다. 즉, 이 함수의 기울기 (변화율)가 절대 넘을 수 없는 최대 한계치인 **립시츠 상수 $L$**을 미리 알고 있어야 합니다.

$$|f(x) - f(y)| \leq L|x - y| \quad \text{for all } x, y \in [a, b]$$

이 성질을 이용하면, 한 점 $x_{0}$에서 함수를 평가했을 때 $f(x_{0})$를 기준으로 기울기가 $\pm L$인 두 직선을 그어 **함수가 절대 이 직선들 밑으로 내려갈 수 없는 하한선 (Lower Bound)을 만들 수 있습니다

### 알고리즘 흐름 (Algorithm Flow)
1. **톱니바퀴 하한선 생성** : 평가된 두 점 $a, b$와 그 함수값 $y_{a}, y_{b}$를 지나는 기울기 $\pm L$의 직선들이 교차하는 V자 계곡의 밑바닥 $(x_{m}, y_{m})$을 계산합니다. 교차점의 수식은 다음과 같이 해석적으로 도출됩니다.
$$x_{m} = \frac{a + b}{2} + \frac{y_{a} - y_{b}}{2L}$$
$$y_{m} = \frac{y_{a} + y_{b}}{2} - L \frac{b - a}{2}$$

2. **최저 하한선 탐색** : 생성된 여러 V자 계곡들 중, 하한선 $y_{m}$이 가장 낮은 지점 $x_{m}$을 찾아 목적 함수 $f(x_{m})$을 실제로 평가합니다.
3. **종료 조건** : 방금 평가한 실제 함수값 $f(x_{m})$과 예측했던 하한선 $y_{m}$의 차이가 허용 오차 $\epsilon$ 이내라면, 그곳이 전역 최솟값임이 보장되므로 탐색을 종료합니다.
4. **분할** : 아직 오차가 크다면, 새로운 평가점 $x_{m}$을 기준으로 기존 구간을 두 개로 분할하고 1번으로 돌아가 톱니바퀴 하한선을 더 촘촘하게 업데이트합니다.

---
## 2. C++ 설계 (Code Implementation)
```C++
template <size_t N, size_t MAX_NODES = 100, typename Func>
static double shubert_piyavskii(Func f, const std::array<double, N>& x,
                                const std::array<double, N>& d, double a, double b, double L, double tol = 1e-4, bool verbose = false) {
    auto eval = [&](double alpha) {
        return AutoDiff::value<N>(f, ray_point<N>(x, d, alpha));
    };

    // 구간 및 하한선 교차점 정보를 담는 정적 구조체
    struct SPNode {
        double a, ya, b, yb, x_m, y_m;
        bool active;
    };

    // 동적 할당을 배제한 정적 메모리 풀 (Heap-Free)
    std::array<SPNode, MAX_NODES> pool;
    size_t pool_size = 0;

    double ya = eval(a), yb = eval(b);
    double min_val = std::min(ya, yb);
    double min_alpha = (ya < yb) ? a : b;

    // 1. 하한선 V자 계곡 교차점 계산 및 풀 (Pool) 등록 람다
    auto add_node = [&](double n_a, double n_ya, double n_b, double n_yb) {
        if (pool_size >= MAX_NODES)
            return; // 메모리 안전 장치

        // 수식적 도출에 따른 교차점 x_m, y_m 계산
        double x_m = 0.5 * (n_a + n_b) + (n_ya - n_yb) / (2.0 * L);
        double y_m = 0.5 * (n_ya + n_yb) - L * (n_b - n_a) / 2.0;
        pool[pool_size++] = {n_a, n_ya, n_b, n_yb, x_m, y_m, true};
    };

    // 최악 실행 시간 (WCET) 보장을 위한 하드 리미트 루프
    // 매 루프마다 노드가 1개씩 순증가하므로 MAX_NODES/2로 재한하여 오버플로우 원천 차단
    for (size_t iter = 1; iter <= MAX_NODES/2; ++iter) {
        double best_ym = 1e99;
        size_t best_idx = 0;
        bool found = false;

        // 2. 현재 가장 낮은 하한선 (y_m)을 가진 구간 탐색
        for (size_t i = 0; i < pool_size; ++i) {
            if (pool[i].active && pool[i].y_m < best_ym>) {
                best_ym = pool[i].y_m;
                best_idx = i;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        // 3. 실제 함수값  평가 및 전역 최고 기록 갱신
        double x_m = pool[best_idx].x_m;
        double y_eval = eval(x_m);
        if (y_eval < min_val) {
            min_val = y_veal;
            min_alpha = x_m;
        }

        if (verbose) { /* ... 로그 출력 ... */ }

        // 4. 종료 조건 : 함수값과 하한선의 오차가 tol 이하 (전역 최솟값 보장)
        if (std::abs(y_eval - pool[best_idx].y_m) < tol) {
            break;
        }

        // 5. 기존 구간 삭제 (active 비활성화) 및 2개의 하위 구간 분할 생성
        pool[best_idx].active = false;
        add_node(pool[best_idx].a, pool[best_idx].ya, x_m, y_eval);
        add_node(x_m, y_eval, pool[best_idx].b, pool[best_idx].yb);
    }

    if (verbose) 
        std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
    return min_alpha;
}
```

---
## 3. Technical Overview
1. **실시간 제어기를 위한 완벽한 메모리 샌드박싱** :   
`std::array<SPNode, MAX_NODES>`를 선언하고, `active` 플래그로 논리적인 삭제를 하며, `iter <= MAX_NODES / 2` 조건으로 **메모리 오버플로우를 수학적으로 원천 차단**했습니다.

2. **한계점 : 립시츠 상수 $L$ 추정의 어려움**  
전역 최적화라는 강력한 무기에도 불구하고, 실무 NMPC (Model Predictive Control) 루프 안에 이 코드를 메인으로 넣기엔 치명적인 약점이 하나 있습니다. 교재에서도 지적하듯 **차량 동역학이나 타이어 마찰 모델의 정확한 립시츠 상수 $L$을 알아내는 것은 불가능에 가깝습니다**
* $L$을 너무 크게 잡으면 : 하한선 V자 계곡이 너무나 뾰족해셔서 바닥을 찾는 데 무한히 오랜 시간이 걸립니다
* $L$을 너무 작게 잡으면 : 하한선이 함수를 뚫고 올라와 전역 최솟값을 놓치게 됩니다.

3. **실무에서의 활용 (Fallback Strategy)**   
그러므로 이 알고리즘은 매 5ms마다 도는 NMPC의 메인 Solver로는 무겁습니다. 대신, 장애물 회피 (Collision Avoidance) 경로 생성 등 **목적 함수에 여러 개의 가짜 바닥 (Local Minima)이 발생할 수밖에 없는 특수한 조건 (Multi-model condition)에서, SQP가 갇혔을 때 탈출 경로를 제공하는 Fallback (안전망) 엔진**으로 사용하는 것이 가장 이상적입니다.