# Algorith 5.5 : AdaGrad
## 1. 수학적 원리 (Mathematical Formulation)
경사 하강법이나 모멘텀 기법은 모든 상태 변수 (State Variables)에 대해 동일한 학습률 $\alpha$를 곱하여 위치를 업데이트합니다. 하지만 차량 제어나 로보틱스에서 변수들은 각기 다른 물리량과 스케일 (예 : $x_{1}$은 m 단위의 위치, $x_{2}$는 rad 단위의 각도)을 가질 수 있습니다.  
**AdaGrad (Adaptive Subgradient)** 는 각 매개변수 (차원)별로 과거의 기울기 (Gradient) 제곱합을 누적하고, 이 누적값이 큰 (자주 변한) 변수는 보폭을 줄이고, 누적값이 작은 (거의 변하지 않은) 변수는 보폭을 키워주는 **맞춤형 학습률 (Adaptive Learning Rate)** 을 제공합니다.

### AdaGrad 업데이트 공식
각 차원 $i$에 대해 과거 기울기의 제곱합 $s_{i}$를 누적합니다.
$$s_i^{(k)} = \sum_{j=1}^{k} \left( g_i^{(j)} \right)^2$$

새로운 위치는 누적된 제곱합의 제곱근에 반비례하도록 업데이트합니다.
$$x_{i}^{(k + 1)} = x_{i}^{(k)} - \frac{\alpha}{\epsilon + \sqrt{s_{i}^{(k)}}}g_{i}^{(k)}$$

* $s_{i}$ : $i$번째 차원의 누적된 기울기 제곱합 (코드의 `G[i]`)
* $\epsilon$ : 분모가 0이 되어 발산하는 것을 막기 위한 매우 작은 안전 상수 (보통 $1 \times 10^{-8}$)
* $\alpha$ : 기본 학습률 (보통 $0.01 ~ 0.1$ 사용)

---
## 2. C++ 설계 (Code Implementation)
$O(1)$의 힙 할당 없는 스택 메모리 (`std::array`) 운용과, In-place 연산 (`-=`, `*=`)을 통해 캐시 히트율을 극대화한 실차 제어기용 초고속 솔버입니다.
```C++
#ifndef OPTIMIZATION_ADAGRAD_HPP_
#define OPTIMIZATION_ADAGRAD_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace Optimization {
    public:
        AdaGrad() = delete;

        template <size_t N, typename Func>
        [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.1, double epsilon = 1e-8, size_t max_iter = 15000, double tol = 1e-4, bool verbose = false) {
            // [핵심 1] 0차원 배열 주입을 컴파일 타임에 원천 차단 (MISRA 준수)
            static_assert(N > 0, "Dimension N must be greater than 0");

            // [핵심 2] 동적 할당 없는 스택 메모리 (교재의 s 벡터 역할)
            std::array<double, N> G = {0.0};

            if (verbose) { /* ... 로그 출력 ... */ }

            for (size_t iter = 1; iter <= max_iter; ++iter) {
                double f_x = 0.0;
                std::array<double, N> g = {0.0};

                // 1. O(1) 할당 초고속 Auto Diff 계산
                AutoDiff::value_and_gradient<N>(f, x, f_x, g);

                // 2. 기울기 l2-Norm 계산 및 조기 종료 검증
                double g_norm_sq = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    g_norm_sq += g[i] * g[i];
                }

                const double g_norm = std::sqrt(g_norm_sq);
                if (g_norm < tol) {
                    break;
                }

                // 3. AdaGrad 파라미터 업데이트 
                for (size_t i = 0; i < N: ++i) {
                    G[i] += g[i] * g[i];    // 과거 기울기 제곱 누적
                    // [수치적 안정성] epsilon을 루트 밖에서 더하여 발산 방지
                    // [실시간성] -= In-place 연산자로 캐시 히트율 극대화
                    x[i] -= (alpha / (std::sqrt(G[i]) + epsilon)) * g[i];
                }

                if (verbose && (iter % 1000 == 0)) { /* ... 로그 출력 ... */ }
            }

            if (verbose) {
                std::cout << " 🏁 Final Optimal Point: [" << x;
                // [OEM 핵심 3] C++17 constexpr if를 활용한 분기 없는 다차원 출력
                if constexpr (N > 1) std::cout << ", " << x[5];
                if constexpr (N > 2) std::cout << ", ...";
                std::cout << "]\n";
            }
            return x;
        }
} // namespace Optimization
#endif // OPTIMIZATION_ADAGRAD_HPP_
```

---
## 3. Technical Review
1. **스케일 불균형 (Ill-conditioned) 지형에서의 자율주행** :   
차량 동역학 제어에서 조향각 (rad)과 차량의 속도 (m/s)를 동시에 최적화해야 할 때, 두 변수는 변화하는 스케일이 완전히 다릅니다. 일반 경사 하강법에서는 한 변수는 발산하고 한 변수는 수렴하지 않는 문제가 생기지만, AdaGrad는 **많이 움직인 조향각은 보폭을 줄이고, 덜 움직인 속도는 보폭을 키워주는** 방식으로 각 변수의 단위를 스스로 보정합니다. 

2. **C++17 `constexpr if`와 `static_assert`** :   
`static_assert(N > 0)`과 `if constexpr` 조합은 런타임에 오버헤드나 에러가 발생할 여지를 컴파일 타임에 100% 잘라내며, $N$의 크기에 따라 불필요한 어셈블리 명령어가 아예 생성되지 않도록 만듭니다.

3. **치명적인 약점 : 학습률의 영구적 감쇠** :   
수학적 원리를 보면 $G_{i}$ (교재의 $s_{i}$)는 제곱수들의 합이므로 무조건 단조 증가 (Strictly nondecreasing)만 할 수 있습니다. 이는 실시간 NMPC 제어 루프에서 치명적입니다. 제어기가 계속 커져서 Iteration이 진행되면 어느 순간 분모가 무한히 커져 사실상 **학습률이 0이 되어 제어기가 멈춰버리는 현상 (Premature Freezing)** 이 발생합니다.
이 때문에 AdaGrad를 실차의 메인 연속 제어기로 사용하는 것은 위험합니다. 과거의 기울기 제곱을 무한히 누적하는 대신, 최신 기울기의 제곱에 더 큰 가중치를 주는 **지수 이동 평균 (Exponential Moving Average)** 을 사용하여 학습률이 0으로 되는 것 대신 **RMSProp**이나 **Adadelta**로 가야만 실시간 연속 제어 환경에서 사용할 수 있습니다.