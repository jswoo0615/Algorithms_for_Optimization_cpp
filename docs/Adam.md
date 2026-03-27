# Algorithm 5.8 : Adam (Adaptive Momentum Estimation)
## 1. 수학적 원리 (Mathematical Formulation)
Adam은 모멘텀 (Momentum)의 '관성'과 RMSProp의 '변수별 맞춤형 보폭'을 완벽하게 융합한 알고리즘입니다.
각 파라미터 (차원)마다 개별적인 학습률을 적용하기 위해 **두 가지의 지수 이동 평균 (Exponential Moving Average)** 을 동시헤 추적합니다.

1. **제 1모멘트 (편향된 모멘텀, 1st Momentum)** : 과거의 기울기 (방향)을 누적하여 평탄한 지형을 빠르게 돌파합니다.
$$m^{(k+1)} = \beta_{1}m^{(k)} + (1 - \beta_{1})g^{(k)}$$

2. **제 2모멘트 (편형된 기울기 제곱합, 2nd Moment)** : 과거 기울기의 제곱을 누적하되, 최신 정보에 가중치를 두어 스케일이 큰 변수의 보폭을 억제
$$v^{(k+1)} = \beta_{2}v^{(k)} + (1 - \beta_{2}) (g^{(k)} \cdot g^{(k)})$$

### 초기 편향 보정 (Bias Correction)
초기 상태 $(m^{(0)} = 0, v^{(0)} = 0)$에서 시작하면 처음 몇 번의 스텝 동안 은 값들이 0에 가깝게 평향되는 문제가 발생합니다. Adam은 매 스텝 $k$마다 이를 수학적으로 교정하여 초기부터 안정적인 스텝을 밟도록 합니다.

$$\hat{m}^{(k+1)} = \frac{m^{(k+1)}}{1 - \beta_{1}^{k}}, \quad \hat{v}^{(k+1)} = \frac{v^{(k+1)}}{1 - \beta_{2}^{k}}$$

### Adam 업데이트 공식
$$x^{(k+1)} = x^{(k)} - \alpha \frac{\hat{m}^{(k+1)}}{\epsilon + \sqrt{\hat{v}^{(k+1)}}}$$

---
## 2. C++ 설계 (Code Implementation)
실시간 제어기의 한정된 컴퓨팅 자원을 줄이기 위해 수학 함수 (`std::pow`, `std::sqrt`) 의 호출을 줄였습니다.

```C++
#ifndef OPTIMIZATION_ADAM_HPP_
#define OPTIMIZATION_ADAM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
    class Adam {
        public:
            Adam() = delete;    // 인스턴스화 방지
            template <size_t N, typename Func>
            [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x, double alpha = 0.1, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, size_t max_iter = 15000, double tol = 1e-4, bool verbose = false) {
                static_assert(N > 0, "Dimension N must be greater than 0");
                std::array<double, N> m = {0.0};    // 1차 모멘트 초기화
                std::array<double, N> v = {0.0};    // 2차 모멘트 초기화

                // [성능 최적화 1] 매 이터레이션마다 sqrt를 호출하지 않기 위해 허용 오차 제곱
                const double tol_sq = tol * tol;

                // [성능 최적화 2] std::pow를 대체하기 위한 누적 곱 (Running product) 변수
                double beta1_t = beta1;
                double beta2_t = beta2;

                if (verbose) { /* ... 로그 출력 ... */

                for (size_t iter = 1; iter <= max_iter; ++iter) {
                    double f_x = 0.0;
                    std::array<double, N> g = {0.0};
                    AutoDiff::value_and_gradient(f, x, f_x, g);

                    // 기울기 L2-Norm의 제곱 계산 및 조기 종료 검사
                    double g_norm_sq = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        g_norm_sq += g[i] * g[i];
                    }

                    if (g_norm_sq < tol_sq) {
                        if (verbose) 
                            std::cout << "✅ Convergence achieved...\n";
                        break;
                    }

                    // [성능 최적화 3] 차원 (N) 루프 진입 전, 공통 분모 (Bias correction)를 미리 계산
                    // 루프 불변 코드 이동 (Loop Invariant Code Motion)
                    const double one_minus_beta1_t = 1.0 - beta1_t;
                    const double one_minus_beta2_t = 1.0 - beta2_t;

                    for (size_t i = 0; i < N; ++i) {
                        // 1. 1차 / 2차 모멘텀 업데이트
                        m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];

                        // 2. 편향 보정 (std::pow 2번 호출을 O(1) 나눗셈으로 압축)
                        const double m_hat = m[i] / one_minus_beta1_t;
                        const double v_hat = v[i] / one_minus_beta2_t;

                        // 3. 파라미터 업데이트 (In-place 연산)
                        x[i] -= (alpha / (std::sqrt(v_hat) + epsilon)) * m_hat;
                    }

                    // [성능 최적화 2] 다음 이터레이션을 위한 누적 곱 업데이트 (O(1) 연산)
                    beta1_t *= beta1;
                    beta2_t *= beta2;

                    if (verbose && (iter % 1000 == 0)) { /* ... 로그 출력 ... */ }
                }
                return x;
            }
    };
} // namespace Optimization
#endif // OPTIMIZATION_ADAM_HPP_
```

---
## 3. Technical Review
1. **`std::pow`의 제거와 Running Product** : 교재에서는 편향 보정 수식에 `v ./ (1 - \gamma_v^k)` 처럼 명시적으로 $k$제곱을 사용하고 있습니다. 하지만 차량용 제어기 (MCU / ECU) 환경에서 매 루프, 매 차원마다 부동소숫점 승수 연산인 `std::pow`를 호출하는 것은 클럭 낭비 (수백-수천 사이클)를 발생시킵니다. `beta1_t *= beta1` 형태로 **매 스텝 상수번의 곱셈 $(O(1))$만 수행해서 $\beta^{(k)}$를 추적하는 누적 곱 (Running Product) 방식**은 실행 속도를 수십배 이상 끌어올리는 최적화 기법입니다.

2. **루프 불변 코드 이동 (Loop Invariant Code Motion)** : `one_minus_beta1_t`와 `one_minus_beta2_t`를 $N$차원 순회하는 `for (size_t i = 0; i < N; ++i)` 루프 바깥으로 빼서 한 번만 계산했습니다. 이 값은 차원 수에 상관없이 $k$번째 스텝에서 모두 동일하기 때문에 반복문 내부에 둘 이유는 없습니다. 컴파일러가 알아서 최적화할 수 있지만, 엔지니어가 명시적으로 구조화하여 Cache와 Register 효율을 극대화했습니다.

3. **다변수 이종 (Heterogeneous) 데이터 제어** : 자율주행 차량에서 `x = [스티어링 각도 (rad), 모터 토크 (Nm), 브레이크 압력 (bar)]`와 같이 물리적 단위와 스케일이 전혀 다른 수백 개의 상태 변수를 동시에 최적화하더라도, Adam의 2차 모멘트 $(\hat(v))$가 각 변수의 스케일을 자동으로 정규하해 주어 안전하고 빠른 수렴을 이끌어낼 것입니다.