# Algorithm 5.2 : 공액 기울기법 (Conjugate Gradient, Fletcher-Reeves)
## 1. 수학적 원리 (Mathematical Formulation)
일반적인 경사 하강법은 매 스텝마다 함수가 가장 가파르게 감소하는 방향 $(-g)$으로 이동한 뒤 멈춥니다. 하지만 정밀 선탐색을 사용하면 다음 스텝의 이동 방향은 항상 이전 스텝의 방향과 직교 (Orthogonal)하게 되어, 좁은 계곡 지형에서 수많은 지그재그 행보를 보이며 수렴이 극도로 느려집니다.

**공액 기울기법 (Conjugate Gradient Method)** 은 이전의 탐색 방향 (관성)을 지우지 않고 새로운 기울기와 결합하여, 목적 함수의 헤시안 행렬 $A$에 대해 서로 **공액 (Mutually conjugate)** 이 되는 방향으로 이동합니다. (즉, $d^{(i)T}Ad^{(j)} = 0$) 이차 함수 (Quadratic function)의 경우, 이 방법은 정확히 차원 수 $N$번의 스텝 만에 전역 최솟값에 수렴함이 수학적으로 증명되어 있습니다.

### 알고리즘 흐름 (Algorithm Flow)
1. **초기화** : 첫 번째 탐색 방향은 순수 경사 하강법과 동일하게 역방향 기울기로 설정합니다.
$$d^{(1)} = -g^{(1)}$$

2. **정밀 선탐색 (Exact Line Search)** : 해당 방향으로 함수를 최소화하는 최적의 보폭 $\alpha$를 찾고 위치를 업데이트합니다.
$$x^{(k+1)} = x^{(k)} + \alpha^{(k)} d^{(k)}$$

3. **가중치 $\beta$ 계산 (Fletcher-Reeves 공식)** : 이전 탐색 방향을 얼마나 유지할지 결정하는 비율 $\beta$를 두 기울기의 크기 (Norm) 비율로 계산합니다.
$$\beta^{(k)} = \frac{(g^{(k+1)})^{T}g^{(k+1)}}{(g^{(k)})^{T}g^{(k)}}$$

4. **새로운 공액 방향 설정** : 새로운 역방향 기울기에 이전 탐색 방향을 $\beta$만큼 더하여 다음 이동 방향을 결정합니다.
$$d^{(k+1)} = -g^{(k+1)} + \beta^{(k)} d^{(k)}$$

---
## 2. C++ 설계 (Code Implementation)
공액 기울기 이론을 바탕으로, **이분법 (Bisection) 기반의 정밀 선탐색 모듈을 통합**하여 실무 NMPC (비선형 모델 예측 제어) 환경에서 즉시 사용할 수 있도록 구현된 고성능 엔진입니다.

```C++
namespace Optimization {
    class ConjugateGradient {
        public:
            template <size_t N, typename Func>
            static std::array<double, N> optimize(Func f, std::array<double, N> x, size_t max_iter = 1000, double tol = 1e-4, bool verbose = false) {
                double f_x;
                std::array<double, N> g;
                AutoDiff::value_and_gradient<N>(f, x, f_x, g);

                // 1. 첫 번째 방향 : 순수 경사 하강 방향 (d = -g)
                std::array<double, N> d;
                for (size_t i = 0; i < N; ++i) {
                    d[i] = -g[i];
                }

                for (size_t iter = 1; iter <= max_iter; ++iter) {
                    // [종료 조건] 현재 기울기의 크기 (Norm)가 허용 오차 이하면 수렴
                    double g_norm_sq = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        g_norm_sq += g[i] * g[i];
                    }
                    if (std::sqrt(g_norm_sq) < tol) {
                        break;
                    }

                    // 2. [핵심 통합] Exact Line Search를 통한 자율 보폭 (alpha) 탐색
                    // Bracket과 Bisection을 재사용하여 최적 alpha 도출
                    auto bracket = LineSearch::bracket_sign_change<N>(f, x, d, 0.0, 0.001, 2.0, false);
                    double alpha = LineSearch::bisection<N>(f, x, d, bracket.first, bracket.second, 1e-4, false);

                    // 3. 상태 업데이트 (x_new = x + alpha * d)
                    for (size_t i = 0; i < N; ++i) {
                        x[i] += alpha * d[i];
                    }

                    // 4. 새로운 위치에서 기울기 계산 (g_norm)
                    std::array<double, N> g_new;
                    AutoDiff::value_and_gradient<N>(f, x, f_x, g_new);

                    // 5. Fletcher-Reeves 공식을 이용한 beta 비율 계산
                    double g_new_norm_sq = 0.0;
                    for (size_t i = 0; i < N; ++i) {
                        g_new_norm_sq += g_new[i] * g_new[i];
                    }
                    double beta = g_new_norm_sq / g_norm_sq;

                    // 6. 새로운 공액 방향 (Conjugate Direction) 설정 : d = -g_new + beta * d_old
                    for (size_t i = 0; i < N; ++i) {
                        d[i] = -g_new[i] + beta * d[i];
                    }

                    // 다음 루프를 위해 기울기 상태 교체
                    g = g_new;
                }
                return x;
            }
    };
} // namespace Optimization
```

---
## 3. Technical Overview
1. **하이퍼파리미터 튜닝의 완전한 제거 (Self-Tuning Engine)** : 딥러닝이나 일반적인 최적화 라이브러리에서 경사 하강법을 쓸 때 가장 큰 문제는 캘리브레이션 엔지니어가 학습률 (Learning Rate, $\alpha$)을 수동을 맞춰주어야 한다는 것입니다. `bracket_sign_change`와 `bisection`을 호출하여 **해당 방향에서 가장 완벽한 학습률 $(\alpha)$을 알고리즘이 스스로 계산** 합니다. 게다가 $\beta$ 역시 Fletcher-Reeves 공식에 의해 자동 산출되므로, 사용자는 목적 함수 $f(x)$만 던져주면 솔버가 알아서 답을 찾아옵니다. 실차 제어 로직에서 튜닝 공수를 극적으로 줄여주는 역할을 합니다. 

2. **$O(N) 메모리 복잡도로 뉴턴 방법 효과$** : 6장에서 다룰 뉴턴 방법이나 BFGS는 $O(N^{2})$ 크기의 헤시안 (또는 그 근사 행렬)을 메모리에 저장하고 연산해야 합니다. 하지만 공액 기울기법은 오직 $x, g, d$와 같은 길이 $N$의 1차원 `std::array` 몇 개만으로 2차 함수 (Quadratic) 지형을 돌파합니다. 메모리가 극히 제한된 마이크로컨트롤러 (MCU)에서 NMPC를 돌려야 한다면 가장 우선적으로 고려해야 할 최적화 엔진입니다.

3. **Polak-Ribiere로의 확장 가능성 (코드 개선 제안)** : 교재에 따르면 순수 이차 함수가 아닌 일반적인 비선형 함수에서는 Fletcher-Reeves 방식보다 **Polak-ribiere 방식**이 더 빠르고 강건하게 수렴합니다. 현재 코드의 `beta = g_new_norm_sq / g_norm_sq` 부분을 다음과 같이 한 줄만 수정하면, 알고리즘 5.2 (Polak-ribiere)로 즉시 업그레이드 될 수 있습니다.
```c++
// Polak-ribiere (식 5.17) 및 자동 리셋 (식 5.18) 적용
double pr_num = 0.0;
for (size_t i = 0; i < N; ++i) {
    pr_num += g_new[i] * (g_new[i] - g[i]);
}
double beta = std::max(0.0, pr_num / g_norm_sq);
```

자동 리셋 (`std::max(0.0, ...)`)은 제어 로직이 복잡하게 꼬여 탐색  방향을 잃었을 때, 스스로 $\beta$를 0으로 만들어 순수 경사 하강법으로 재시작하게 해주는 안전장치가 됩니다.