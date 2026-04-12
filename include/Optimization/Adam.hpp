#ifndef OPTIMIZATION_ADAM_HPP_
#define OPTIMIZATION_ADAM_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief Adam (Adaptive Moment Estimation) 최적화 알고리즘을 구현한 정적 클래스
 *
 * Adam은 모멘텀(Momentum) 최적화 방식과 RMSProp 알고리즘의 장점을 결합하여 만든 가장 대중적이고
 * 강력한 최적화 기법 중 하나입니다. 각 파라미터(변수)마다 1차 모멘트(Gradient의 지수 이동 평균)와
 * 2차 모멘트(Gradient 제곱의 지수 이동 평균)를 모두 계산하여, 학습 방향과 스텝 크기(학습률)를
 * 동시에 동적으로 조절합니다.
 *
 * 핵심 아이디어:
 * 1. **1차 모멘트 (m)**: 과거 기울기의 방향(관성)을 유지하여, 진동을 줄이고 빠르게 수렴하도록
 * 돕습니다. (Momentum의 역할)
 * 2. **2차 모멘트 (v)**: 과거 기울기 크기의 제곱을 추적하여, 자주 변하는 파라미터는 세밀하게(작은
 * 스텝), 거의 변하지 않는 파라미터는 큼직하게(큰 스텝) 이동시킵니다. (RMSProp의 역할)
 * 3. **편향 보정 (Bias Correction)**: 초기 반복 시 m과 v가 0으로 편향(Bias)되는 것을 수학적으로
 * 보정하여 초기부터 올바른 방향을 잡도록 돕습니다.
 *
 * @note 실시간(RT) 환경과 최적화 성능을 위해 값비싼 `std::pow()` 함수 호출을 배제하고, 반복 시마다
 * `beta`를 곱해나가는 누적 곱(Running Product) 기법을 사용했습니다.
 */
class Adam {
   public:
    // 상태를 저장하지 않는 정적 유틸리티 클래스로만 사용되도록 인스턴스화를 방지합니다.
    Adam() = delete;

    /**
     * @brief Adam 메인 최적화 함수
     *
     * @tparam N 최적화 변수의 차원
     * @tparam Func 목적 함수 타입 (람다식, 펑터 등)
     * @param f 최소화할 목적 함수
     * @param x 최적화 시작 지점 (초기 추정값)
     * @param alpha 학습률 (Learning Rate, 기본값: 0.1)
     * @param beta1 1차 모멘트(기울기)에 대한 지수 감쇠율 (Exponential Decay Rate, 기본값: 0.9)
     * @param beta2 2차 모멘트(기울기 제곱)에 대한 지수 감쇠율 (보통 1차 모멘트보다 0.999 등으로 1에
     * 더 가깝게 설정합니다)
     * @param epsilon 0으로 나누어지는 것을 방지하기 위한 수치적 안정성 상수 (기본값: 1e-8)
     * @param max_iter 무한 루프를 방지하기 위한 최대 허용 반복 횟수
     * @param tol 수렴 판정을 위한 허용 오차 (Gradient의 Norm 제곱이 이 값보다 작아지면 수렴으로
     * 판정)
     * @param verbose 콘솔에 최적화 진행 상황을 출력할지 여부
     * @return std::array<double, N> 최적화가 완료된 해(Optimal Point)
     */
    template <size_t N, typename Func>
    [[nodiscard]] static std::array<double, N> optimize(Func f, std::array<double, N> x,
                                                        double alpha = 0.1, double beta1 = 0.9,
                                                        double beta2 = 0.999, double epsilon = 1e-8,
                                                        size_t max_iter = 15000, double tol = 1e-4,
                                                        bool verbose = false) {
        // 0차원 배열 주입을 컴파일 타임에 원천 차단
        static_assert(N > 0, "Dimension N must be greater than 0");

        std::array<double, N> m = {0.0};  // 1차 모멘트 (기울기의 지수 이동 평균) 초기화
        std::array<double, N> v = {0.0};  // 2차 모멘트 (기울기 제곱의 지수 이동 평균) 초기화

        // [성능 최적화] 매 이터레이션(루프)마다 g_norm 계산 시 std::sqrt()를 호출하는 것을 막기
        // 위해 허용 오차(tol)를 미리 제곱해 둡니다.
        const double tol_sq = tol * tol;

        // 편향 보정에 사용될 beta1^t 와 beta2^t 의 누적값
        // std::pow(beta1, iter) 를 매번 호출하는 대신, 매 반복마다 변수에 beta1, beta2를
        // 곱해줍니다.
        double beta1_t = beta1;
        double beta2_t = beta2;

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 👑 Adam Optimizer Started (alpha=" << alpha << ")\n";
            std::cout << "========================================================\n";
        }

        // 최적화 반복 루프
        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x = 0.0;
            std::array<double, N> g = {0.0};

            // 1. Auto Diff 호출 (O(1) 정적 메모리 환경에서 기울기 계산)
            AutoDiff::value_and_gradient(f, x, f_x, g);

            // 2. 기울기 제곱합(Norm^2) 계산 및 수렴 판정
            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g[i] * g[i];
            }

            if (g_norm_sq < tol_sq) {
                if (verbose) {
                    std::cout << "✅ Convergence achieved at iteration " << iter
                              << " (f(x) = " << f_x << ", ||g||^2 = " << g_norm_sq << ")\n";
                }
                break;
            }

            // [성능 최적화] 차원(N) 루프 진입 전, 모든 차원에 공통으로 적용되는
            // 편향 보정(Bias Correction) 분모를 단 한 번만 미리 계산합니다.
            const double one_minus_beta1_t = 1.0 - beta1_t;
            const double one_minus_beta2_t = 1.0 - beta2_t;

            // 3. 파라미터 업데이트 (차원별 순회)
            for (size_t i = 0; i < N; ++i) {
                // Step 3-1: 1차 모멘텀과 2차 모멘텀 업데이트
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];

                // Step 3-2: 편향 보정 (Bias Correction)
                // 초기화 시점(t=0)에 m과 v가 0으로 초기화되어 있으므로, 초기 몇 번의 반복 동안은
                // 0에 가깝게 편향됩니다. 이를 방지하기 위해 (1 - beta^t) 로 나누어 스케일을
                // 원복시킵니다.
                const double m_hat = m[i] / one_minus_beta1_t;
                const double v_hat = v[i] / one_minus_beta2_t;

                // Step 3-3: 파라미터 업데이트 (In-place 연산으로 캐시 히트율 증가)
                // x_t = x_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)
                x[i] -= (alpha / (std::sqrt(v_hat) + epsilon)) * m_hat;
            }

            // 4. 다음 이터레이션을 위해 누적 곱 업데이트 (O(1) 연산)
            // 비싼 std::pow 연산을 피하는 핵심 메커니즘입니다.
            beta1_t *= beta1;
            beta2_t *= beta2;

            // 진행 상태 로깅 (분기 예측 최적화를 위해 1000번에 1번만 출력)
            if (verbose && (iter % 1000 == 0)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << std::sqrt(g_norm_sq)
                          << "\n";
            }
        }

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            // C++17 constexpr if를 활용하여 차원 수에 맞춰 컴파일 타임에 분기
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        // 최종적으로 업데이트된 최적해 배열 반환
        return x;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_ADAM_HPP_