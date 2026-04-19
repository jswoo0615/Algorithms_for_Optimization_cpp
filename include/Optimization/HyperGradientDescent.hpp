#ifndef OPTIMIZATION_HYPER_GRADIENT_DESCENT_HPP_
#define OPTIMIZATION_HYPER_GRADIENT_DESCENT_HPP_

/**
 * @file HyperGradientDescent.hpp
 * @brief Hyper Gradient Descent (하이퍼 경사하강법) 최적화 알고리즘 구현
 * 
 * Hyper Gradient Descent는 기본 Gradient Descent의 고정된 보폭(Learning Rate) 또는
 * 복잡한 Line Search 방법의 단점을 극복하기 위해, 보폭(alpha) 자체를 또 다른 경사하강법으로
 * 학습(업데이트)하는 메타-최적화(Meta-optimization) 기법입니다.
 * 
 * 현재 스텝의 기울기(Gradient)와 이전 스텝의 기울기의 내적(Dot product)을 이용하여
 * 보폭을 적응적으로 조절합니다. (Algorithms for Optimization, Sec 5.9 참고)
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @class HyperGradientDescent
 * @brief Hyper Gradient Descent Optimizer를 수행하는 정적 클래스
 * 
 * @note MISRA C++ 준수 지향. 동적 메모리 할당이 없는 O(1) 메모리 구조를 가지며,
 * 발산(Divergence)을 방지하기 위한 안전장치(Fail-safe) 및 Gradient Clipping이 적용되어 있습니다.
 */
class HyperGradientDescent {
   public:
    // 인스턴스화 방지
    HyperGradientDescent() = delete;

    /**
     * @brief Hyper Gradient Descent 알고리즘을 실행하여 함수의 최솟값을 찾습니다.
     * 
     * @tparam N 최적화할 변수의 차원 수
     * @tparam Func 목적 함수 타입
     * @param f 최적화할 목적 함수 (AutoDiff 호환)
     * @param x 초기 시작 위치
     * @param alpha_init 초기 학습률 (보폭, alpha). 기본값: 0.001
     * @param mu 학습률(alpha)을 업데이트하기 위한 메타-학습률(Meta-learning rate). 기본값: 1e-6
     * @param tol 수렴 판정을 위한 기울기 L2 Norm의 허용 오차. 기본값: 1e-5
     * @param max_iter 최대 반복 횟수. 기본값: 50000
     * @param verbose 진행 상황 콘솔 출력 여부. 기본값: false
     * @return std::array<double, N> 최적화된 결과 파라미터 배열 (최적해)
     */
    template <size_t N, typename Func>
    [[nodiscard]] static constexpr std::array<double, N> optimize(
        Func f, std::array<double, N> x, double alpha_init = 0.001, double mu = 1e-6,
        double tol = 1e-5, size_t max_iter = 50000, bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0.");

        double alpha = alpha_init;                     // 현재 학습률 (동적으로 업데이트됨)
        std::array<double, N> g_prev = {0.0};          // 이전 단계의 기울기를 저장하는 배열
        std::array<double, N> x_valid = x;             // 발산 시 복구하기 위한 안전 백업(Fail-safe) 상태

        const double tol_sq = tol * tol;               // 루트 연산을 피하기 위한 제곱 허용 오차

        // [안전성 보장] 수치적 발산을 막기 위한 학습률(alpha)의 범위 설정
        constexpr double MIN_ALPHA = 1e-8;             // 학습률이 너무 작아지는 것 방지
        constexpr double MAX_ALPHA = 0.005;            // Rosenbrock과 같은 급경사 함수에서 발산하는 것을 막기 위한 보수적 상한치
        constexpr double MAX_GRAD_NORM = 100.0;        // Gradient Clipping을 위한 기울기 크기 최대 임계값

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double f_x = 0.0;
            std::array<double, N> g = {0.0};

            // 1. Auto Differentiation (자동 미분)
            // 목적 함수의 현재 위치(x)에서의 값(f_x)과 기울기(g)를 계산합니다. (메모리 재할당 없음 O(1))
            AutoDiff::value_and_gradient(f, x, f_x, g);

            // [Fail-safe] 목적 함수값이 NaN(Not a Number) 또는 Inf(Infinity)로 발산한 경우, 
            // 최적화를 중단하고 가장 마지막에 확인된 유효한 위치(x_valid)를 반환합니다.
            if (std::isnan(f_x) || std::isinf(f_x)) {
                if (verbose)
                    std::cout << " ⚠️ Warning: Numerical explosion detected. Rolling back.\n";
                return x_valid;
            }
            // 현재 상태가 정상(유효)하므로 백업을 갱신합니다.
            x_valid = x;  

            double g_norm_sq = 0.0;    // 현재 기울기의 L2 Norm 제곱
            double g_dot_gprev = 0.0;  // 현재 기울기(g)와 이전 기울기(g_prev)의 내적

            // 2. 기울기 크기(Norm) 및 내적(Dot product) 동시 계산
            // SIMD 및 FMA(Fused Multiply-Add) 명령어를 통해 하드웨어 가속을 유도합니다.
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq = std::fma(g[i], g[i], g_norm_sq);           // g_norm_sq += g[i] * g[i]
                g_dot_gprev = std::fma(g[i], g_prev[i], g_dot_gprev);  // g_dot_gprev += g[i] * g_prev[i]
            }

            // 3. 수렴(종료) 조건 검사
            // 기울기의 크기 제곱이 목표 허용 오차 제곱보다 작다면 (즉, 극소점에 충분히 가까워졌다면) 종료
            if (g_norm_sq < tol_sq) {
                break;
            }

            // [안전성 보장] Gradient Clipping (기울기 자르기)
            // 기울기가 비정상적으로 커서 매개변수가 우주로 날아가는(발산하는) 것을 방지합니다.
            double g_norm = std::sqrt(g_norm_sq);
            // 기울기 크기가 최대 허용치(MAX_GRAD_NORM)를 넘으면 그 비율만큼 축소(Scale down)합니다.
            double clip_scale = (g_norm > MAX_GRAD_NORM) ? (MAX_GRAD_NORM / g_norm) : 1.0;

            // 4. 하이퍼그레디언트(학습률 alpha) 업데이트
            // g_dot_gprev (이전 기울기와 현재 기울기의 내적)이 양수이면, 같은 방향으로 가고 있으므로 보폭(alpha) 증가.
            // 음수이면, 계곡을 지그재그로 넘나들고(Overshooting) 있다는 뜻이므로 보폭(alpha) 감소.
            // 이때 클리핑된 스케일을 제곱하여 보폭 업데이트에도 안전성을 반영합니다.
            alpha += mu * (g_dot_gprev * clip_scale * clip_scale);
            
            // 학습률이 설정된 안전 범위 [MIN_ALPHA, MAX_ALPHA]를 벗어나지 않도록 제한(Clamp)합니다.
            alpha = std::clamp(alpha, MIN_ALPHA, MAX_ALPHA);

            // 5. 파라미터 업데이트 (In-place)
            // 새로운 위치 = 현재 위치 - 학습률 * 기울기
#pragma omp simd
            for (size_t i = 0; i < N; ++i) {
                g[i] *= clip_scale;          // 클리핑(비율 축소)을 실제 기울기 벡터에 적용
                x[i] -= alpha * g[i];        // 경사하강법 스텝 이동
                g_prev[i] = g[i];            // 다음 반복(Iteration)을 위해 현재 기울기를 저장
            }

            // 일정 주기(5000회)마다 진행 상태를 콘솔에 출력합니다.
            if (verbose && (iter % 5000 == 0)) {
                std::cout << "[Iter " << std::setw(5) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << " | ||g||: " << g_norm
                          << " | alpha: " << alpha << "\n";
            }
        }
        
        // 탐색이 끝난 후 유효한 최적해를 반환
        return x_valid;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_HYPER_GRADIENT_DESCENT_HPP_