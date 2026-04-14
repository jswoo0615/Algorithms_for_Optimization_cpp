#ifndef OPTIMIZATION_CONJUGATE_GRADIENT_HPP_
#define OPTIMIZATION_CONJUGATE_GRADIENT_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/LineSearch.hpp"

namespace Optimization {

/**
 * @brief 공액 기울기법 (Conjugate Gradient Method) 최적화 알고리즘
 *
 * 1차 미분(Gradient) 정보만을 사용하여 최적점을 찾는 알고리즘으로, 순수 경사 하강법(Gradient
 * Descent)이 가진 지그재그(Zig-zag) 현상과 느린 수렴 속도를 개선하기 위해 고안되었습니다.
 *
 * 가장 큰 특징은 이전의 탐색 방향과 새로운 탐색 방향이 서로 '공액(Conjugate, 직교를 일반화한
 * 개념)'을 이루도록 방향을 설정한다는 것입니다. 이를 통해 N차원 2차 함수(Quadratic Function)의 경우
 * 최대 N번의 반복 내에 정확한 최적해를 찾을 수 있다는 수학적 보장이 있습니다.
 *
 * 본 구현에서는 비선형 함수에 널리 사용되는 Fletcher-Reeves 방식을 채택하였으며,
 * 학습률(Learning Rate, alpha)을 고정하지 않고 Exact Line Search(선 탐색)를 통해 매 스텝마다 최적의
 * 보폭을 스스로 찾습니다.
 */
class ConjugateGradient {
   public:
    // 유틸리티 클래스로만 사용되도록 인스턴스 생성을 막음
    ConjugateGradient() = delete;

    /**
     * @brief Conjugate Gradient (Fletcher-Reeves Method) 메인 최적화 함수
     *
     * @tparam N 최적화 변수의 차원 수
     * @tparam Func 목적 함수 타입 (람다식, 펑터 등)
     * @param f 최소화하고자 하는 목적 함수
     * @param x 탐색을 시작할 초기 위치 배열
     * @param max_iter 무한 루프를 방지하기 위한 최대 허용 반복 횟수
     * @param tol 수렴 판정을 위한 허용 오차 (기울기 벡터의 크기가 이 값보다 작아지면 수렴 판정)
     * @param verbose 최적화 진행 상황(반복 횟수, 함수값 등)을 콘솔에 출력할지 여부
     * @return std::array<double, N> 최적화가 완료된 최종 해(파라미터 벡터)
     */
    template <size_t N, typename Func>
    static std::array<double, N> optimize(Func f, std::array<double, N> x, size_t max_iter = 1000,
                                          double tol = 1e-4, bool verbose = false) {
        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " ⚔️ Conjugate Gradient (Fletcher-Reeves) Started\n";
            std::cout << "========================================================\n";
        }

        double f_x;
        std::array<double, N> g;

        // [초기화] AutoDiff를 통해 초기 위치 x에서의 함수값(f_x)과 기울기(g)를 계산합니다.
        AutoDiff::value_and_gradient<N>(f, x, f_x, g);

        // 첫 번째 탐색 방향(d)은 순수 경사 하강법과 동일하게 가장 가파른 역방향(-g)으로 설정합니다.
        std::array<double, N> d;
        for (size_t i = 0; i < N; ++i) {
            d[i] = -g[i];
        }

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            // [Step 1] 현재 기울기 벡터의 크기(Norm의 제곱) 계산 및 수렴 확인
            double g_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_norm_sq += g[i] * g[i];
            }

            // 기울기 크기(루트 g_norm_sq)가 지정된 허용 오차(tol)보다 작으면 극솟값에 도달했다고
            // 판단합니다.
            if (std::sqrt(g_norm_sq) < tol) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // [Step 2] Exact Line Search를 통한 최적의 보폭(alpha) 탐색
            // 사용자가 고정된 학습률(Learning Rate)을 지정하지 않고, 해당 방향(d)으로 가장 많이
            // 감소할 수 있는 최적의 전진 거리(alpha)를 수학적으로 찾아냅니다.

            // 2-1. 브래킷 부호 변경(Bracket Sign Change)을 통해 방향 미분값이 0을
            // 가로지르는(감소하다가 증가하는) 확실한 구간을 잡습니다.
            auto bracket = LineSearch::bracket_sign_change<N>(f, x, d, 0.0, 0.001, 2.0, false);

            // 2-2. 확보된 구간(bracket.first ~ bracket.second) 내에서 이분법(Bisection)을 수행하여
            // 정확한 보폭 alpha를 산출합니다.
            double alpha =
                LineSearch::bisection<N>(f, x, d, bracket.first, bracket.second, 1e-4, false);

            // [Step 3] 실제 파라미터(위치 x) 업데이트
            // 현재 위치에서 최적 보폭(alpha)만큼 탐색 방향(d)으로 전진합니다.
            for (size_t i = 0; i < N; ++i) {
                x[i] += alpha * d[i];
            }

            // [Step 4] 새로운 위치에서의 함수값과 기울기(g_new) 계산
            std::array<double, N> g_new;
            AutoDiff::value_and_gradient<N>(f, x, f_x, g_new);

            // [Step 5] Fletcher-Reeves 공식을 이용한 공액 비율(beta) 계산
            // 이전 방향(d)의 정보를 얼마나 섞어서 다음 방향을 결정할지 그 비율을 계산합니다.
            double g_new_norm_sq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                g_new_norm_sq += g_new[i] * g_new[i];
            }

            // Fletcher-Reeves 식: beta = ||g_new||^2 / ||g||^2
            // 새로운 기울기의 크기와 이전 기울기의 크기의 비율을 사용합니다.
            double beta = g_new_norm_sq / g_norm_sq;

            // [Step 6] 새로운 공액 탐색 방향(Conjugate Direction, d) 설정
            // 새로운 방향은 현재의 가장 가파른 하강 방향(-g_new)에 이전 방향(d)을 beta 비율만큼
            // 섞어 만들어집니다. 이를 통해 이전까지 탐색해온 계곡의 축(Axis) 정보를 잃지 않고
            // 나아갈 수 있습니다.
            for (size_t i = 0; i < N; ++i) {
                d[i] = -g_new[i] + beta * d[i];
            }

            // 다음 루프의 연산을 위해 새로운 기울기(g_new)를 현재 기울기(g)로 덮어씁니다.
            g = g_new;

            // 진행 상태 주기적 로깅 (10번 반복마다 또는 첫 번째 반복 시 출력)
            if (verbose && (iter % 10 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_x << "\n";
            }
        }

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        return x;
    }
};

}  // namespace Optimization

#endif  // OPTIMIZATION_CONJUGATE_GRADIENT_HPP_