#ifndef OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_
#define OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {

/**
 * @brief Algorithm 4.2 : Backtracking Line Search (백트래킹 선 탐색) 알고리즘
 * 
 * 최적화 과정에서 주어진 탐색 방향(d)으로 얼마나 멀리 이동할지(스텝 사이즈, alpha)를 결정하는 근사 선 탐색(Inexact Line Search) 기법입니다.
 * 함수 값이 충분히 감소했는지를 판단하는 **아르미호(Armijo) 충분 감소 조건**을 만족할 때까지 스텝 사이즈를 일정 비율(p)로 줄여나갑니다(Backtracking).
 * 
 * 특징:
 * 1. 정확한 최소점을 찾는 Exact Line Search 보다 계산 비용이 훨씬 저렴하며, 실전에서 널리 쓰입니다.
 * 2. 초기 스텝 사이즈(alpha=1.0)부터 시작하여 조건을 만족하지 못하면 뒤로 물러나므로 "Backtracking"이라 부릅니다.
 * 3. 뉴턴 계열(Newton, BFGS) 최적화에서는 1.0 스텝이 자연 스텝이므로 초기 alpha로 1.0을 권장합니다.
 * 
 * @note 성능 최적화: SIMD 병렬 처리(#pragma omp simd) 및 FMA(Fused Multiply-Add) 명령어를 활용하여 
 *       동적 메모리 할당 없이 O(1) 정적 배열 기반으로 초고속 연산을 수행합니다.
 */
class BacktrackingLineSearch {
   private:
    /**
     * @brief 탐색 방향(Ray)을 따라 일정 거리(alpha)만큼 이동한 새로운 좌표점을 반환합니다.
     * 
     * 수식: pt = x + alpha * d
     * 
     * @tparam N 차원 수
     * @param x 현재 위치 벡터
     * @param d 탐색 방향 벡터 (예: 뉴턴 방향, 기울기 반대 방향 등)
     * @param alpha 스텝 사이즈 (이동 거리)
     * @return std::array<double, N> 이동된 새로운 위치 벡터
     */
    template <size_t N>
    [[nodiscard]] static constexpr std::array<double, N> ray_point(const std::array<double, N>& x,
                                                                   const std::array<double, N>& d,
                                                                   double alpha) noexcept {
        std::array<double, N> pt = {0.0};
// 컴파일러에게 벡터화(SIMD)를 지시하여 하드웨어 병렬 처리를 극대화합니다.
#pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            // std::fma(a, b, c) = a * b + c : 부동소수점 오차를 줄이고 속도를 높이는 FMA 연산
            pt[i] = std::fma(alpha, d[i], x[i]);  // x[i] + (alpha * d[i])
        }
        return pt;
    }

   public:
    /**
     * @brief 함수값과 기울기(Gradient)가 이미 계산된 상태에서 백트래킹 선 탐색을 수행합니다.
     * 
     * @tparam N 차원 수
     * @tparam Func 목적 함수 타입
     * @param f 목적 함수
     * @param x 현재 위치 벡터
     * @param d 탐색 방향 벡터
     * @param f_x 현재 위치에서의 목적 함수 값 (f(x))
     * @param grad_x 현재 위치에서의 목적 함수 기울기 벡터 (∇f(x))
     * @param alpha 초기 스텝 사이즈 (기본값: 1.0)
     * @param p 스텝 사이즈 감소 비율 (Decay rate, 보통 0.1 ~ 0.8 사용. 기본값: 0.5)
     * @param c Armijo 조건의 엄격함을 결정하는 상수 (보통 1e-4 처럼 매우 작은 양수 사용. 작을수록 쉽게 승인됨)
     * @param verbose 진행 상태 로그 출력 여부
     * @return double 조건을 만족하는 최종 스텝 사이즈 (alpha)
     */
    template <size_t N, typename Func>
    [[nodiscard]] static double search(Func f, const std::array<double, N>& x,
                                       const std::array<double, N>& d, double f_x,
                                       const std::array<double, N>& grad_x, double alpha = 1.0,
                                       double p = 0.5, double c = 1e-4,
                                       bool verbose = false) noexcept {
        // 1. 방향 도함수 (Directional Derivative) 계산 : ∇f(x)^T * d
        // 탐색 방향 d로 이동할 때, 시작점에서 함수값이 감소하는 초기 기울기를 의미합니다.
        double dir_deriv = 0.0;
#pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            dir_deriv = std::fma(grad_x[i], d[i], dir_deriv);
        }

        // 방향 도함수가 0 이상이라는 것은 현재 방향(d)이 목적 함수 값을 줄이는 '하강 방향(Descent Direction)'이 아님을 의미합니다.
        if (dir_deriv >= 0.0 && verbose) {
            std::cout << "  [Warning] Not a descent direction! dir_deriv: " << dir_deriv << "\n";
        }

        size_t iter = 0;
        
        // 2. Armijo 조건을 만족할 때까지 반복하며 alpha를 줄여나갑니다.
        while (true) {
            iter++;
            
            // 현재 스텝 사이즈(alpha)만큼 이동한 새로운 좌표(x_new) 산출
            auto x_new = ray_point<N>(x, d, alpha);
            
            // 새로운 좌표에서의 목적 함수 값(f_new) 평가
            double f_new = AutoDiff::value<N>(f, x_new);

            // 3. 목표 감소치(Target Value) 계산 (Armijo 충분 감소 조건)
            // 수식: f(x + alpha * d) <= f(x) + c * alpha * (∇f(x)^T * d)
            // 단순히 함수값이 감소하는 것(f_new < f_x)만으로는 무한 루프에 빠질 수 있으므로,
            // 1차 테일러 전개 기반의 예측 감소량에 상수 c를 곱한 만큼은 '최소한' 감소해야 한다고 엄격하게 제한합니다.
            double target_val = f_x + c * alpha * dir_deriv;

            if (verbose) {
                std::cout << "  ↳ [Backtrack Iter " << std::setw(2) << iter
                          << "] alpha: " << std::fixed << std::setprecision(6) << alpha
                          << " | f_new: " << f_new << " | Target: " << target_val << "\n";
            }

            // Armijo 조건 검사 : 함수 값이 목표치보다 작거나 같으면 승인(Accept)하고 스텝 사이즈 확정
            if (f_new <= target_val) {
                if (verbose) std::cout << "  ↳ [Accepted] Armijo condition satisfied!\n";
                return alpha;
            }

            // 조건을 만족하지 못했으므로 스텝 사이즈를 p 비율(예: 절반)만큼 줄여서 다시 시도
            alpha *= p;

            // 4. 무한 루프 방지를 위한 Failsafe (수치적으로 너무 작은 alpha는 의미가 없음)
            if (alpha < 1e-10) {
                if (verbose) std::cout << "  ↳ [Failsafe] Alpha reached minimum limit.\n";
                return alpha;
            }
        }
    }

    /**
     * @brief 함수값과 기울기를 함수 내부에서 자동으로 계산한 후 선 탐색을 수행하는 편의성(Overload) 함수입니다.
     * 
     * @tparam N 차원 수
     * @tparam Func 목적 함수 타입
     * @param f 목적 함수
     * @param x 현재 위치 벡터
     * @param d 탐색 방향 벡터
     * @param alpha 초기 스텝 사이즈 (기본값: 1.0)
     * @param p 스텝 사이즈 감소 비율 (기본값: 0.5)
     * @param c Armijo 조건의 엄격함을 결정하는 상수 (기본값: 1e-4)
     * @param verbose 진행 상태 로그 출력 여부
     * @return double 조건을 만족하는 최종 스텝 사이즈 (alpha)
     */
    template <size_t N, typename Func>
    [[nodiscard]] static double search(Func f, const std::array<double, N>& x,
                                       const std::array<double, N>& d, double alpha = 1.0,
                                       double p = 0.5, double c = 1e-4,
                                       bool verbose = false) noexcept {
        double f_x = 0.0;
        std::array<double, N> grad_x = {0.0};
        
        // AutoDiff를 통해 1차 도함수(기울기)와 함수값을 한 번에 O(1)로 평가합니다.
        AutoDiff::value_and_gradient<N>(f, x, f_x, grad_x);

        // 위에서 구현된 본체 search 함수 호출
        return search<N>(f, x, d, f_x, grad_x, alpha, p, c, verbose);
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_BACKTRACKING_LINE_SEARCH_HPP_