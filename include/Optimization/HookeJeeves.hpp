#ifndef OPTIMIZATION_HOOKE_JEEVES_HPP_
#define OPTIMIZATION_HOOKE_JEEVES_HPP_

/**
 * @file HookeJeeves.hpp
 * @brief Hooke-Jeeves Pattern Search 최적화 알고리즘 구현
 *
 * Hooke-Jeeves 알고리즘은 목적 함수의 미분(Gradient) 정보를 사용하지 않는
 * 대표적인 직접 탐색법(Direct Search Method) 중 하나입니다.
 * 이 알고리즘은 각 변수 축을 따라 이동해보는 탐색 이동(Exploratory Move)과,
 * 과거의 성공적인 이동 방향을 기반으로 가속하여 이동하는 패턴 이동(Pattern Move)을
 * 번갈아가며 수행하여 최솟값을 찾습니다. 미분이 불가능하거나 노이즈가 많은 함수에 유용합니다.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 저장하는 구조체
 * @tparam N 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt;  ///< 최적화된 변수 값 (최적해)
    double f_opt;                 ///< 최적해에서의 목적 함수 값
    size_t iterations;            ///< 수렴할 때까지 걸린 총 반복 횟수
    long long elapsed_ns;         ///< 알고리즘 실행에 소요된 시간 (나노초)
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @class HookeJeeves
 * @brief Hooke-Jeeves Pattern Search 알고리즘을 수행하는 정적 클래스
 *
 * @note 미분 (Gradient) 없이 탐색 (Exploratory)과 패턴 이동 (Pattern)을 결합하여 탐색 속도를
 * 가속합니다. 성능 향상을 위해 힙 동적 할당을 차단하고, 캐시 라인 정렬 (alignas(64))을 적용하여
 * 최적화되었습니다.
 */
class HookeJeeves {
   private:
    /**
     * @brief 탐색 이동 (Exploratory Move / 더듬기)
     * @details 현재 위치에서 각 차원(축)을 기준으로 양의 방향과 음의 방향으로 순차적으로
     * 탐색합니다. 한 축 방향으로의 이동이 함수값을 감소시킨다면(개선된다면), 즉시 그 위치를
     * 확정하고 다음 축 탐색을 이어갑니다.
     *
     * @tparam N 차원 수
     * @tparam Func 목적 함수 타입
     * @param f 목적 함수
     * @param x 탐색을 시작할 베이스 위치
     * @param alpha 현재 설정된 탐색 보폭 (Step size)
     * @param[in,out] f_best 현재까지 발견된 최소 함수값 (탐색 중 갱신됨)
     * @return std::array<double, N> 탐색 이동이 완료된 후의 새로운 위치
     */
    template <size_t N, typename Func>
    static constexpr std::array<double, N> explore(Func f, std::array<double, N> x, double alpha,
                                                   double& f_best) noexcept {
        for (size_t i = 0; i < N; ++i) {
            // 1. 양 (+)의 방향 탐색
            x[i] += alpha;
            double f_plus = f(x);
            if (f_plus < f_best) {
                // 개선되었을 경우: 최적값 갱신 후 해당 축 탐색 종료 (위치 유지)
                f_best = f_plus;
                continue;
            }

            // 2. 음 (-)의 방향 탐색 (-2.0 * alpha를 빼서 원래 위치를 지나 반대편으로 이동)
            x[i] -= 2.0 * alpha;
            double f_minus = f(x);
            if (f_minus < f_best) {
                // 개선되었을 경우: 최적값 갱신 후 해당 축 탐색 종료 (위치 유지)
                f_best = f_minus;
                continue;
            }

            // 3. 양쪽 모두 탐색에 실패(개선 없음) 시 원래 위치로 복구 (+alpha)
            x[i] += alpha;
        }
        return x;  // 각 축별 탐색 결과가 반영된 최종 위치 반환
    }

   public:
    // 인스턴스화 방지
    HookeJeeves() = delete;

    /**
     * @brief Hooke-Jeeves 최적화를 실행합니다.
     *
     * @tparam N 차원 수
     * @tparam Func 목적 함수 타입
     * @param f 최적화할 목적 함수
     * @param x_start 초기 시작 위치
     * @param alpha 초기 탐색 보폭 (Step Size). 기본값: 0.5
     * @param shrink 보폭 축소 비율. 탐색 실패 시 보폭을 줄이는 비율입니다. 기본값: 0.5
     * @param tol 수렴 기준. 보폭(alpha)이 이 값보다 작아지면 최적화 종료. 기본값: 1e-6
     * @param max_iter 최대 반복 횟수. 기본값: 2000
     * @param verbose 진행 상황 출력 여부. 기본값: false
     * @return OptimizationResultND<N> 최적해, 함수값, 반복 횟수, 소요 시간을 담은 구조체
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_start,
                                                          double alpha = 0.5, double shrink = 0.5,
                                                          double tol = 1e-6, size_t max_iter = 2000,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0");
        auto start_clock = std::chrono::high_resolution_clock::now();

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🐸 Hooke-Jeeves Pattern Search Started\n";
            std::cout << "========================================================\n";
        }

        // 메모리 정렬을 통한 SIMD / 캐시 최적화
        alignas(64) std::array<double, N> x = x_start;
        double f_best = f(x);
        size_t iter = 0;

        for (iter = 1; iter <= max_iter; ++iter) {
            double f_prev = f_best;

            // Step 1. 현재 베이스 포인트(x)에서 탐색 이동 (Exploratory Move) 수행
            alignas(64) std::array<double, N> x_new = explore<N>(f, x, alpha, f_best);

            // 부동 소수점 좌표 배열 자체를 비교하는 대신, 목적 함수값의 개선 여부로 탐색 성공을
            // 판별합니다. 이는 부동소수점 오차에 대해 더 강건(Robust)합니다.
            if (f_best < f_prev) {
                // 탐색 이동 성공: 이전 베이스(x)에서 새로운 베이스(x_new)로 이동한 방향을 기억함.

                // Step 2. 패턴 이동 (Pattern Move) 계산
                // 모멘텀 가속: 과거 이동 방향(x_new - x)을 연장하여 한 번 더 도약해봅니다.
                alignas(64) std::array<double, N> x_pattern = {0.0};
#pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    // 수식: x_pattern = x_new + (x_new - x) = 2 * x_new - x
                    // FMA(Fused Multiply-Add) 명령어를 사용하여 계산 속도 및 정밀도 향상
                    x_pattern[i] = std::fma(2.0, x_new[i], -x[i]);
                }

                // 점프(Pattern Move)한 위치에서의 함수값 계산
                double f_pattern = f(x_pattern);

                // 도약한 위치(x_pattern)에서 다시 한 번 탐색 이동(Exploratory Move) 수행
                alignas(64) std::array<double, N> x_pattern_new =
                    explore<N>(f, x_pattern, alpha, f_pattern);

                // 도약 후의 결과가 단순 탐색 이동 결과(f_best)보다 더 좋은지 검증
                if (f_pattern < f_best) {
                    // 패턴 이동 성공: 더 좋은 결과를 얻었으므로 베이스 위치와 최적값을 도약 위치로
                    // 갱신
                    x = x_pattern_new;
                    f_best = f_pattern;
                } else {
                    // 패턴 이동 실패: 도약이 오히려 안 좋은 결과를 낳았으므로 단순 탐색
                    // 결과(x_new)만 적용 (롤백)
                    x = x_new;
                }
            } else {
                // 탐색 이동 실패 (f_best >= f_prev):
                // 현재 위치 근처에서 더 이상 좋은 곳을 찾을 수 없음을 의미함.

                // Step 3. 보폭 수축 (Shrink)
                // 주변 탐색에 실패했으므로, 더 세밀한 탐색을 위해 보폭(alpha)을 줄입니다.
                alpha *= shrink;

                // 보폭이 설정된 허용 오차(tol)보다 작아지면 충분히 극솟값에 도달했다고 판단하고
                // 종료합니다.
                if (alpha < tol) {
                    if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                    break;
                }
            }

            // 지정된 주기마다 현재 진행 상태 출력
            if (verbose && (iter % 100 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(4) << iter << "] f(x): " << std::fixed
                          << std::setprecision(6) << f_best << " | Step(alpha): " << alpha << "\n";
            }
        }

        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << x[0];
            if constexpr (N > 1) std::cout << ", " << x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        return {x, f_best, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_HOOKE_JEEVES_HPP_