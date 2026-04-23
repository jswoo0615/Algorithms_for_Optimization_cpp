#ifndef OPTIMIZATION_NELDER_MEAD_HPP_
#define OPTIMIZATION_NELDER_MEAD_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

// 최적화 결과를 담는 구조체 중복 정의 방지용 매크로
#ifndef OPTIMIZATION_RESULT_ND_DEFINED
#define OPTIMIZATION_RESULT_ND_DEFINED
namespace Optimization {
/**
 * @brief N차원 최적화 결과를 저장하는 구조체
 * @tparam N 최적화 대상 변수의 차원 수
 */
template <size_t N>
struct OptimizationResultND {
    std::array<double, N> x_opt; ///< 최적해 (Optimal solution)
    double f_opt;                ///< 최적 함수값 (Optimal function value)
    size_t iterations;           ///< 총 반복 횟수 (Total iterations)
    long long elapsed_ns;        ///< 소요 시간 (나노초 단위)
};
}  // namespace Optimization
#endif  // OPTIMIZATION_RESULT_ND_DEFINED

namespace Optimization {

/**
 * @class NelderMead
 * @brief Nelder-Mead Simplex Method (넬더-미드 심플렉스 방법) 정적 클래스
 * 
 * @details Nelder-Mead 알고리즘은 함수의 미분 정보(기울기)를 사용하지 않는 Derivative-Free Optimization(DFO) 기법입니다.
 * N차원 공간에서 N+1개의 정점(Vertex)으로 이루어진 다면체인 '심플렉스(Simplex)'를 구성하고, 
 * 함수값이 가장 나쁜(Worst) 정점을 중심점(Centroid)을 기준으로 반대편으로 넘기는(Reflection) 등의 기하학적 변형을 반복하며 
 * 마치 '아메바(Amoeba)'가 기어가듯 최솟값을 향해 이동합니다.
 * 
 * 주요 연산:
 * 1. Reflection (반사): 최악의 점을 반대편으로 넘김
 * 2. Expansion (확장): 반사한 점이 매우 좋으면 그 방향으로 더 멀리 나아감
 * 3. Contraction (수축): 반사한 점이 여전히 나쁘면 탐색 보폭을 줄임 (Outside/Inside)
 * 4. Shrinkage (축소): 수축마저 실패하면 모든 점을 최우수 점(Best)을 향해 당김
 * 
 * @note 본 구현체는 다음과 같은 성능 최적화를 포함합니다:
 * 1. **MISRA C++ 준수 및 O(1) 동적 할당 제로**: std::vector 대신 std::array를 사용하여 힙(Heap) 메모리 할당 오버헤드 제거
 * 2. **SIMD / FMA 가속**: #pragma omp simd 및 std::fma를 적극 활용하여 벡터 연산 및 분산 계산 속도 극대화
 */
class NelderMead {
   private:
    /**
     * @brief 심플렉스를 구성하는 하나의 정점(Vertex) 정보
     * @tparam N 차원 수
     */
    template <size_t N>
    struct Vertex {
        alignas(64) std::array<double, N> x; ///< 정점의 위치 (캐시 정렬됨)
        double f_val;                        ///< 해당 위치에서의 목적 함수 평가값
        
        // 정점들을 함수값 기준으로 오름차순 정렬하기 위한 연산자 오버로딩 (f_val이 작을수록 좋은 점)
        bool operator<(const Vertex& other) const noexcept { return f_val < other.f_val; }
    };

   public:
    NelderMead() = delete; // 정적 메서드만 제공하므로 인스턴스화 방지

    /**
     * @brief Nelder-Mead 알고리즘을 수행하여 최적해를 찾는 함수
     * 
     * @param f 최적화할 목적 함수 (미분 불필요)
     * @param x_start 초기 탐색 시작 위치 (N차원 벡터)
     * @param step 초기 심플렉스를 구성하기 위해 직교 방향으로 이동할 스텝 크기 (기본값: 1.0)
     * @param tol 허용 오차. 심플렉스 정점들의 함수값 분산(Variance)이 이 값의 제곱보다 작아지면 수렴한 것으로 판단 (기본값: 1e-6)
     * @param max_iter 최대 반복 횟수 (기본값: 2000)
     * @param verbose 콘솔에 진행 과정을 출력할지 여부
     * @return 최적해(x_opt), 최적값(f_opt), 반복 횟수, 연산 시간을 포함하는 구조체
     */
    template <size_t N, typename Func>
    [[nodiscard]] static OptimizationResultND<N> optimize(Func f, std::array<double, N> x_start,
                                                          double step = 1.0, double tol = 1e-6,
                                                          size_t max_iter = 2000,
                                                          bool verbose = false) noexcept {
        static_assert(N > 0, "Dimension N must be greater than 0"); // 컴파일 타임 차원 검사
        auto start_clock = std::chrono::high_resolution_clock::now(); // 소요 시간 측정 시작

        // Nelder-Mead 형태 변환 파라미터 (표준 설정값)
        const double alpha = 1.0;         // 반사 계수 (Reflection)
        const double gamma = 2.0;         // 확장 계수 (Expansion)
        const double rho = 0.5;           // 수축 계수 (Contraction)
        const double sigma = 0.5;         // 축소 계수 (Shrinkage)
        
        // 수렴 조건 검사 시 불필요한 제곱근(sqrt) 연산을 피하기 위해 오차의 제곱을 미리 계산
        const double tol_sq = tol * tol;  

        // 1. 초기 심플렉스 (Simplex) 생성
        // N차원 공간에서는 N+1개의 정점이 필요합니다. (예: 2차원이면 3개의 정점인 삼각형 구성)
        std::array<Vertex<N>, N + 1> simplex = {};
        
        // 첫 번째 정점은 주어진 초기 위치로 설정
        simplex[0].x = x_start;
        simplex[0].f_val = f(simplex[0].x);

        // 나머지 N개의 정점은 초기 위치에서 각 축 방향으로 step만큼 이동하여 생성 (직교 정점)
        for (size_t i = 0; i < N; ++i) {
            simplex[i + 1].x = x_start;
            simplex[i + 1].x[i] += step;  
            simplex[i + 1].f_val = f(simplex[i + 1].x);
        }

        // 반복문 시작 전 최초 1회 정렬 (0: Best, N: Worst)
        std::sort(simplex.begin(), simplex.end());

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🦠 Nelder-Mead Simplex (The Amoeba) Started\n";
            std::cout << "========================================================\n";
        }

        size_t iter = 0;
        for (iter = 1; iter <= max_iter; ++iter) {
            
            // 2. 정점 정렬 (0번 인덱스가 최고(Best), N번 인덱스가 최악(Worst)의 점이 됨)
            // 매 이터레이션의 끝에서 정점이 갱신되므로 항상 정렬 상태를 유지해야 함
            std::sort(simplex.begin(), simplex.end());

            // 3. 수렴 조건 검사 (Variance Check)
            // N+1개의 함수값들의 분산을 계산하여, 이 분산이 허용 오차의 제곱(tol_sq) 미만이면 수렴한 것으로 판단
            double mean_y = 0.0;
#pragma omp simd
            for (size_t i = 0; i <= N; ++i) {
                mean_y += simplex[i].f_val;
            }
            mean_y /= (N + 1);

            double var_y = 0.0;
#pragma omp simd
            for (size_t i = 0; i <= N; ++i) {
                double diff = simplex[i].f_val - mean_y;
                // var_y = diff * diff + var_y 를 fma로 단일 명령어 처리
                var_y = std::fma(diff, diff, var_y); 
            }
            var_y /= (N + 1);

            if (var_y < tol_sq) {
                if (verbose) std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                break;
            }

            // 4. Centroid (무게중심) 계산
            // 가장 함수값이 나쁜 점(simplex[N])을 제외한 나머지 N개 정점들의 평균 위치를 구함
            alignas(64) std::array<double, N> x_o = {0.0};
            for (size_t i = 0; i < N; ++i) {  // i < N (최악점 인덱스 N은 제외됨)
#pragma omp simd
                for (size_t j = 0; j < N; ++j) {
                    x_o[j] += simplex[i].x[j];
                }
            }
#pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                x_o[j] /= N;
            }

            // 5. 반사 (Reflection)
            // 최악의 점을 중심점(x_o)을 기준으로 반대편으로 넘김
            alignas(64) std::array<double, N> x_r;
#pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                // x_r = x_o + alpha * (x_o - x_worst)
                x_r[j] = std::fma(alpha, x_o[j] - simplex[N].x[j], x_o[j]);
            }
            double f_r = f(x_r);

            // 반사된 점(x_r)이 최고점(Best)보다는 나쁘지만 차악점(Second Worst, N-1)보다는 좋을 경우
            // 평범하게 반사된 점을 수용함
            if (f_r >= simplex[0].f_val && f_r < simplex[N - 1].f_val) {
                simplex[N].x = x_r;
                simplex[N].f_val = f_r;
            } 
            // 반사된 점이 현재 최고점(Best)보다도 더 좋은 경우 (올바른 방향을 찾았음)
            else if (f_r < simplex[0].f_val) {
                // 6. 확장 (Expansion)
                // 방향이 좋으므로 반사된 방향으로 더 멀리(gamma) 확장하여 탐색함
                alignas(64) std::array<double, N> x_e;
#pragma omp simd
                for (size_t j = 0; j < N; ++j) {
                    // x_e = x_o + gamma * (x_r - x_o)
                    x_e[j] = std::fma(gamma, x_r[j] - x_o[j], x_o[j]);
                }
                double f_e = f(x_e);

                // 확장한 점이 반사한 점보다 더 좋으면 확장을 채택, 아니면 그냥 반사를 채택
                if (f_e < f_r) {
                    simplex[N].x = x_e;
                    simplex[N].f_val = f_e;
                } else {
                    simplex[N].x = x_r;
                    simplex[N].f_val = f_r;
                }
            } 
            // 반사된 점이 차악점(Second Worst)보다도 나쁜 경우 (해당 방향이 아니거나 계곡임)
            else {
                // 7. 수축 (Contraction)
                bool do_shrink = false; // 축소(Shrinkage) 연산 수행 플래그

                // 그래도 최악의 점(Worst)보다는 반사된 점이 나은 경우 -> 외부 수축 (Outside Contraction)
                if (f_r < simplex[N].f_val) {
                    alignas(64) std::array<double, N> x_c;
#pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        // x_c = x_o + rho * (x_r - x_o) -> 반사점과 중심점의 사이
                        x_c[j] = std::fma(rho, x_r[j] - x_o[j], x_o[j]);
                    }
                    double f_c = f(x_c);

                    // 수축한 점이 반사점보다 좋거나 같으면 채택
                    if (f_c <= f_r) {
                        simplex[N].x = x_c;
                        simplex[N].f_val = f_c;
                    } else {
                        do_shrink = true; // 수축도 실패했으므로 전체 축소
                    }
                } 
                // 반사된 점이 최악의 점보다도 더 나빠진 경우 -> 내부 수축 (Inside Contraction)
                else {
                    alignas(64) std::array<double, N> x_c;
#pragma omp simd
                    for (size_t j = 0; j < N; ++j) {
                        // x_c = x_o + rho * (x_worst - x_o) -> 중심점과 최악점의 사이
                        x_c[j] = std::fma(rho, simplex[N].x[j] - x_o[j], x_o[j]);
                    }
                    double f_c = f(x_c);

                    // 수축한 점이 기존 최악점보다 좋아졌으면 채택
                    if (f_c < simplex[N].f_val) {
                        simplex[N].x = x_c;
                        simplex[N].f_val = f_c;
                    } else {
                        do_shrink = true; // 수축도 실패했으므로 전체 축소
                    }
                }

                // 8. 축소 (Shrinkage)
                // 수축(Contraction)으로도 나은 점을 찾지 못한 경우, 심플렉스가 골짜기에 꼈다고 판단하고
                // 최우수 점(simplex[0])을 제외한 모든 점들을 최우수 점 방향으로 일정 비율(sigma) 당김
                if (do_shrink) {
                    for (size_t i = 1; i <= N; ++i) { // 0번(Best)은 놔두고 나머지 모두
#pragma omp simd
                        for (size_t j = 0; j < N; ++j) {
                            // x_new = x_best + sigma * (x_old - x_best)
                            simplex[i].x[j] =
                                std::fma(sigma, simplex[i].x[j] - simplex[0].x[j], simplex[0].x[j]);
                        }
                        // 위치가 바뀌었으므로 함수값 재평가
                        simplex[i].f_val = f(simplex[i].x);
                    }
                }
            }
            
            // 진행 상태 출력: 첫 이터레이션이거나 매 50번마다 출력
            if (verbose && (iter % 50 == 0 || iter == 1)) {
                std::cout << "[Iter " << std::setw(3) << iter << "] Best f(x): " << std::fixed
                          << std::setprecision(8) << simplex[0].f_val << " | Variance: " << var_y
                          << "\n";
            }
        }

        // 최적화 종료 및 소요 시간 계산
        auto end_clock = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

        if (verbose) {
            std::cout << "========================================================\n";
            std::cout << " 🏁 Final Optimal Point: [" << simplex[0].x[0];
            if constexpr (N > 1) std::cout << ", " << simplex[0].x[1];
            if constexpr (N > 2) std::cout << ", ...";
            std::cout << "]\n========================================================\n";
        }

        // 가장 함수값이 작은 최우수 정점의 위치(simplex[0].x) 반환
        return {simplex[0].x, simplex[0].f_val, iter, duration.count()};
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_NELDER_MEAD_HPP_