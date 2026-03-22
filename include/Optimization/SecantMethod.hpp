#ifndef OPTIMIZATION_SECANT_METHOD_HPP_
#define OPTIMIZATION_SECANT_METHOD_HPP_

#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
    /**
     * @brief 1차원 최적화 결과를 담는 구조체
     */
    struct OptimizationResult1D {
        double x_opt;
        double f_opt;
        size_t iterations;
        long long elapsed_ns;       // 실제 소요 시간 (나노초)
    };

    /**
     * @brief Secant Method (할선법) 최적화 알고리즘 (Algorithm 6.2)
     * @note 1차원 (Univariate) 스칼라 함수의 2차 미분 (Hessian)을
     * 이전 두 지점의 1차 미분 (Gradient) 차이로 근사하여 최적해를 찾습니다.
     * AutoDiff를 연동하여 목적 함수 (f)만으로 최적화를 수행합니다
     */
    class SecantMethod {
        public:
            SecantMethod() = delete;    // 인스턴스화 방지
            template <typename Func>
            [[nodiscard]] static OptimizationResult1D optimize(
                Func f, double x0, double x1, double tol=1e-6, size_t max_iter=100, bool verbose=false) {
                auto start_clock = std::chrono::high_resolution_clock::now();

                // AutoDiff는 std::array 기반이므로, 1D 스칼라를 감싸는 래퍼 람다 생성 (Inlining)
                auto eval_grad = [&f](double x) -> double {
                    return AutoDiff::gradient<1>(f, std::array<double, 1>{x})[0];
                };

                double g0 = eval_grad(x0);
                double delta = 0.0;
                size_t iter = 0;

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🎯 Secant Method Started \n";
                    std::cout << "========================================================\n";
                }

                for (iter = 1; iter <= max_iter; ++iter) {
                    double g1 = eval_grad(x1);

                    double diff = g1 - g0;

                    // [안전성 보장] 분모가 0이 되는 것을 방지 (지형이 평탄하여 기울기가 같을 때 발산 방어)
                    if (std::abs(diff) < 1e-12) {
                        if (verbose) 
                            std::cout << "  ↳ ⚠️ Terminated due to zero gradient difference.\n";
                        break;
                    }

                    // 1. 할선법 업데이트 수식 
                    delta = (x1 - x0) / diff * g1;

                    // 2. 상태 업데이트 (과거 상태를 현재로 덮어씌움)
                    x0 = x1;
                    g0 = g1;
                    x1 = x1 - delta;

                    if (verbose) {
                        std::cout << "[Iter " << std::setw(3) << iter 
                                << "] x: " << std::fixed << std::setprecision(6) << x1 
                                << " | f'(x): " << g1 
                                << " | |Δ|: " << std::abs(delta) << "\n";
                    }

                    // 3. 종료 조건 검사 (이동 거리가 허용 오차 이하이거나, 기울기가 0에 극도로 가까울 때)
                    if (std::abs(delta) < tol || std::abs(g1) < 1e-9) {
                        if (verbose) 
                            std::cout << "  ↳ ✅ Converged at Iteration: " << iter << "\n";
                        break;
                    }
                }

                auto end_clock = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock);

                double f_opt = AutoDiff::value<1>(f, std::array<double, 1>{x1});

                if (verbose) {
                    std::cout << "========================================================\n";
                    std::cout << " 🏁 Final Optimal Point: " << x1 << "\n";
                    std::cout << "========================================================\n";
                }
                return {x1, f_opt, iter, duration.count()};
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_SECANT_METHOD_HPP_