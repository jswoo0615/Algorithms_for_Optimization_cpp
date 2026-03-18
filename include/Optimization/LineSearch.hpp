#ifndef OPTIMIZATION_LINE_SEARCH_HPP_
#define OPTIMIZATION_LINE_SEARCH_HPP_

#include <array>
#include <cmath>
#include <utility>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "Optimization/AutoDiff.hpp"

namespace Optimization {
    class LinearSearch {
        private:
            template <size_t N>
            static std::array<double, N> ray_point(const std::array<double, N>& x, const std::array<double, N>& d, double alpha) {
                std::array<double, N> pt;
                for (size_t i = 0; i < N; ++i) {
                    pt[i] = x[i] + alpha * d[i];
                }
                return pt;
            }

            template <size_t N, typename Func>
            static double directional_derivative(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double alpha) {
                auto pt = ray_point<N>(x, d, alpha);
                auto grad = Functional::gradient<N>(f, pt);
                double dir_deriv = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    dir_deri += grad[i] * d[i];
                }
                return dir_deriv;
            }
        public:
            // ==================================================================
            // 1. Bracket Minimum (Algorithm 3.1)
            // 핵심 수학적 아이디어 : a < b < c 위치에 있는 세 점을 찾을 때,
            // 함수값이 f(a) > f(b)이면서 동시에 f(b) < f(c)를 만족한다면
            // 함수가 감소하다가 다시 증가한 것이므로, 구간 [a, c] 내부에 반드시 최솟값이 존재
            // ==================================================================
            template <size_t N, typename Func>
            static std::pair<double, double> bracket_minimum(Func f, const std::array<double, N>& x, const std::array<double, N>& d, double s = 1e-2, double k = 2.0, bool verbose = false) {
                double a = 0.0;
                double ya = AutoDiff::value<N>(f, ray_point(x, d, a));
                double b = s;
                double yb = AutoDiff::value<N>(f, ray_point<N>(x, d, b));

                /**
                 * 1. 초기화 및 초기 방향 설정
                 * 시작점 x (기본값 0)에서 시작하여, 아주 작은 기본 보폭 s (0.01)만큼 이동한 지점을 평가
                 * 만약 새로운 지점의 함수값 (yb)이 시작점 (ya)보다 크다면 오르막 방향이 잘못 간 것이므로, 
                 * 두 지점을 맞바꾸고 보폭의 부호를 반대로 (s = -s) 설정하여 내리막 방향으로 탐색 방향 전환
                 */
                if (yb > ya) { // f(a) > f(b)이면서
                    std::swap(a, b);
                    std::swap(ya, yb);
                    s = -s;
                }

                size_t iter = 0;
                while (true) {
                    iter++;
                    /**
                     * 새로운 지점 평가 (루프 시작)
                     * 현재 지점 b에서 보폭 s만큼 더 나아간 새로운 지점 c = b + s를 잡고 함수값 yc를 계산
                     */
                    double c = b + s;
                    double yc = AutoDiff::value<N>(f, ray_point<N>(f, d, c));

                    if (verbose) {
                        std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] a: " << std::fixed << std::setprecision(5) << a 
                                  << " | b: " << b << " | c: " << c << " | f(c): " << yc << "\n";
                    }

                    if (yc > yb) {
                        /**
                         * 종료 조건 확인 (최솟값 구간 발견)
                         * 만약 새로운 지점의 함수값 yc가 이전 지점의 함수값 yb보다 크다면 (yc > yb),
                         * 함수가 내리막을 걷다가 다시 오르막으로 꺾인 것입니다.
                         * 이 순간 f(a) > f(b) < f(c) 조건이 성립하므로
                         * a와 c 사이의 구간을 최솟값이 포함된 최종 브래킷으로 반환하고 탐색 종료합니다.
                         */
                        return (a < c) ? std::make_pair(a, c) : std::make_pair(c, a);
                    }

                    /**
                     * 여전히 내리막일 경우 (yc <= yb)
                     * 함수가 여전히 내리막길이어서 탐색 지점을 앞으로 한 칸씩 전진시킬 때입니다.
                     * c 지점의 함수값이 b 지점보다 여전히 작거나 같아서 아직 골짜기의 바닥을 지나치지 않았다는 뜻입니다.
                     * 따라서 탐색 윈도우를 한 칸 앞으로 이동시키기 위해
                     * 기존 b를 새로운 시작점 a로 버리고, 기존의 c를 새로운 b로 갱신
                     */
                    a = b;
                    ya = yb;
                    b = c;
                    yb = yc;
                    s *= k;
                }
            }
    };
} // namespace Optimization

#endif // OPTIMIZATION_LINE_SEARCH_HPP_