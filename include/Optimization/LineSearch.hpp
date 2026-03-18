#ifndef OPTIMIZATION_LINE_SEARCH_HPP_
#define OPTIMIZATION_LINE_SEARCH_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

#include "Optimization/AutoDiff.hpp"

namespace Optimization {
class LineSearch {
   private:
    template <size_t N>
    static std::array<double, N> ray_point(const std::array<double, N>& x,
                                           const std::array<double, N>& d, double alpha) {
        std::array<double, N> pt;
        for (size_t i = 0; i < N; ++i) {
            pt[i] = x[i] + alpha * d[i];
        }
        return pt;
    }

    template <size_t N, typename Func>
    static double directional_derivative(Func f, const std::array<double, N>& x,
                                         const std::array<double, N>& d, double alpha) {
        auto pt = ray_point<N>(x, d, alpha);
        auto grad = AutoDiff::gradient<N>(f, pt);
        double dir_deriv = 0.0;
        for (size_t i = 0; i < N; ++i) {
            dir_deriv += grad[i] * d[i];
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
    static std::pair<double, double> bracket_minimum(Func f, const std::array<double, N>& x,
                                                     const std::array<double, N>& d,
                                                     double s = 1e-2, double k = 2.0,
                                                     bool verbose = false) {
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
        if (yb > ya) {  // f(a) > f(b)이면서
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
            double yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] a: " << std::fixed
                          << std::setprecision(5) << a << " | b: " << b << " | c: " << c
                          << " | f(c): " << yc << "\n";
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
             * c 지점의 함수값이 b 지점보다 여전히 작거나 같아서 아직 골짜기의 바닥을 지나치지
             * 않았다는 뜻입니다. 따라서 탐색 윈도우를 한 칸 앞으로 이동시키기 위해 기존 b를 새로운
             * 시작점 a로 버리고, 기존의 c를 새로운 b로 갱신
             */
            a = b;
            ya = yb;
            b = c;
            yb = yc;
            s *= k;
        }
    }

    // ================================================================================
    // 2. Golden Section Search (Algorithm 3.3)
    // 제한된 평가 횟수 내에서 구간을 최대로 축소하는 피보나치 탐색을 근사한 방법
    // 탐색 횟수 n이 커질수록 피보나치 수열의 연속된 두 항의 비율이 황금비에 수렴한다는 특징 이용
    // ================================================================================
    template <size_t N, typename Func>
    static double golden_section_search(Func f, const std::array<double, N>& x,
                                        const std::array<double, N>& d, double a, double b,
                                        double tol = 1e-5, bool verbose = false) {
        const double phi = (3.0 - std::sqrt(5.0)) / 2.0;  // 황금비의 역수
        double d_step = b - a;
        double c = a + phi * d_step;
        double d_val = b - phi * d_step;

        double yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));
        double yd = AutoDiff::value<N>(f, ray_point<N>(x, d, d_val));

        size_t iter = 0;
        while (std::abs(d_step) > tol) {
            iter++;
            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] Range: [" << std::fixed
                          << std::setprecision(6) << a << ", " << b
                          << "] | Gap: " << std::abs(b - a) << "\n";
            }

            if (yc < yd) {
                b = d_val;
                d_val = c;
                yd = yc;
                d_step = b - a;
                c = a + phi * d_step;
                yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));
            } else {
                a = c;
                c = d_val;
                yc = yd;
                d_step = b - a;
                d_val = b - phi * d_step;
                yd = AutoDiff::value<N>(f, ray_point<N>(x, d, d_val));
            }
        }
        if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
        return (a + b) / 2.0;
    }

    // ======================================
    // 3. Quadratic Fit Search (Algorithm 3.4)
    // 핵심 아이디어 : 최솟값 근처는 포물선 모양이다
    // 국소 최솟값 부근으로 충분히 확대해 들어가면 대부분의 부드러운 목적 함수는 2차 함수 (Quadratic
    // function), 즉 U자 형태의 그릇 모양의 띄게 됩니다. 이 점에 착안아혀, 현재 파악한 세 개의 점을
    // 지나는 2차 함수 곡선을 가상으로 그리고, 그 가상의 2차 함수의 최솟값을 다음 탐색 지점으로 바로
    // 점프하는 방법
    //
    // 수학적 원리
    // - 최솟값을 감싸고 있는 세 개의 브레킷 지점 a < b < c와 그 지점들에 대한 함수값 ya, yb, yc
    // - 이 세 점을 완벽하게 지나는 2차 함수 q(x) = p1 + p2 * x + p3 * x^2 형태의 행렬 역연산 등을
    // 통해 찾아냅니다
    // - 2차 함수의 최솟값은 미분값이 0이 되는 지점이므로, 수식을 풀면 해석적으로 유일한 최솟값
    // x^{*}의 위치를 계산 가능
    // ======================================
    template <size_t N, typename Func>
    static double quadratic_fit_search(Func f, const std::array<double, N>& x,
                                       const std::array<double, N>& d, double a, double b, double c,
                                       size_t max_iter = 50, double tol = 1e-5,
                                       bool verbose = false) {
        auto eval = [&](double alpha) { return AutoDiff::value<N>(f, ray_point<N>(x, d, alpha)); };

        double ya = eval(a), yb = eval(b), yc = eval(c);

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            double num = (b - a) * (b - a) * (yb - yc) - (b - c) * (b - c) * (yb - ya);
            double den = (b - a) * (yb - yc) - (b - c) * (yb - ya);
            if (std::abs(den) < 1e-16) {
                break;
            }
            double x_star = b - 0.5 * num / den;
            double y_star = eval(x_star);

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] x*: " << std::fixed
                          << std::setprecision(6) << x_star << " | f(x*): " << y_star
                          << " | Shift: " << std::abs(x_star - b) << "\n";
            }

            if (std::abs(x_star - b) < tol) {
                if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
                return x_star;
            }

            if (x_star > b) {
                if (y_star > yb) {
                    c = x_star;
                    yc = y_star;
                } else {
                    a = b;
                    ya = yb;
                    b = x_star;
                    yb = y_star;
                }
            } else {
                if (y_star > yb) {
                    a = x_star;
                    ya = y_star;
                } else {
                    c = b;
                    yc = yb;
                    b = x_star;
                    yb = y_star;
                }
            }
        }
        return b;
    }

    // ==================================================================
    // 4. Shubert-Piyavskii Algorithm (Algorithm 3.5)
    // 특이점 : 구간 내 골짜기가 몇 개가 있든 상관없이 반드시 진짜 바닥인 '전역 최솟값'을 확정적으로
    // 찾아내는 전역 최적화 알고리즘
    // 1. 전제 조건 : 립시츠 상수 - 함수의 기울기 (가파른 정도)가 아무리 커져도 절대 넘을 수 없는
    // 최대 한계치인 lIPSCHITZ CONSTANT를 미리 알고 있어야 함
    // 2. 핵심 원리 : 톱니바퀴 모양의 하한선 만들기
    // ==================================================================
    template <size_t N, size_t MAX_NODES = 100, typename Func>
    static double shubert_piyavskii(Func f, const std::array<double, N>& x,
                                    const std::array<double, N>& d, double a, double b, double L,
                                    double tol = 1e-4, bool verbose = false) {
        auto eval = [&](double alpha) { return AutoDiff::value<N>(f, ray_point<N>(x, d, alpha)); };
        struct SPNode {
            double a, ya, b, yb, x_m, y_m;
            bool active;
        };
        std::array<SPNode, MAX_NODES> pool;
        size_t pool_size = 0;
        double ya = eval(a), yb = eval(b);
        double min_val = std::min(ya, yb);
        double min_alpha = (ya < yb) ? a : b;

        auto add_node = [&](double n_a, double n_ya, double n_b, double n_yb) {
            if (pool_size >= MAX_NODES) {
                return;
            }
            double x_m = 0.5 * (n_a + n_b) + (n_ya - n_yb) / (2.0 * L);
            double y_m = 0.5 * (n_ya + n_yb) - L * (n_b - n_a) / 2.0;
            pool[pool_size++] = {n_a, n_ya, n_b, n_yb, x_m, y_m, true};
        };
        add_node(a, ya, b, yb);

        size_t iter = 0;
        for (iter = 1; iter <= MAX_NODES / 2; ++iter) {
            double best_ym = 1e99;
            size_t best_idx = 0;
            bool found = false;
            for (size_t i = 0; i < pool_size; ++i) {
                if (pool[i].active && pool[i].y_m < best_ym) {
                    best_ym = pool[i].y_m;
                    best_idx = i;
                    found = true;
                }
            }

            if (!found) break;
            double x_m = pool[best_idx].x_m;
            double y_eval = eval(x_m);

            if (y_eval < min_val) {
                min_val = y_eval;
                min_alpha = x_m;
            }

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] Probe x_m: " << std::fixed
                          << std::setprecision(5) << x_m
                          << " | LowerBound y_m: " << pool[best_idx].y_m
                          << " | Gap: " << std::abs(y_eval - pool[best_idx].y_m) << "\n";
            }

            if (std::abs(y_eval - pool[best_idx].y_m) < tol) {
                break;
            }
            pool[best_idx].active = false;
            add_node(pool[best_idx].a, pool[best_idx].ya, x_m, y_eval);
            add_node(x_m, y_eval, pool[best_idx].b, pool[best_idx].yb);
        }
        if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
        return min_alpha;
    }

    // ==================================================================
    // 5. Bracket Sign Change (Algorithm 3.6)
    // 방향 도함수 (Derivative)의 부호가 바뀌는 구간을 찾습니다 (이분법의 전제 조건)
    // ==================================================================
    template <size_t N, typename Func>
    static std::pair<double, double> bracket_sign_change(Func f, const std::array<double, N>& x,
                                                         const std::array<double, N>& d, double a,
                                                         double b, double k = 2.0,
                                                         bool verbose = false) {
        if (a > b) {
            std::swap(a, b);
        }
        double center = (a + b) / 2.0;
        double half_width = (b - a) / 2.0;
        auto eval_deriv = [&](double alpha) { return directional_derivative<N>(f, x, d, alpha); };
        double fp_a = eval_deriv(a);
        double fp_b = eval_deriv(b);

        size_t iter = 0;
        while (fp_a * fp_b > 0.0) {
            iter++;
            half_width *= k;
            a = center - half_width;
            b = center + half_width;
            fp_a = eval_deriv(a);
            fp_b = eval_deriv(b);

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] Expanding: [" << std::fixed
                          << std::setprecision(5) << a << ", " << b << "] | f'(a)=" << std::showpos
                          << fp_a << " | f'(b)=" << fp_b << std::noshowpos << "\n";
            }
        }
        return std::make_pair(a, b);
    }
    // ======================================
    // 6. Bisection Method (Algorithm 3.7)
    // ======================================
    template <size_t N, typename Func>
    static double bisection(Func f, const std::array<double, N>& x, const std::array<double, N>& d,
                            double a, double b, double tol = 1e-5, bool verbose = false) {
        if (a > b) std::swap(a, b);
        auto eval_deriv = [&](double alpha) { return directional_derivative<N>(f, x, d, alpha); };
        double ya = eval_deriv(a);
        double yb = eval_deriv(b);

        if (std::abs(ya) <= 1e-9) return a;
        if (std::abs(yb) <= 1e-9) return b;

        size_t iter = 0;
        while ((b - a) > tol) {
            iter++;
            double mid = (a + b) / 2.0;
            double y_mid = eval_deriv(mid);

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] Root Search: ["
                          << std::fixed << std::setprecision(6) << a << ", " << b
                          << "] | Mid: " << mid << " | f'(mid): " << std::showpos << y_mid
                          << std::noshowpos << "\n";
            }

            if (std::abs(y_mid) <= 1e-9) {
                if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
                return mid;
            }

            // 도함수의 부호가 같은 쪽의 구간을 축소
            if ((y_mid > 0.0 && ya > 0.0) || (y_mid < 0.0 && ya < 0.0)) {
                a = mid;
                ya = y_mid;
            } else {
                b = mid;
                yb = y_mid;
            }
        }
        if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
        return (a + b) / 2.0;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_LINE_SEARCH_HPP_