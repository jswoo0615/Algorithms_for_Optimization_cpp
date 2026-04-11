#ifndef OPTIMIZATION_LINE_SEARCH_HPP_
#define OPTIMIZATION_LINE_SEARCH_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

#include "Optimization/AutoDiff.hpp"
#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {
class LineSearch {
   private:
    template <size_t N>
    static StaticVector<double, N> ray_point(const StaticVector<double, N>& x,
                                             const StaticVector<double, N>& d, double alpha) {
        StaticVector<double, N> pt;
        for (size_t i = 0; i < N; ++i) {
            pt(static_cast<int>(i)) = x(static_cast<int>(i)) + alpha * d(static_cast<int>(i));
        }
        return pt;
    }

    template <size_t N, typename Func>
    static double directional_derivative(Func f, const StaticVector<double, N>& x,
                                         const StaticVector<double, N>& d, double alpha) {
        auto pt = ray_point<N>(x, d, alpha);
        auto grad = AutoDiff::gradient<N>(f, pt);
        double dir_deriv = 0.0;
        for (size_t i = 0; i < N; ++i) {
            dir_deriv += grad(static_cast<int>(i)) * d(static_cast<int>(i));
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
    static std::pair<double, double> bracket_minimum(Func f, const StaticVector<double, N>& x,
                                                     const StaticVector<double, N>& d,
                                                     double s = 1e-2, double k = 2.0,
                                                     bool verbose = false) {
        double a = 0.0;
        double ya = AutoDiff::value<N>(f, ray_point(x, d, a));
        double b = s;
        double yb = AutoDiff::value<N>(f, ray_point<N>(x, d, b));

        if (yb > ya) {
            std::swap(a, b);
            std::swap(ya, yb);
            s = -s;
        }

        size_t iter = 0;
        while (true) {
            iter++;
            double c = b + s;
            double yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] a: " << std::fixed
                          << std::setprecision(5) << a << " | b: " << b << " | c: " << c
                          << " | f(c): " << yc << "\n";
            }

            if (yc > yb) {
                return (a < c) ? std::make_pair(a, c) : std::make_pair(c, a);
            }

            a = b;
            ya = yb;
            b = c;
            yb = yc;
            s *= k;
        }
    }

    // ================================================================================
    // 2. Golden Section Search (Algorithm 3.3)
    // ================================================================================
    template <size_t N, typename Func>
    static double golden_section_search(Func f, const StaticVector<double, N>& x,
                                        const StaticVector<double, N>& d, double a, double b,
                                        double tol = 1e-5, bool verbose = false) {
        const double phi = (3.0 - std::sqrt(5.0)) / 2.0;
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
    // ======================================
    template <size_t N, typename Func>
    static double quadratic_fit_search(Func f, const StaticVector<double, N>& x,
                                       const StaticVector<double, N>& d, double a, double b,
                                       double c, size_t max_iter = 50, double tol = 1e-5,
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
    // ==================================================================
    template <size_t N, size_t MAX_NODES = 100, typename Func>
    static double shubert_piyavskii(Func f, const StaticVector<double, N>& x,
                                    const StaticVector<double, N>& d, double a, double b, double L,
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
    // ==================================================================
    template <size_t N, typename Func>
    static std::pair<double, double> bracket_sign_change(Func f, const StaticVector<double, N>& x,
                                                         const StaticVector<double, N>& d, double a,
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
    static double bisection(Func f, const StaticVector<double, N>& x,
                            const StaticVector<double, N>& d, double a, double b, double tol = 1e-5,
                            bool verbose = false) {
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