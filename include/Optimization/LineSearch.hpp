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

/**
 * @class LineSearch
 * @brief 라인 서치(Line Search) 알고리즘들을 모아놓은 정적 클래스
 * 
 * 최적화 문제에서 주어진 탐색 방향(d)을 따라 이동할 때, 
 * 목적 함수 f(x + alpha * d)를 최소화하는 적절한 스텝 사이즈(alpha)를 찾는 알고리즘들을 제공합니다.
 * 최솟값을 감싸는 구간을 찾는 브래킷(Bracket) 기법부터, 해당 구간 내에서 정밀하게
 * 최솟값을 찾는 황금분할탐색(Golden Section Search), 이분법(Bisection), 2차 적합(Quadratic Fit) 등의 기법을 포함합니다.
 */
class LineSearch {
   private:
    /**
     * @brief 기준점 x에서 방향 d로 alpha만큼 이동한 지점을 계산합니다.
     * @param x 현재 위치 (N차원 벡터)
     * @param d 탐색 방향 (N차원 벡터)
     * @param alpha 이동할 거리 (스텝 사이즈)
     * @return x + alpha * d 위치의 새로운 벡터
     */
    template <size_t N>
    static StaticVector<double, N> ray_point(const StaticVector<double, N>& x,
                                             const StaticVector<double, N>& d, double alpha) {
        StaticVector<double, N> pt;
        for (size_t i = 0; i < N; ++i) {
            pt(static_cast<int>(i)) = x(static_cast<int>(i)) + alpha * d(static_cast<int>(i));
        }
        return pt;
    }

    /**
     * @brief 특정 지점(x + alpha * d)에서의 목적 함수의 방향 도함수(Directional Derivative)를 계산합니다.
     * @details 방향 도함수는 함수의 기울기(Gradient)와 탐색 방향(d)의 내적(Dot Product)으로 정의됩니다.
     *          AutoDiff를 이용해 기울기를 구한 후 방향 벡터와의 내적을 계산합니다.
     * @param f 목적 함수
     * @param x 현재 위치
     * @param d 탐색 방향
     * @param alpha 현재 스텝 사이즈
     * @return alpha 위치에서의 방향 도함수 값 (스칼라)
     */
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
    // 함수가 감소하다가 다시 증가한 것이므로, 구간 [a, c] 내부에 반드시 최솟값이 존재합니다.
    // 이 알고리즘은 초기 스텝 s에서 시작해 조건을 만족할 때까지 스텝을 k배씩 늘려가며 탐색합니다.
    // ==================================================================
    /**
     * @brief 최솟값을 포함하는 구간 [a, c]를 찾는 브래킷 알고리즘
     * @param f 목적 함수
     * @param x 초기 위치
     * @param d 탐색 방향
     * @param s 초기 스텝 크기 (기본값: 0.01)
     * @param k 스텝 확장 배수 (기본값: 2.0)
     * @param verbose 진행 과정 출력 여부
     * @return 최솟값을 포함하는 구간 (a, c) (a < c 보장)
     */
    template <size_t N, typename Func>
    static std::pair<double, double> bracket_minimum(Func f, const StaticVector<double, N>& x,
                                                     const StaticVector<double, N>& d,
                                                     double s = 1e-2, double k = 2.0,
                                                     bool verbose = false) {
        double a = 0.0;
        double ya = AutoDiff::value<N>(f, ray_point(x, d, a));
        double b = s;
        double yb = AutoDiff::value<N>(f, ray_point<N>(x, d, b));

        // 탐색 방향의 반대쪽으로 이동해야 함수값이 감소하는 경우
        // 방향을 반대로 뒤집어서 탐색을 시작합니다.
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

            // f(c) > f(b)를 만족하면 f(a) > f(b) < f(c) 조건이 완성되어
            // 구간 내에 지역 최솟값이 존재함이 보장됩니다.
            if (yc > yb) {
                return (a < c) ? std::make_pair(a, c) : std::make_pair(c, a);
            }

            // 아직 감소 추세라면 범위를 다음으로 이동하고 스텝(s)을 늘립니다.
            a = b;
            ya = yb;
            b = c;
            yb = yc;
            s *= k; // 스텝 크기를 k배 증가 (가속 탐색)
        }
    }

    // ================================================================================
    // 2. Golden Section Search (Algorithm 3.3)
    // 최솟값이 포함된 구간 [a, b]가 주어졌을 때, 황금비(약 0.618)를 이용하여
    // 평가 횟수를 최소화하면서 구간을 지속적으로 축소해 나가는 탐색 기법입니다.
    // 기존 구간을 황금비로 나누는 두 점 c, d 중 하나는 다음 스텝에서도 재사용되므로 연산 효율이 좋습니다.
    // ================================================================================
    /**
     * @brief 황금 분할 탐색법을 이용한 라인 서치
     * @param f 목적 함수
     * @param x 기준 위치
     * @param d 탐색 방향
     * @param a 탐색 구간 시작점
     * @param b 탐색 구간 끝점
     * @param tol 허용 오차 (구간의 길이가 이 값보다 작아지면 탐색 종료)
     * @param verbose 진행 과정 출력 여부
     * @return 추정된 최적의 스텝 사이즈 (alpha)
     */
    template <size_t N, typename Func>
    static double golden_section_search(Func f, const StaticVector<double, N>& x,
                                        const StaticVector<double, N>& d, double a, double b,
                                        double tol = 1e-5, bool verbose = false) {
        const double phi = (3.0 - std::sqrt(5.0)) / 2.0; // 약 0.381966 (1 - 0.618034)
        double d_step = b - a;
        double c = a + phi * d_step;
        double d_val = b - phi * d_step;

        double yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));
        double yd = AutoDiff::value<N>(f, ray_point<N>(x, d, d_val));

        size_t iter = 0;
        // 구간의 길이(d_step)가 tol 이하로 좁혀질 때까지 반복
        while (std::abs(d_step) > tol) {
            iter++;
            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] Range: [" << std::fixed
                          << std::setprecision(6) << a << ", " << b
                          << "] | Gap: " << std::abs(b - a) << "\n";
            }

            // f(c) < f(d) 이면 최솟값은 [a, d_val] 사이에 존재
            if (yc < yd) {
                b = d_val;     // 구간의 오른쪽 끝을 d_val로 축소
                d_val = c;     // 이전 c가 새로운 d_val이 됨 (함수값 재사용)
                yd = yc;
                d_step = b - a;
                c = a + phi * d_step; // 새로운 c점만 평가
                yc = AutoDiff::value<N>(f, ray_point<N>(x, d, c));
            } 
            // f(c) >= f(d) 이면 최솟값은 [c, b] 사이에 존재
            else {
                a = c;         // 구간의 왼쪽 끝을 c로 축소
                c = d_val;     // 이전 d_val이 새로운 c가 됨 (함수값 재사용)
                yc = yd;
                d_step = b - a;
                d_val = b - phi * d_step; // 새로운 d_val점만 평가
                yd = AutoDiff::value<N>(f, ray_point<N>(x, d, d_val));
            }
        }
        if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
        // 좁혀진 구간의 중앙값을 반환
        return (a + b) / 2.0;
    }

    // ======================================
    // 3. Quadratic Fit Search (Algorithm 3.4)
    // 세 점 (a, f(a)), (b, f(b)), (c, f(c))를 지나는 2차 다항식(포물선)을 근사하여,
    // 그 포물선의 최솟값을 가지는 꼭짓점 위치(x*)를 다음 탐색점으로 삼는 방법입니다.
    // 함수가 최솟값 근처에서 2차 함수와 비슷한 형태를 띠는 경우 수렴 속도가 매우 빠릅니다.
    // ======================================
    /**
     * @brief 2차 적합 탐색(Quadratic Fit Search)을 통한 라인 서치
     * @param f 목적 함수
     * @param x 기준 위치
     * @param d 탐색 방향
     * @param a 탐색점 1
     * @param b 탐색점 2 (a < b < c 또는 a > b > c이며 가운데 있는 점)
     * @param c 탐색점 3
     * @param max_iter 최대 반복 횟수
     * @param tol 허용 오차 (위치 변화량이 이 값보다 작아지면 수렴한 것으로 간주)
     * @param verbose 진행 과정 출력 여부
     * @return 추정된 최적의 스텝 사이즈 (alpha)
     */
    template <size_t N, typename Func>
    static double quadratic_fit_search(Func f, const StaticVector<double, N>& x,
                                       const StaticVector<double, N>& d, double a, double b,
                                       double c, size_t max_iter = 50, double tol = 1e-5,
                                       bool verbose = false) {
        auto eval = [&](double alpha) { return AutoDiff::value<N>(f, ray_point<N>(x, d, alpha)); };

        double ya = eval(a), yb = eval(b), yc = eval(c);

        for (size_t iter = 1; iter <= max_iter; ++iter) {
            // 세 점을 지나는 2차 다항식의 꼭짓점 계산을 위한 분자와 분모
            double num = (b - a) * (b - a) * (yb - yc) - (b - c) * (b - c) * (yb - ya);
            double den = (b - a) * (yb - yc) - (b - c) * (yb - ya);
            
            // 세 점이 거의 일직선상에 있어 2차 근사가 불가능한 경우 루프 탈출
            if (std::abs(den) < 1e-16) {
                break;
            }
            
            // 2차 근사 다항식의 최소점 위치 (x_star)
            double x_star = b - 0.5 * num / den;
            double y_star = eval(x_star);

            if (verbose) {
                std::cout << "  ↳ [Iter " << std::setw(2) << iter << "] x*: " << std::fixed
                          << std::setprecision(6) << x_star << " | f(x*): " << y_star
                          << " | Shift: " << std::abs(x_star - b) << "\n";
            }

            // 이동한 거리가 허용 오차보다 작으면 수렴한 것으로 판단
            if (std::abs(x_star - b) < tol) {
                if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
                return x_star;
            }

            // 새로 찾은 x_star와 b의 위치 관계, 그리고 각각의 함수값을 비교하여
            // 다음 반복에서 사용할 a, b, c 세 점을 갱신합니다.
            // 항상 b가 세 점 중 가운데 위치를 유지하도록 조정합니다.
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
    // 립시츠 연속(Lipschitz Continuous) 함수에서, 립시츠 상수(L)를 알고 있을 때
    // 함수의 전역 최솟값(Global Minimum)을 찾기 위해 사용되는 알고리즘입니다.
    // 기존에 평가된 점들로부터 V자 형태의 하한선(Lower Bound)을 생성하여, 
    // 하한선의 교차점 중 가장 낮은 지점을 다음 탐색점으로 선택합니다.
    // ==================================================================
    /**
     * @brief 슈베르트-피야프스키 전역 탐색 알고리즘
     * @param f 목적 함수
     * @param x 기준 위치
     * @param d 탐색 방향
     * @param a 탐색 구간 시작점
     * @param b 탐색 구간 끝점
     * @param L 립시츠 상수 (Lipschitz Constant, 함수 기울기의 최대 절댓값)
     * @param tol 허용 오차 (예측 하한값과 실제 함수값의 차이가 tol 이하면 수렴)
     * @param verbose 진행 과정 출력 여부
     * @return 추정된 전역 최솟값을 가지는 스텝 사이즈 (alpha)
     */
    template <size_t N, size_t MAX_NODES = 100, typename Func>
    static double shubert_piyavskii(Func f, const StaticVector<double, N>& x,
                                    const StaticVector<double, N>& d, double a, double b, double L,
                                    double tol = 1e-4, bool verbose = false) {
        auto eval = [&](double alpha) { return AutoDiff::value<N>(f, ray_point<N>(x, d, alpha)); };
        
        // 탐색 구간의 하한선 교차점을 저장하는 구조체
        struct SPNode {
            double a, ya, b, yb, x_m, y_m;
            bool active;
        };
        std::array<SPNode, MAX_NODES> pool;
        size_t pool_size = 0;
        
        double ya = eval(a), yb = eval(b);
        double min_val = std::min(ya, yb);
        double min_alpha = (ya < yb) ? a : b;

        // 새로운 구간 [n_a, n_b]에 대한 하한선 교차점(x_m, y_m)을 계산하여 노드 풀에 추가
        auto add_node = [&](double n_a, double n_ya, double n_b, double n_yb) {
            if (pool_size >= MAX_NODES) {
                return;
            }
            // 두 점을 지나는 기울기 ±L인 직선들의 교차점 x 좌표
            double x_m = 0.5 * (n_a + n_b) + (n_ya - n_yb) / (2.0 * L);
            // 교차점에서의 y 좌표 (하한값)
            double y_m = 0.5 * (n_ya + n_yb) - L * (n_b - n_a) / 2.0;
            pool[pool_size++] = {n_a, n_ya, n_b, n_yb, x_m, y_m, true};
        };
        
        add_node(a, ya, b, yb);

        size_t iter = 0;
        // 최대 반복 횟수는 노드 공간을 초과하지 않도록 MAX_NODES / 2 로 제한
        for (iter = 1; iter <= MAX_NODES / 2; ++iter) {
            double best_ym = 1e99;
            size_t best_idx = 0;
            bool found = false;
            
            // 현재 활성화된 구간들 중 가장 낮은 하한값(y_m)을 가진 구간을 선택
            for (size_t i = 0; i < pool_size; ++i) {
                if (pool[i].active && pool[i].y_m < best_ym) {
                    best_ym = pool[i].y_m;
                    best_idx = i;
                    found = true;
                }
            }

            if (!found) break; // 더 이상 탐색할 활성 구간이 없음
            
            double x_m = pool[best_idx].x_m;
            double y_eval = eval(x_m); // 교차점 위치에서의 실제 함수값 평가

            // 현재까지 발견된 최소 함수값 갱신
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

            // 실제 함수값(y_eval)과 예측 하한값(y_m)의 차이가 tol 이하라면 수렴한 것으로 간주
            if (std::abs(y_eval - pool[best_idx].y_m) < tol) {
                break;
            }
            
            // 탐색한 구간을 비활성화하고, x_m을 기준으로 두 개의 새로운 하위 구간을 생성
            pool[best_idx].active = false;
            add_node(pool[best_idx].a, pool[best_idx].ya, x_m, y_eval);
            add_node(x_m, y_eval, pool[best_idx].b, pool[best_idx].yb);
        }
        if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
        return min_alpha;
    }

    // ==================================================================
    // 5. Bracket Sign Change (Algorithm 3.6)
    // 함수의 도함수(Derivative)가 연속이라고 가정할 때, 방향 도함수의 부호가 
    // 음수에서 양수로 바뀌는 구간을 찾는 알고리즘입니다. 도함수의 부호가 바뀌는 
    // 구간 내부에는 도함수가 0이 되는 지점(즉, 최솟값 후보인 임계점)이 존재하게 됩니다.
    // ==================================================================
    /**
     * @brief 도함수의 부호가 바뀌는 구간을 찾는 브래킷 알고리즘
     * @param f 목적 함수
     * @param x 기준 위치
     * @param d 탐색 방향
     * @param a 초기 탐색 구간의 한 점
     * @param b 초기 탐색 구간의 다른 점
     * @param k 구간 확장 배수 (기본값: 2.0)
     * @param verbose 진행 과정 출력 여부
     * @return 도함수의 부호가 바뀌는 구간 (a, b)
     */
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
        // 양 끝점의 도함수 부호가 같으면 (곱이 양수) 최솟값을 포함하지 않으므로 구간을 확장
        while (fp_a * fp_b > 0.0) {
            iter++;
            half_width *= k; // 구간을 k배 만큼 중심을 기준으로 확장
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
    // 도함수의 부호가 다른 두 점 a, b (f'(a) < 0, f'(b) > 0)가 주어졌을 때,
    // 구간을 반씩 줄여가며 도함수가 0이 되는 지점(근)을 찾는 이분법 기반 라인 서치입니다.
    // Bracket Sign Change 알고리즘과 연계하여 사용하기 좋습니다.
    // ======================================
    /**
     * @brief 방향 도함수의 근(최솟값 위치)을 찾는 이분법 탐색
     * @param f 목적 함수
     * @param x 기준 위치
     * @param d 탐색 방향
     * @param a 탐색 구간 시작점
     * @param b 탐색 구간 끝점 (f'(a)와 f'(b)의 부호가 달라야 함)
     * @param tol 허용 오차 (구간의 길이가 이 값 이하가 되면 수렴)
     * @param verbose 진행 과정 출력 여부
     * @return 추정된 최적의 스텝 사이즈 (alpha)
     */
    template <size_t N, typename Func>
    static double bisection(Func f, const StaticVector<double, N>& x,
                            const StaticVector<double, N>& d, double a, double b, double tol = 1e-5,
                            bool verbose = false) {
        if (a > b) std::swap(a, b); // a가 항상 작도록 정렬
        auto eval_deriv = [&](double alpha) { return directional_derivative<N>(f, x, d, alpha); };
        double ya = eval_deriv(a);
        double yb = eval_deriv(b);

        // 이미 양 끝점 중 하나가 최솟점(도함수가 0에 매우 가까움)인 경우 바로 반환
        if (std::abs(ya) <= 1e-9) return a;
        if (std::abs(yb) <= 1e-9) return b;

        size_t iter = 0;
        // 탐색 구간 크기가 허용 오차보다 클 때까지 반복
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

            // 중간점의 도함수 값이 0에 매우 가까우면 찾은 것으로 판단하고 종료
            if (std::abs(y_mid) <= 1e-9) {
                if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
                return mid;
            }

            // 중간점에서의 도함수 부호에 따라 탐색 구간 축소
            // ya와 y_mid의 부호가 같으면(둘 다 양수거나 둘 다 음수), 근은 mid와 b 사이에 존재
            if ((y_mid > 0.0 && ya > 0.0) || (y_mid < 0.0 && ya < 0.0)) {
                a = mid;
                ya = y_mid;
            } 
            // 다르면 근은 a와 mid 사이에 존재
            else {
                b = mid;
                yb = y_mid;
            }
        }
        if (verbose) std::cout << "  ↳ [Converged] Total Iterations: " << iter << "\n";
        // 허용 오차 이내로 구간이 좁혀지면, 해당 구간의 중앙값을 결과로 반환
        return (a + b) / 2.0;
    }
};
}  // namespace Optimization

#endif  // OPTIMIZATION_LINE_SEARCH_HPP_