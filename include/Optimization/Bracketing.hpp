#ifndef OPTIMIZATION_BRACKETING_HPP_
#define OPTIMIZATION_BRACKETING_HPP_

#include <cmath>
#include <vector>
#include <algorithm>                // std::swap, std::min, std::max
#include <limits>                   // std::numeric_limits
#include "Optimization/Dual.hpp"

namespace Optimization {
    /**
     * @brief 탐색 범위를 나타내는 구조체
     */
    struct Range {
        double a;
        double b;
    };

    /**
     * @brief Algorithm 3.7 Bracket Minimum (최솟값이 포함된 구간 확보)
     * @param f 평가 함수 (Scalar)
     * @param x 시작 지점
     * @param s 초기 스텝 크기
     * @param k 확장 배율
     */
    template <typename Func>
    inline Range bracket_minimum(Func f, double x, double s = 0.01, double k = 2.0) {
        double a = x;
        double ya = f(a);
        double b = a + s;
        double yb = f(b);

        // 하강 방향이 아니면 방향 전환
        if (yb > ya) {
            std::swap(a, b);
            std::swap(ya, yb);
            s = -s;
        }

        while (true) {
            double c = b + s;
            double yc = f(c);

            // 함수값이 다시 커지는 지점을 발견하면 구간 변환
            if (yc > yb) {
                return {std::min(a, c), std::max(a, c)};
            }
            a = b;
            ya = yb;
            b = c;
            yb = yc;
            s *= k;     // 스텝 확장
        }
    }

    /**
     * @brief Algorithm 3.3 : Golden Section Search (황금 분할 탐색)
     * 미분을 사용하지 않는 0차 최적화 알고리즘
     */
    template <typename Func>
    inline double golden_section_search(Func f, Range r, int n) {
        const double phi = (std::sqrt(5.0) - 1.0) / 2.0;    // 황금비 (~0.618)
        double a = r.a;
        double b = r.b;

        double x1 = b - phi * (b - a);
        double x2 = a + phi * (b - a);
        double y1 = f(x1);
        double y2 = f(x2);

        for (int i = 0; i < n; ++i) {
            if (y1 < y2) {
                b = x2;
                x2 = x1;
                y2 = y1;
                x1 = b - phi * (b - a);
                y1 = f(x1);
            } else {
                a = x1;
                x1 = x2;
                y1 = y2;
                x2 = a + phi * (b - a);
                y2 = f(x2);
            }
        }
        return (a + b) / 2.0;
    }

    /**
     * @brief Algorithm 3.6 : Bisection Method (이분법)
     * Dual.hpp의 미분 정보 f'(x)를 활용하여 최적점 (f'(x) = 0) 추적
     */
    template <typename FuncDual>
    inline double bisection_method(FuncDual f, Range range, int n) {
        double a = range.a;
        double b = range.b;

        for (int i = 0; i < n; ++i) {
            double mid = (a + b) / 2.0;

            // 1.0을 시딩 (Seeding)하여 mid 지점에서 1차 미분값 획득
            auto res = f(Dual<double>(mid, 1.0));

            // 기울기가 양수 (+)면 오른쪽이 높으므로 왼쪽 구간 (a ~ mid) 선택
            // 기울기가 음수 (-)면 왼쪽이 높으므로 오른쪽 구간 (mid ~ b) 선택
            if (res.d > 0) {
                b = mid;
            } else {
                a = mid;
            }
        }
        return (a + b) / 2.0;
    }

    /**
     * @brief Algorithm 3.5 : Shubert-Piyavskii (전역 최적화 알고리즘)
     * 립시츠 상수 (Lipschitz Constant)를 활용하여 골짜기를 톱날 형태로 채워나감
     */
    struct Pt {
        double x;
        double y;
    };

    template <typename Func>
    inline double shubert_piyavskii(Func f, Range r, double l, double eps, int max_iter) {
        // std::vector 오류 수정 및 리스트 초기화 문법 준수
        std::vector<Pt> pts = {{r.a, f(r.a)}, {r.b, f(r.b)}};

        double min_f = std::min(pts[0].y, pts[1].y);

        for (int i = 0; i < max_iter; ++i) {
            int best_idx = 0;
            double best_z = std::numeric_limits<double>::infinity();

            // 가장 깊게 파인 톱날 (하한선) 탐색
            for (size_t j = 0; j < pts.size() - 1; ++j) {
                double z = (pts[j].y + pts[j + 1].y) / 2.0 - l * (pts[j + 1].x - pts[j].x) / 2.0;
                if (z < best_z) {
                    best_z = z;
                    best_idx = static_cast<int>(j);
                }
            }

            // 새로운 탐색 지점 계산 및 평가
            double new_x = (pts[best_idx].x + pts[best_idx + 1].x) / 2.0 - (pts[best_idx + 1].y - pts[best_idx].y) / (2.0 * l);
            double new_y = f(new_x);

            min_f = std::min(min_f, new_y);

            // 새로운 점 삽입 (x좌표 기분 정렬 유지)
            pts.insert(pts.begin() + best_idx + 1, {new_x, new_y});

            // 수렴 조건 판단
            if (std::abs(min_f - best_z) < eps) 
                break;
        }
        return min_f;
    }
} // namespace Optimization

#endif // OPTIMIZATION_BRACKETING_HPP_