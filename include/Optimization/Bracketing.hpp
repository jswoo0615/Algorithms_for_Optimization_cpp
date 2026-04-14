#ifndef OPTIMIZATION_BRACKETING_HPP_
#define OPTIMIZATION_BRACKETING_HPP_

#include <algorithm>  // std::swap, std::min, std::max
#include <cmath>
#include <limits>  // std::numeric_limits
#include <vector>

#include "Optimization/Dual.hpp"

namespace Optimization {

/**
 * @brief 탐색 범위를 나타내는 구조체 (Bracketing Range)
 * 
 * 1차원 최적화(Line Search) 등에서 극솟값(Local Minimum)이 반드시 존재한다고 보장되는 
 * 구간의 양 끝점(시작점 a와 끝점 b)을 저장하기 위해 사용됩니다.
 */
struct Range {
    double a;
    double b;
};

/**
 * @brief Algorithm 3.7: Bracket Minimum (최솟값이 포함된 구간 탐색)
 * 
 * 주어진 시작 지점(x)에서 출발하여 함수값이 감소하는 방향으로 스텝(s)을 밟아가며,
 * 함수값이 다시 증가하는 지점을 찾을 때까지 스텝의 크기를 점진적으로 확장(k 배율)합니다.
 * 이를 통해 극솟값이 확실히 포함된 세 점(a, b, c)을 찾아내고, 최종적으로 그 구간(Range)을 반환합니다.
 * 
 * @tparam Func 평가 함수 타입 (Scalar 1차원 함수)
 * @param f 최솟값을 찾고자 하는 1차원 목적 함수
 * @param x 탐색을 시작할 초기 지점
 * @param s 초기 탐색 스텝 크기 (기본값: 0.01). 이 값은 방향에 따라 음수로 변환될 수 있습니다.
 * @param k 스텝 확장 배율 (기본값: 2.0). 한 번 탐색할 때마다 다음 스텝의 크기를 k배만큼 늘려 빠른 구간 탐색을 유도합니다.
 * @return Range 극솟값이 포함되어 있음이 보장되는 탐색 구간 [min(a, c), max(a, c)]
 */
template <typename Func>
inline Range bracket_minimum(Func f, double x, double s = 0.01, double k = 2.0) {
    double a = x;
    double ya = f(a);
    double b = a + s;
    double yb = f(b);

    // [Step 1] 탐색 방향 결정
    // 현재 스텝 방향(오른쪽)으로 이동했을 때 함수값이 커진다면(yb > ya), 
    // 이는 하강 방향(Descent Direction)이 아니라는 의미입니다.
    // 따라서 탐색 방향을 반대(왼쪽)로 바꾸기 위해 a와 b를 스왑하고 스텝(s)의 부호를 뒤집습니다.
    if (yb > ya) {
        std::swap(a, b);
        std::swap(ya, yb);
        s = -s;
    }

    // [Step 2] 하강 방향을 따라 구간 확장
    while (true) {
        // 이전 지점(b)에서 확장된 스텝(s)만큼 더 이동한 새로운 지점(c)를 평가합니다.
        double c = b + s;
        double yc = f(c);

        // 함수값이 이전 지점(yb)보다 커진다면(yc > yb), 
        // 하강하던 추세가 꺾이고 다시 상승한다는 의미이므로 구간(a, c) 사이에 극솟값이 존재하게 됩니다.
        if (yc > yb) {
            // 구간의 양 끝점이 항상 (작은 값, 큰 값) 순서가 되도록 정렬하여 반환합니다.
            return {std::min(a, c), std::max(a, c)};
        }
        
        // 탐색 지점들을 한 칸씩 앞으로 당깁니다. (a <- b, b <- c)
        a = b;
        ya = yb;
        b = c;
        yb = yc;
        
        // 다음 탐색을 위해 스텝 크기를 k배(보통 2배)로 확장하여 탐색 속도를 가속합니다.
        s *= k;  
    }
}

/**
 * @brief Algorithm 3.3: Golden Section Search (황금 분할 탐색)
 * 
 * 미분(Gradient) 정보를 사용하지 않고 오직 함수값(f(x))만을 평가하여 
 * 1차원 함수에서 극솟값을 찾는 대표적인 0차 최적화(Zero-Order Optimization) 알고리즘입니다.
 * 주어진 탐색 구간(Range)을 황금비율(약 0.618)로 좁혀가며 최적점을 추적합니다.
 * 매 반복마다 한 점만 새롭게 평가하므로 함수 평가 횟수가 적다는 장점이 있습니다.
 * 
 * @tparam Func 평가 함수 타입
 * @param f 최소화하고자 하는 목적 함수
 * @param r 극솟값이 포함된 것이 보장된 초기 탐색 구간 (예: bracket_minimum의 결과)
 * @param n 탐색을 수행할 총 반복 횟수 (반복 횟수가 많을수록 정확해짐)
 * @return double 발견된 극솟값의 위치 (최종 구간의 중점)
 */
template <typename Func>
inline double golden_section_search(Func f, Range r, int n) {
    // 황금비(Golden Ratio): (sqrt(5) - 1) / 2 ≈ 0.61803398...
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;  
    double a = r.a;
    double b = r.b;

    // 구간 내에서 황금비를 기준으로 두 개의 내분점(x1, x2)을 생성
    double x1 = b - phi * (b - a);
    double x2 = a + phi * (b - a);
    double y1 = f(x1);
    double y2 = f(x2);

    for (int i = 0; i < n; ++i) {
        // [구간 축소 로직]
        if (y1 < y2) {
            // x1 지점의 함수값이 더 작다면, 최솟값은 무조건 [a, x2] 사이에 존재합니다.
            // 따라서 오른쪽 경계(b)를 x2로 축소합니다.
            b = x2;
            x2 = x1;     // 이전의 x1은 새로운 구간의 오른쪽 내분점(x2)이 됩니다.
            y2 = y1;     // 함수 평가 재사용 (성능 이점)
            x1 = b - phi * (b - a); // 새로운 왼쪽 내분점 계산
            y1 = f(x1);
        } else {
            // x2 지점의 함수값이 더 작거나 같다면, 최솟값은 무조건 [x1, b] 사이에 존재합니다.
            // 따라서 왼쪽 경계(a)를 x1로 축소합니다.
            a = x1;
            x1 = x2;     // 이전의 x2는 새로운 구간의 왼쪽 내분점(x1)이 됩니다.
            y1 = y2;     // 함수 평가 재사용 (성능 이점)
            x2 = a + phi * (b - a); // 새로운 오른쪽 내분점 계산
            y2 = f(x2);
        }
    }
    // 탐색이 완료된 후 최종적으로 축소된 구간의 중앙값을 최적점으로 반환합니다.
    return (a + b) / 2.0;
}

/**
 * @brief Algorithm 3.6: Bisection Method (이분법)
 * 
 * Dual Number(이원수) 연산을 통해 얻은 1차 미분(Gradient, f'(x)) 정보를 활용하여 
 * 미분값이 0이 되는 지점(f'(x) = 0, 즉 임계점)을 빠르게 찾는 알고리즘입니다.
 * 구간을 항상 절반으로 나누어가며 탐색하므로 수렴 속도가 비교적 빠르고 안정적입니다.
 * (단, 미분 가능해야 하며 목적 함수가 유니모달(Unimodal)하다고 가정합니다.)
 * 
 * @tparam FuncDual 이원수(Dual Number)를 매개변수로 받아 함수값과 미분값을 반환하는 목적 함수 타입
 * @param f 평가 함수. `Dual.hpp`에 정의된 이원수를 입력받아 평가합니다.
 * @param range 탐색을 수행할 초기 구간 [a, b]
 * @param n 탐색을 수행할 총 반복 횟수
 * @return double 미분값이 0에 가까워지는 최적점의 위치
 */
template <typename FuncDual>
inline double bisection_method(FuncDual f, Range range, int n) {
    double a = range.a;
    double b = range.b;

    for (int i = 0; i < n; ++i) {
        // 현재 탐색 구간의 정중앙(mid) 지점을 계산합니다.
        double mid = (a + b) / 2.0;

        // 중앙 지점(mid)에서 미분값을 평가하기 위해 이원수(Dual Number)를 생성합니다.
        // 실수부(Real)에는 위치 'mid'를, 쌍대부(Dual)에는 미분 계수 시딩(Seeding)을 위한 '1.0'을 넣습니다.
        auto res = f(Dual<double>(mid, 1.0));

        // [구간 축소 로직]
        // res.d 는 이원수 연산을 통해 자동으로 계산된 현재 위치(mid)에서의 1차 미분값(기울기, f'(mid))입니다.
        if (res.d > 0) {
            // 기울기가 양수(+)라는 것은 현재 지점 기준 오른쪽은 함수값이 증가한다는 뜻입니다.
            // 극솟값은 함수값이 감소하는 왼쪽 구간 [a, mid]에 존재하므로 오른쪽 경계(b)를 당깁니다.
            b = mid;
        } else {
            // 기울기가 음수(-)라는 것은 현재 지점 기준 왼쪽은 함수값이 증가(또는 감소 중)한다는 뜻입니다.
            // 극솟값은 오른쪽 구간 [mid, b]에 존재하므로 왼쪽 경계(a)를 당깁니다.
            a = mid;
        }
    }
    // 탐색이 완료된 후 최종적으로 축소된 구간의 중앙값을 최적점으로 반환합니다.
    return (a + b) / 2.0;
}

/**
 * @brief Algorithm 3.5: Shubert-Piyavskii Method (전역 최적화 알고리즘)
 * 
 * 립시츠 상수(Lipschitz Constant, L)를 활용하여 다수의 국소 최적해(Local Minima)가 존재하는 
 * 함수에서도 확실한 전역 최솟값(Global Minimum)을 찾아내는 알고리즘입니다.
 * 함수의 기울기가 L을 넘지 않는다는 성질을 이용하여, 이미 평가한 점들을 바탕으로 함수값의 '하한선(Lower Bound)'을 
 * V자 톱날(Sawtooth) 모양으로 그려가며, 그 하한선이 가장 깊게 파인 곳을 다음 탐색 지점으로 선정합니다.
 */
struct Pt {
    double x;  ///< X 좌표 (위치)
    double y;  ///< Y 좌표 (해당 위치에서의 함수값)
};

/**
 * @brief Shubert-Piyavskii 알고리즘 실행 함수
 * 
 * @tparam Func 평가 함수 타입
 * @param f 전역 최솟값을 찾고자 하는 목적 함수
 * @param r 탐색을 수행할 초기 구간
 * @param l 함수의 립시츠 상수 (Lipschitz Constant). 함수 기울기의 최대 절댓값이어야 합니다.
 * @param eps 수렴 판정을 위한 허용 오차. 하한선(best_z)과 현재 최소 함수값(min_f)의 차이가 이보다 작으면 종료.
 * @param max_iter 무한 루프를 방지하기 위한 최대 반복 횟수
 * @return double 탐색을 통해 발견된 전역 최소 함수값(Global Minimum Value)
 */
template <typename Func>
inline double shubert_piyavskii(Func f, Range r, double l, double eps, int max_iter) {
    // 초기화: 구간의 양 끝점(a, b)을 평가하여 톱날의 시작점으로 리스트에 저장합니다.
    // X 좌표를 기준으로 항상 오름차순 정렬 상태를 유지해야 합니다.
    std::vector<Pt> pts = {{r.a, f(r.a)}, {r.b, f(r.b)}};

    // 지금까지 발견된 가장 작은 실제 함수값 추적
    double min_f = std::min(pts[0].y, pts[1].y);

    for (int i = 0; i < max_iter; ++i) {
        int best_idx = 0;
        double best_z = std::numeric_limits<double>::infinity();

        // [Step 1] V자 톱날(하한선)의 교점 중 가장 깊게 파인(가장 작은) z 좌표 찾기
        // pts 배열에 인접한 두 점 사이에서 만들어지는 V자 모양의 꼭짓점 높이(z)를 계산합니다.
        for (size_t j = 0; j < pts.size() - 1; ++j) {
            // 립시츠 상수를 이용한 하한선 교점의 Y 좌표(z) 수식
            double z = (pts[j].y + pts[j + 1].y) / 2.0 - l * (pts[j + 1].x - pts[j].x) / 2.0;
            if (z < best_z) {
                best_z = z;
                best_idx = static_cast<int>(j); // 가장 유망한(낮은) 교점이 속한 구간의 왼쪽 인덱스 저장
            }
        }

        // [Step 2] 가장 유망한 교점의 X 좌표를 계산하고 실제 함수 평가 수행
        double new_x = (pts[best_idx].x + pts[best_idx + 1].x) / 2.0 -
                       (pts[best_idx + 1].y - pts[best_idx].y) / (2.0 * l);
        
        // 새로 예측한 지점에서 실제 함수값을 평가합니다.
        double new_y = f(new_x);

        // 탐색된 함수값 중 최소값을 갱신합니다.
        min_f = std::min(min_f, new_y);

        // [Step 3] 새로운 점 삽입 및 배열 정렬 상태 유지
        // best_idx와 best_idx + 1 사이에 새로운 점을 삽입하여 X 좌표 오름차순을 유지합니다.
        pts.insert(pts.begin() + best_idx + 1, {new_x, new_y});

        // [Step 4] 수렴 조건(Convergence Check) 확인
        // 현재 발견된 최소 함수값(min_f)과 립시츠 하한선의 최소값(best_z) 사이의 갭(오차)이 
        // 허용 오차(eps)보다 작아졌다면 전역 최솟값을 충분히 찾았다고 판단하고 종료합니다.
        if (std::abs(min_f - best_z) < eps) break;
    }
    
    // 최종적으로 발견된 가장 작은 함수값을 반환합니다.
    return min_f;
}
}  // namespace Optimization

#endif  // OPTIMIZATION_BRACKETING_HPP_