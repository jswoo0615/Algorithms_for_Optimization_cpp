#include <iostream>
#include <cmath>
#include <string>
#include "Optimization/SQPSolver.hpp"

using namespace Optimization;

static int g_total = 0;
static int g_passed = 0;
static int g_failed = 0;

void check(bool cond, const std::string& name) {
    ++g_total;
    if (cond) {
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        ++g_failed;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

void section(const std::string& title) {
    std::cout << "\n── " << title << " ──\n";
}

// =========================================================================
// 테스트 문제 정의 (비선형 최적화 Functors)
// =========================================================================

// 목적 함수: J(u) = (u1 - 3)^2 + (u2 - 3)^2
struct CostFunc {
    template <typename VecType>
    auto operator()(const VecType& u) const {
        using T = std::decay_t<decltype(u(0))>;
        T d1 = u(0) - T(3.0);
        T d2 = u(1) - T(3.0);
        return d1 * d1 + d2 * d2;
    }
};

// 등식 제약: 없음 (N_eq = 0 이므로 실제로 호출되지 않음)
struct DummyEq {
    template <typename VecType>
    auto operator()(const VecType& u) const {
        (void)u; // 미사용 변수 경고 방지
        using T = std::decay_t<decltype(u(0))>;
        return StaticVector<T, 0>(); 
    }
};

// 부등식 제약: u1^2 + u2^2 - 2 <= 0  (반지름 sqrt(2)인 원형 영역 내부)
struct IneqFunc {
    template <typename VecType>
    auto operator()(const VecType& u) const {
        using T = std::decay_t<decltype(u(0))>;
        StaticVector<T, 1> res;
        res(0) = u(0) * u(0) + u(1) * u(1) - T(2.0);
        return res;
    }
};

// =========================================================================
// 테스트 실행
// =========================================================================
void test_SQP_nonlinear_circle() {
    section("SQP 비선형 제약 최적화 (원형 경계면 충돌 테스트)");

    // 변수 2개, 등식 0개, 부등식 1개
    SQPSolver<2, 0, 1> solver;
    
    // 초기 추정치 (원점에서 탐색 시작)
    StaticVector<double, 2> u;
    u(0) = 0.0;
    u(1) = 0.0;

    CostFunc cost_f;
    DummyEq eq_f;
    IneqFunc ineq_f;

    // SQP 메인 엔진 가동
    bool success = solver.solve(u, cost_f, eq_f, ineq_f, 50);

    check(success, "SQP Solver 수렴 완료 (Iteration Limit 이내 종료)");

    std::cout << "  [*] [Result u*] : [" << u(0) << ", " << u(1) << "]\n";

    double u1_err = std::abs(u(0) - 1.0);
    double u2_err = std::abs(u(1) - 1.0);
    
    // 비선형 근사(Taylor Expansion) 과정의 오차를 감안하여 1e-4 허용치 적용
    constexpr double TOL = 1e-4; 

    check(u1_err < TOL && u2_err < TOL, "비선형 부등식 제약 최적해(1.0, 1.0) 도출 정확도 검증");
}

int main() {
    std::cout << std::string(50, '=') << "\n";
    std::cout << "  SQP Solver 기반 비선형 NLP 검증 테스트\n";
    std::cout << std::string(50, '=') << "\n";

    test_SQP_nonlinear_circle();

    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "  결과: " << g_passed << " / " << g_total << " PASSED";
    if (g_failed > 0) std::cout << "  (" << g_failed << " FAILED)";
    std::cout << "\n" << std::string(50, '=') << "\n";

    return g_failed == 0 ? 0 : 1;
}