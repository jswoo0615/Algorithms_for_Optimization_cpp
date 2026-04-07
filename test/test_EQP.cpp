#include <cmath>
#include <iostream>

#include "Optimization/EQPSolver.hpp"
#include "Optimization/KKTMonitor.hpp"

// 오차 허용 범위 (Tolerance)
constexpr double TOLERANCE = 1e-6;

void run_eqp_test() {
    std::cout << "========== [ EQP Solver Test ] ==========\n";

    // 1. 솔버 인스턴스 생성 (변수 2개, 제약조건 1개)
    // 동적 할당 없이 스택(Stack) 메모리에 64바이트 정렬되어 생성됨.
    EQPSolver<2, 1> solver;

    // 2. 목적 함수 P 행렬 세팅 (대칭 양정치)
    solver.P(0, 0) = 2.0;
    solver.P(0, 1) = 0.0;
    solver.P(1, 0) = 0.0;
    solver.P(1, 1) = 2.0;

    // 3. 목적 함수 q 벡터 세팅
    solver.q(0) = -2.0;
    solver.q(1) = -5.0;

    // 4. 등식 제약 조건 A 행렬 및 b 벡터 세팅
    solver.A(0, 0) = 1.0;
    solver.A(0, 1) = -1.0;
    solver.b(0) = -2.0;

    // 5. 엔진 구동
    bool success = solver.solve();

    if (!success) {
        std::cerr << "[FAIL] Solver failed to converge or singular matrix!\n";
        return;
    }

    std::cout << "[Result u*]      : [" << solver.u_opt(0) << ", " << solver.u_opt(1) << "]\n";
    std::cout << "[Result lambda*] : [" << solver.lambda_opt(0) << "]\n\n";

    // ==========================================================
    // [Architect's Check] KKT Monitor 가동
    // ==========================================================
    auto kkt_metrics = KKTMonitor<2, 1>::evaluate_EQP(solver.P, solver.q, solver.A, solver.b,
                                                      solver.u_opt, solver.lambda_opt);

    KKTMonitor<2, 1>::print_metrics(kkt_metrics);

    if (!kkt_metrics.is_optimal) {
        std::cerr << ">> [FATAL] KKT Conditions Violated. Do not trust this solution.\n";
    } else {
        std::cout << ">> [PASS] KKT System solved and analytically verified.\n";
    }
}

int main() {
    run_eqp_test();
    return 0;
}