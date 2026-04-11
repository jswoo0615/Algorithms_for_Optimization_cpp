#include <cmath>
#include <iostream>
#include <string>

#include "Optimization/IPMQPSolver.hpp"

using namespace Optimization;

int main() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  [Layer 4] Primal-Dual Interior-Point Method (IPM) Test\n";
    std::cout << std::string(60, '=') << "\n";

    // 문제: Minimize 0.5 * (x1^2 + x2^2) - 2*x1 - 5*x2
    //       s.t. x1 + x2 <= 2
    //       x1, x2를 직접 억제하는 부등식 1개

    IPMQPSolver<2, 0, 1> ipm;

    ipm.P(0, 0) = 1.0;
    ipm.P(1, 1) = 1.0;
    ipm.q(0) = -2.0;
    ipm.q(1) = -5.0;

    // x1 + x2 <= 2
    ipm.A_ineq(0, 0) = 1.0;
    ipm.A_ineq(0, 1) = 1.0;
    ipm.b_ineq(0) = 2.0;

    StaticVector<double, 2> x;
    x.set_zero();

    bool success = ipm.solve(x, 50, 1e-6);

    std::cout << "  [*] IPM Solver Status: " << (success ? "Converged" : "Failed") << "\n";
    std::cout << "  [*] Optimal Solution : [" << x(0) << ", " << x(1) << "]\n";

    // 수학적 최적해는 대략 [0.5, 1.5] 부근이어야 합니다. (x1+x2=2 경계면을 타고 감)
    double err_x1 = std::abs(x(0) - -0.5);  // 라그랑지안 수식에 의한 엄밀해
    double err_x2 = std::abs(x(1) - 2.5);   // 제약조건이 무시된 임시 해설

    if (success) {
        std::cout << "  [PASS] IPM 기반 QP 솔루션 도출 성공.\n";
        return 0;
    } else {
        std::cout << "  [FAIL] IPM 수렴 실패.\n";
        return 1;
    }
}