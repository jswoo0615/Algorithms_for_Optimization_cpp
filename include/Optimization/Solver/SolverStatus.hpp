#ifndef OPTIMIZATION_SOLVER_STATUS_HPP_
#define OPTIMIZATION_SOLVER_STATUS_HPP_

namespace Optimization {
// =========================================================================================
// Layer 4 : 최적화 솔버 상태 코드 (Solver Status)
// 수학적 연산 (Layer 2)의 성공 여부 및 '수렴성 (Convergence)'와 '제약조건 (Feasibility)'을
// 진단합니다
// =========================================================================================
enum class SolverStatus {
    SUCCESS = 1,                 // 허용 오차 (Tolerance) 내에 수렴
    MAX_ITERATION_REACHED = -1,  // 최대 반복 횟수를 초과함 (해를 찾지 못함)
    MATH_ERROR = -2,  // 내부 선형대수 엔진에서 수치적 붕괴 발생 (Singular 등)
    INFEASIBLE = -3  // 제약 조건을 만족하는 해 영역이 존재하지 않음 (향후 QP/SQP용)
};
}  // namespace Optimization

#endif  // OPTIMIZATION_SOLVER_STATUS_HPP_