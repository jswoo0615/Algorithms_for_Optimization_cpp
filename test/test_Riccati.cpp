#include "Optimization/Matrix/MatrixEngine.hpp"
#include "Optimization/BlockBandedSolver.hpp"
#include <iostream>
#include <iomanip>

using namespace Optimization;

int main() {
    std::cout << "[NMPC Architect] Block-Banded Riccati Solver Test\n";
    std::cout << "--------------------------------------------------\n";

    // 1. 차원 및 호라이즌 설정 (1D 위치/속도 제어)
    constexpr size_t Nx = 2;  // 상태: [위치 오차, 속도 오차]
    constexpr size_t Nu = 1;  // 입력: [가속도]
    constexpr size_t Np = 15; // 호라이즌 길이

    BlockBandedSolver<Nx, Nu, Np> solver;

    double dt = 0.1; // 제어 주기 100ms

    // 2. 시스템 및 비용 함수 정의 (Dynamic Programming 세팅)
    for (size_t k = 0; k < Np; ++k) {
        auto& s = solver.stages[k];
        
        // A 행렬 (상태 전이: 위치 = 위치 + 속도*dt, 속도 = 속도)
        s.A(0, 0) = 1.0; s.A(0, 1) = dt;
        s.A(1, 0) = 0.0; s.A(1, 1) = 1.0;

        // B 행렬 (제어 입력: 위치에 미치는 영향 0, 속도에 미치는 영향 dt)
        s.B(0, 0) = 0.0;
        s.B(1, 0) = dt;

        // Q 행렬 (상태 오차 페널티 - 목적지와의 거리)
        s.Q(0, 0) = 10.0; // 위치 오차는 빨리 0으로 만들어야 함 (강한 페널티)
        s.Q(1, 1) = 1.0;  // 속도는 부드럽게 변해도 됨 (약한 페널티)

        // R 행렬 (제어 입력 페널티)
        s.R(0, 0) = 1.0;  // 가속도를 너무 급격하게 쓰지 않도록 제어
    }

    // 터미널 코스트 (Terminal Cost: 목적지 도달 시점의 페널티)
    solver.P_N(0, 0) = 10.0;
    solver.P_N(1, 1) = 1.0;

    // 3. 초기 상태 오차 설정 (현재 목표 지점으로부터 5.0m 떨어져 있음)
    StaticVector<double, Nx> dx_0;
    dx_0(0) = 5.0;  // 위치 오차: 5.0m
    dx_0(1) = 0.0;  // 속도 오차: 0.0m/s

    // 4. 솔버 실행 (O(Np) 연산의 심장부)
    bool success = solver.solve(dx_0);

    if (!success) {
        std::cout << "[ERROR] 수치적 불안정성 발생 (H_uu Matrix is strictly not Positive Definite).\n";
        return -1;
    }

    // 5. 결과 검증 (KKT Monitor의 기초 형태)
    std::cout << "[SUCCESS] Riccati Recursion Completed.\n\n";
    std::cout << "Step | Pos Error | Vel Error | Control Input (Acc)\n";
    std::cout << "--------------------------------------------------\n";

    for (size_t k = 0; k < Np; ++k) {
        std::cout << std::setw(4) << k << " | "
                  << std::fixed << std::setw(9) << std::setprecision(4) << solver.dx[k](0) << " | "
                  << std::setw(9) << solver.dx[k](1) << " | "
                  << std::setw(13) << solver.du[k](0) << "\n";
    }
    
    // 터미널 상태 출력
    std::cout << std::setw(4) << Np << " | "
              << std::fixed << std::setw(9) << std::setprecision(4) << solver.dx[Np](0) << " | "
              << std::setw(9) << solver.dx[Np](1) << " | "
              << std::setw(13) << "-" << "\n";

    return 0;
}