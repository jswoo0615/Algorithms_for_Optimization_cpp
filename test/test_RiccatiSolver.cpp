#include <gtest/gtest.h>

#include <chrono>

#include "Optimization/Solver/RiccatiSolver.hpp"

using namespace Optimization;
using namespace Optimization::solver;

TEST(RiccatiSolverTest, PerformanceAndCorrectness) {
    constexpr size_t H = 20;  // 예측 구간 20스텝
    constexpr size_t Nx = 6;
    constexpr size_t Nu = 2;

    RiccatiSolver<H, Nx, Nu> riccati;

    // 가상의 이산화된 선형 차량 시스템 (A, B) 및 비용 행렬 (Q, R) 더미 데이터 주입
    for (size_t k = 0; k < H; ++k) {
        // 단위 행렬 기반의 단순화된 A, Q, R 세팅
        for (size_t i = 0; i < Nx; ++i) {
            riccati.A[k](i, i) = 1.0;
            riccati.Q[k](i, i) = 10.0;
            riccati.q[k](i) = 0.1;
        }
        for (size_t i = 0; i < Nu; ++i) {
            riccati.R[k](i, i) = 1.0;
            riccati.r[k](i) = 0.05;
        }
        // B 행렬 (대각 성분 일부)
        riccati.B[k](3, 0) = 0.5;  // 가속도 -> vx
        riccati.B[k](5, 1) = 0.2;  // 조향각 -> 요레이트

        // 동역학 갭 (d)
        riccati.d[k](0) = 0.01;
    }
    for (size_t i = 0; i < Nx; ++i) {
        riccati.Q[H](i, i) = 10.0;
        riccati.q[H](i) = 0.1;
    }

    // 시간 측정 시작
    auto start = std::chrono::high_resolution_clock::now();

    SolverStatus status = riccati.solve();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(status, SolverStatus::SUCCESS);

    std::cout << "\n[Sparse Riccati Solver Performance]" << std::endl;
    std::cout << "Dimension equivalent to : 166 x 166 Dense Matrix" << std::endl;
    std::cout << "Execution Time          : " << duration.count() << " microseconds ("
              << duration.count() / 1000.0 << " ms)" << std::endl;

    // 솔버가 극도로 빨라야 함 (보통 100us ~ 500us 이내)
    EXPECT_LT(duration.count(), 2000);
}