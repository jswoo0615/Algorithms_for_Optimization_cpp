#include <gtest/gtest.h>
#include "Optimization/Solver/NewtonSolver.hpp"

using namespace Optimization;
using namespace Optimization::solver;

// 테스트용 비선형 연립방정식 (Functor)
// f1(x0, x1) = x0^2 + x1^2 - 4 = 0 (반지름이 2인 원)
// f2(x0, x1) = x0 * x1 - 1 = 0     (쌍곡선)
// 교점 중 하나는 대략 x0 = 1.9318, x1 = 0.5176
struct NonlinearSystem {
    template <typename T>
    StaticVector<T, 2> operator()(const StaticVector<T, 2>& x) const {
        StaticVector<T, 2> y;
        y(0) = x(0) * x(0) + x(1) * x(1) - 4.0;
        y(1) = x(0) * x(1) - 1.0;
        return y;
    }
};

TEST(NewtonSolverTest, MaxIterationReached) {
    // 1. 자코비안이 터지지 않는 정상적인 위치에서 시작 (x0=3.0, x1=1.0)
    StaticVector<double, 2> x_init;
    x_init(0) = 3.0; 
    x_init(1) = 1.0; 

    NonlinearSystem sys;
    // 2. 루프를 단 1번만 돌도록 제한하여 고의로 수렴하지 못하게 만듦
    SolverStatus status = solve_newton(x_init, sys, 1, 1e-6);

    // 수학적 에러 없이 루프가 끝났으므로 MAX_ITERATION_REACHED 반환 확인
    EXPECT_EQ(status, SolverStatus::MAX_ITERATION_REACHED);
}

TEST(NewtonSolverTest, SingularJacobianMathError) {
    // 1. 자코비안 행렬식이 0이 되는 악의적인 대칭점 주입
    StaticVector<double, 2> x_init;
    x_init(0) = 1e5; 
    x_init(1) = -1e5; 

    NonlinearSystem sys;
    SolverStatus status = solve_newton(x_init, sys, 20, 1e-6);

    // 2. 엔진이 헛발질을 하기 전에 특이 행렬을 감지하고 즉시 차단하는지 확인 (Fail-fast 검열)
    EXPECT_EQ(status, SolverStatus::MATH_ERROR);
}