#include <iostream>
#include <iomanip>
#include <string>
#include "Optimization/Physics/VehicleModel.hpp"
#include "Optimization/Simulation/Integrator.hpp"
#include "Optimization/AutoDiff.hpp"

using namespace Optimization;

template <typename MatType>
void print_matrix(const std::string& name, const MatType& mat, size_t rows, size_t cols) {
    std::cout << "── " << name << " ──\n";
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "  [ ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << mat(static_cast<int>(i), static_cast<int>(j)) << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << std::string(50, '=') << "\n";
    std::cout << "  [Layer 7] Vehicle Model & RK4 Integrator AD Test\n";
    std::cout << std::string(50, '=') << "\n";

    DynamicBicycleModel model;
    double dt = 0.05; // 50ms 제어 주기

    // Nominal State x: [X, Y, theta, vx, vy, omega]
    StaticVector<double, 6> x_nom;
    x_nom.set_zero();
    x_nom(3) = 10.0; // 10 m/s 직진 상태 (특이점 회피)

    // Nominal Input u: [a, delta]
    StaticVector<double, 2> u_nom;
    u_nom(0) = 0.0;  // 가속 없음
    u_nom(1) = 0.1;  // 약 5.7도 조향 (비선형 횡거동 유도)

    // 1. 시스템 행렬 A 추출용 Functor (x를 변수, u를 상수로 고정)
    auto func_A = [&](const auto& x_dual) {
        using T = std::decay_t<decltype(x_dual(0))>;
        StaticVector<T, 2> u_t;
        u_t(0) = T(u_nom(0));
        u_t(1) = T(u_nom(1));
        return Integrator::rk4<6, 2, DynamicBicycleModel, T>(model, x_dual, u_t, dt);
    };

    // 2. 입력 행렬 B 추출용 Functor (u를 변수, x를 상수로 고정)
    auto func_B = [&](const auto& u_dual) {
        using T = std::decay_t<decltype(u_dual(0))>;
        StaticVector<T, 6> x_t;
        for(size_t i=0; i<6; ++i) x_t(static_cast<int>(i)) = T(x_nom(static_cast<int>(i)));
        return Integrator::rk4<6, 2, DynamicBicycleModel, T>(model, x_t, u_dual, dt);
    };

    std::cout << "  [*] 추출 진행: Forward-Mode AD 가동...\n\n";

    // 단 두 줄의 코드로 RK4 적분기를 통과한 모델의 Jacobian(A, B)을 수치 오차 없이 추출해냅니다.
    StaticMatrix<double, 6, 6> A = AutoDiff::jacobian<6, 6>(func_A, x_nom);
    StaticMatrix<double, 6, 2> B = AutoDiff::jacobian<6, 2>(func_B, u_nom);

    print_matrix("System Matrix A (6x6)", A, 6, 6);
    print_matrix("Input Matrix B (6x2)", B, 6, 2);

    std::cout << "  [PASS] 이산화된 차량 동역학의 Jacobian (A, B) 추출 완료.\n";
    std::cout << std::string(50, '=') << "\n";

    return 0;
}