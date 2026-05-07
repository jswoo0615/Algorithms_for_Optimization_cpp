#ifndef OPTIMIZATION_EKF_HPP_
#define OPTIMIZATION_EKF_HPP_

#include <cmath>

#include "Optimization/Dual.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Matrix/LinearAlgebra.hpp"

namespace Optimization {
namespace estimator {

template <size_t Nx = 6, size_t Nu = 2>
class EKF {
   public:
    StaticVector<double, Nx> x_est;
    StaticMatrix<double, Nx, Nx> P;
    StaticMatrix<double, Nx, Nx> Q;  // Process noise covariance
    StaticMatrix<double, Nx, Nx> R;  // Measurement noise covariance

    EKF() {
        x_est.set_zero();

        P.set_zero();
        for (size_t i = 0; i < Nx; ++i) P(i, i) = 1.0;

        Q.set_zero();
        for (size_t i = 0; i < Nx; ++i) Q(i, i) = 0.01;

        R.set_zero();
        R(0, 0) = 0.2;
        R(1, 1) = 0.2;   
        R(2, 2) = 0.05;  
        R(3, 3) = 0.1;   
        R(4, 4) = 0.1;
        R(5, 5) = 0.1;
    }

    // 역행렬기 (추후 StaticMatrix 레이어 또는 별도 Math 모듈로 분리 권장)
    StaticMatrix<double, Nx, Nx> invert(StaticMatrix<double, Nx, Nx> mat) {
        StaticMatrix<double, Nx, Nx> inv;
        inv.set_zero();
        for (size_t i = 0; i < Nx; ++i) inv(i, i) = 1.0;

        for (size_t i = 0; i < Nx; ++i) {
            double pivot = mat(i, i);
            if (std::abs(pivot) < 1e-12) pivot = (pivot < 0) ? -1e-12 : 1e-12;
            for (size_t j = 0; j < Nx; ++j) {
                mat(i, j) /= pivot;
                inv(i, j) /= pivot;
            }
            for (size_t k = 0; k < Nx; ++k) {
                if (k != i) {
                    double factor = mat(k, i);
                    for (size_t j = 0; j < Nx; ++j) {
                        mat(k, j) -= factor * mat(i, j);
                        inv(k, j) -= factor * inv(i, j);
                    }
                }
            }
        }
        return inv;
    }

    template <typename Model>
    void predict(Model& model, const StaticVector<double, Nu>& u, double dt) {
        // 1. 상태 예측 (RK4)
        x_est = integrator::step_rk4<Nx, Nu>(model, x_est, u, dt);

        // 2. 비선형 동역학 자코비안 F 추출 (AD Engine 적용)
        StaticMatrix<double, Nx, Nx> F;
        using ADVar = DualVec<double, Nx>;
        StaticVector<ADVar, Nx> x_dual;
        StaticVector<ADVar, Nu> u_dual;
        
        for (size_t i = 0; i < Nx; ++i) x_dual(i) = ADVar::make_variable(x_est(i), i);
        for (size_t i = 0; i < Nu; ++i) u_dual(i) = ADVar(u(i));

        StaticVector<ADVar, Nx> x_next_dual =
            integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);

        for (size_t i = 0; i < Nx; ++i) {
            for (size_t j = 0; j < Nx; ++j) {
                F(i, j) = x_next_dual(i).g[j];
            }
        }

        // 3. 공분산 예측 (3중 for문 제거 및 완벽한 수식화)
        P = (F * P * F.transpose()) + Q;
    }

    void update(const StaticVector<double, Nx>& z) {
        // 1. S = P + R
        StaticMatrix<double, Nx, Nx> S = P + R;

        // 2. 칼만 게인 K = P * S^-1
        StaticMatrix<double, Nx, Nx> K = P * invert(S);

        // 3. 잔차 y = z - x_est (각도 wrap-around 처리 포함)
        StaticVector<double, Nx> y = z - x_est;
        while (y(2) > M_PI) y(2) -= 2.0 * M_PI;
        while (y(2) < -M_PI) y(2) += 2.0 * M_PI;

        // 4. 상태 업데이트 x_est = x_est + K * y
        x_est = x_est + (K * y);

        // 5. 공분산 업데이트 P = (I - K) * P
        StaticMatrix<double, Nx, Nx> I;
        I.set_zero();
        for (size_t i = 0; i < Nx; ++i) I(i, i) = 1.0;

        P = (I - K) * P;
    }
};

}  // namespace estimator
}  // namespace Optimization
#endif