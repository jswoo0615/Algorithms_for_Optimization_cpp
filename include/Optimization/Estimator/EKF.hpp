#ifndef OPTIMIZATION_EKF_HPP_
#define OPTIMIZATION_EKF_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"
#include "Optimization/Dual.hpp"
#include "Optimization/Integrator/RK4.hpp"
#include <cmath>

namespace Optimization {
namespace estimator {

template <size_t Nx = 6, size_t Nu = 2>
class EKF {
public:
    StaticVector<double, Nx> x_est;
    StaticMatrix<double, Nx, Nx> P;
    StaticMatrix<double, Nx, Nx> Q; // Process noise covariance (모델 신뢰도)
    StaticMatrix<double, Nx, Nx> R; // Measurement noise covariance (센서 신뢰도)
    
    EKF() {
        x_est.set_zero();
        
        // [Architect's Fix] set_identity() 대신 set_zero() 후 대각 성분 초기화
        P.set_zero(); 
        for(size_t i=0; i<Nx; ++i) P(i,i) = 1.0;
        
        Q.set_zero();
        for(size_t i=0; i<Nx; ++i) Q(i,i) = 0.01;
        
        R.set_zero();
        R(0,0) = 0.2; R(1,1) = 0.2; // X, Y 노이즈
        R(2,2) = 0.05;              // Yaw 노이즈
        R(3,3) = 0.1;               // Vx 노이즈
        R(4,4) = 0.1; R(5,5) = 0.1; 
    }

    // AD 엔진의 한계를 보완하는 순수 역행렬기 (의존성 최소화)
    StaticMatrix<double, Nx, Nx> invert(StaticMatrix<double, Nx, Nx> mat) {
        StaticMatrix<double, Nx, Nx> inv; 
        inv.set_zero();
        for(size_t i=0; i<Nx; ++i) inv(i,i) = 1.0; // set_identity() 대체

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
        for(size_t i=0; i<Nx; ++i) x_dual(i) = ADVar::make_variable(x_est(i), i);
        for(size_t i=0; i<Nu; ++i) u_dual(i) = ADVar(u(i)); 

        StaticVector<ADVar, Nx> x_next_dual = integrator::step_rk4<Nx, Nu>(model, x_dual, u_dual, dt);
        
        for(size_t i=0; i<Nx; ++i) {
            for(size_t j=0; j<Nx; ++j) {
                F(i, j) = x_next_dual(i).g[j];
            }
        }

        // 3. 공분산 예측: P = F * P * F^T + Q
        StaticMatrix<double, Nx, Nx> FP; FP.set_zero();
        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                for(size_t k=0; k<Nx; ++k)
                    FP(i,j) += F(i,k) * P(k,j);

        StaticMatrix<double, Nx, Nx> FPFt; FPFt.set_zero();
        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                for(size_t k=0; k<Nx; ++k)
                    FPFt(i,j) += FP(i,k) * F(j,k); 

        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                P(i,j) = FPFt(i,j) + Q(i,j);
    }

    void update(const StaticVector<double, Nx>& z) {
        // S = P + R
        StaticMatrix<double, Nx, Nx> S; S.set_zero();
        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                S(i,j) = P(i,j) + R(i,j);

        // K = P * S^-1
        StaticMatrix<double, Nx, Nx> S_inv = invert(S);
        StaticMatrix<double, Nx, Nx> K; K.set_zero();
        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                for(size_t k=0; k<Nx; ++k)
                    K(i,j) += P(i,k) * S_inv(k,j);

        // y = z - x_est
        StaticVector<double, Nx> y;
        for(size_t i=0; i<Nx; ++i) {
            y(i) = z(i) - x_est(i);
            if (i == 2) {
                while (y(i) > M_PI) y(i) -= 2.0 * M_PI;
                while (y(i) < -M_PI) y(i) += 2.0 * M_PI;
            }
        }

        // x = x + K * y
        StaticVector<double, Nx> dx; dx.set_zero();
        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                dx(i) += K(i,j) * y(j);

        for(size_t i=0; i<Nx; ++i) x_est(i) += dx(i);

        // P = (I - K) * P
        StaticMatrix<double, Nx, Nx> I_minus_K; I_minus_K.set_zero();
        for(size_t i=0; i<Nx; ++i) {
            I_minus_K(i,i) = 1.0;
            for(size_t j=0; j<Nx; ++j) {
                I_minus_K(i,j) -= K(i,j);
            }
        }

        StaticMatrix<double, Nx, Nx> P_new; P_new.set_zero();
        for(size_t i=0; i<Nx; ++i)
            for(size_t j=0; j<Nx; ++j)
                for(size_t k=0; k<Nx; ++k)
                    P_new(i,j) += I_minus_K(i,k) * P(k,j);
        
        P = P_new;
    }
};

} // namespace estimator
} // namespace Optimization
#endif