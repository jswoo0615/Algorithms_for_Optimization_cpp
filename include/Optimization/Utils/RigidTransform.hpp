#ifndef OPTIMIZATION_RIGID_TRANSFORM_HPP_
#define OPTIMIZATION_RIGID_TRANSFORM_HPP_

#include "Optimization/Matrix/StaticMatrix.hpp"
#include <cmath>

namespace Optimization {
namespace utils {

/**
 * @brief SE(2) 평면상의 동차 변환(Homogeneous Transform) 유틸리티
 */
class SE2Transform {
public:
    /**
     * @brief 전역 좌표(X, Y)를 차량 기준 로컬 좌표(x, y)로 변환하는 행렬 생성
     * @param tx 차량의 현재 전역 X
     * @param ty 차량의 현재 전역 Y
     * @param psi 차량의 현재 헤딩(Yaw)
     */
    static StaticMatrix<double, 3, 3> get_global_to_local(double tx, double ty, double psi) {
        StaticMatrix<double, 3, 3> T;
        double cos_p = std::cos(psi);
        double sin_p = std::sin(psi);

        // Inverse Transform Matrix (R^T and -R^T * t)
        T(0, 0) = cos_p;  T(0, 1) = sin_p; T(0, 2) = -(cos_p * tx + sin_p * ty);
        T(1, 0) = -sin_p; T(1, 1) = cos_p; T(1, 2) = -(-sin_p * tx + cos_p * ty);
        T(2, 0) = 0.0;    T(2, 1) = 0.0;   T(2, 2) = 1.0;
        
        return T;
    }

    /**
     * @brief 포인트 변환 실행 (3x1 벡터 연산)
     */
    static void transform_point(const StaticMatrix<double, 3, 3>& T, double global_x, double global_y, double& local_x, double& local_y) {
        local_x = T(0, 0) * global_x + T(0, 1) * global_y + T(0, 2);
        local_y = T(1, 0) * global_x + T(1, 1) * global_y + T(1, 2);
    }
};

} // namespace utils
} // namespace Optimization

#endif // OPTIMIZATION_RIGID_TRANSFORM_HPP_