#ifndef OPTIMIZATION_PATH_PLANNER_HPP_
#define OPTIMIZATION_PATH_PLANNER_HPP_

#include <array>
#include <cmath>
#include <vector>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

template <size_t Np, size_t Nx>
class PathPlanner {
   public:
    struct ObstacleInfo {
        double x, y, r;
    };

    // [Architect's Extreme Slalom] 장애물 사이의 '게이트'를 통과하도록 경로 강제 생성
    static void generate_slalom_reference(const StaticVector<double, Nx>& x_curr,
                                          const StaticVector<double, Nx>& x_final,
                                          const std::array<ObstacleInfo, 10>& obstacles, double dt,
                                          StaticVector<double, Nx> ref_horizon[Np]) {
        double v_ref = 2.0;

        // 1. 게이트(Waypoints) 설정: 장애물 사이사이를 통과하도록 좌표 지정
        struct Point {
            double x, y;
        };
        std::vector<Point> gates;

        // 장애물을 페어로 묶어 그 사이를 통과점으로 설정
        for (int i = 0; i < 9; i += 2) {
            gates.push_back({(obstacles[i].x + obstacles[i + 1].x) / 2.0,
                             (obstacles[i].y + obstacles[i + 1].y) / 2.0});
        }
        gates.push_back({x_final(0), x_final(1)});

        // 2. 현재 위치에서 가장 적합한 게이트를 추적하며 Horizon 생성
        for (size_t k = 0; k < Np; ++k) {
            double lookahead_dist = v_ref * dt * (k + 1);
            double target_x = x_final(0), target_y = x_final(1);

            for (auto& gate : gates) {
                if (gate.x > x_curr(0) + lookahead_dist * 0.5) {
                    target_x = gate.x;
                    target_y = gate.y;
                    break;
                }
            }

            double dx = target_x - x_curr(0);
            double dy = target_y - x_curr(1);
            double angle = std::atan2(dy, dx);

            ref_horizon[k](0) = x_curr(0) + std::cos(angle) * lookahead_dist;
            ref_horizon[k](1) = x_curr(1) + std::sin(angle) * lookahead_dist;
            ref_horizon[k](2) = angle;
            ref_horizon[k](3) = v_ref;
            ref_horizon[k](4) = 0.0;
            ref_horizon[k](5) = 0.0;
        }
    }
};

}  // namespace Optimization
#endif