#ifndef OPTIMIZATION_PATH_PLANNER_HPP_
#define OPTIMIZATION_PATH_PLANNER_HPP_

#include <array>
#include <cmath>
#include <vector>

#include "Optimization/Matrix/MatrixEngine.hpp"

namespace Optimization {

/**
 * @brief 장애물 회피 및 경로 생성을 수행하는 PathPlanner 클래스
 * 
 * @tparam Np 예측 구간(Prediction Horizon)의 길이 (미래 경로점의 개수)
 * @tparam Nx 상태 벡터(State Vector)의 차원
 */
template <size_t Np, size_t Nx>
class PathPlanner {
   public:
    /**
     * @brief 장애물의 정보를 담는 구조체
     * 
     * x, y는 장애물의 2차원 위치 중심점, r은 장애물의 반지름(크기)을 나타냅니다.
     */
    struct ObstacleInfo {
        double x, y, r;
    };

    /**
     * @brief [Architect's Extreme Slalom] 장애물 사이의 '게이트'를 통과하도록 강제로 슬라롬(Slalom) 참조 경로를 생성합니다.
     * 
     * 현재 상태(x_curr)에서 목표 상태(x_final)까지 이동하되, 주어진 장애물(obstacles) 사이를 
     * 지그재그 혹은 사이사이로 통과하도록 Np개의 참조(Reference) 경로(Horizon)를 생성합니다.
     * 
     * @param x_curr 현재 차량/에이전트의 상태 벡터
     * @param x_final 최종 목표 상태 벡터
     * @param obstacles 10개의 장애물 정보를 담은 배열 (장애물 2개씩 짝지어 게이트로 활용)
     * @param dt 샘플링 시간(이산화 주기)
     * @param ref_horizon 생성된 참조 상태(Reference state)들이 저장될 Np 크기의 배열 (출력용)
     */
    static void generate_slalom_reference(const StaticVector<double, Nx>& x_curr,
                                          const StaticVector<double, Nx>& x_final,
                                          const std::array<ObstacleInfo, 10>& obstacles, double dt,
                                          StaticVector<double, Nx> ref_horizon[Np]) {
        // 참조 속도(목표로 하는 일정한 이동 속도)
        double v_ref = 2.0;

        // 1. 게이트(Waypoints) 설정: 장애물 사이사이를 통과하도록 좌표 지정
        struct Point {
            double x, y;
        };
        std::vector<Point> gates;

        // 10개의 장애물을 2개씩 짝지어(페어로 묶어) 그 중앙을 통과점(게이트)으로 설정
        for (int i = 0; i < 9; i += 2) {
            gates.push_back({(obstacles[i].x + obstacles[i + 1].x) / 2.0,
                             (obstacles[i].y + obstacles[i + 1].y) / 2.0});
        }
        // 마지막 게이트로는 최종 목표 지점을 추가
        gates.push_back({x_final(0), x_final(1)});

        // 2. 현재 위치에서 가장 적합한 게이트를 추적하며 Np개의 Horizon 생성
        for (size_t k = 0; k < Np; ++k) {
            // 시간 스텝에 따른 전방 주시 거리(Lookahead distance)
            double lookahead_dist = v_ref * dt * (k + 1);
            
            // 초기 목표점은 최종 목표 위치로 설정
            double target_x = x_final(0), target_y = x_final(1);

            // 현재 위치보다 앞에 있는(x축 기준) 첫 번째 게이트를 탐색하여 타겟으로 지정
            for (auto& gate : gates) {
                if (gate.x > x_curr(0) + lookahead_dist * 0.5) {
                    target_x = gate.x;
                    target_y = gate.y;
                    break;
                }
            }

            // 타겟 게이트를 향하는 방향 벡터 및 각도(Heading angle) 계산
            double dx = target_x - x_curr(0);
            double dy = target_y - x_curr(1);
            double angle = std::atan2(dy, dx);

            // 상태 벡터 구성 (차량 모델 등을 가정한 6차원 상태 공간: [x, y, theta, v, w, a] 등)
            ref_horizon[k](0) = x_curr(0) + std::cos(angle) * lookahead_dist; // x 위치
            ref_horizon[k](1) = x_curr(1) + std::sin(angle) * lookahead_dist; // y 위치
            ref_horizon[k](2) = angle;                                        // 요각(Heading angle, theta)
            ref_horizon[k](3) = v_ref;                                        // 참조 속도
            ref_horizon[k](4) = 0.0;                                          // 각속도(or 조향각 등) 참조값 (0으로 유지)
            ref_horizon[k](5) = 0.0;                                          // 가속도 참조값 (등속 이동 가정)
        }
    }
};

}  // namespace Optimization
#endif