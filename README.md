# 🏎️ Zero-Alloc NMPC Architecture (IPM & Sparse-Ready)

![C++](https://img.shields.io/badge/C++-17%2B-blue.svg) ![Build](https://img.shields.io/badge/build-passing-brightgreen.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Zero-Allocation](https://img.shields.io/badge/memory-Zero_Allocation-orange.svg)

**Autonomous Vehicle Control Stack built from scratch. No external solver libraries. 100% Static Memory.**

본 프로젝트는 자율주행 차량의 실시간 궤적 생성 및 장애물 회피를 위한 **비선형 모델 예측 제어(Nonlinear Model Predictive Control, NMPC)** 스택입니다. 타이어 동역학(Vehicle Dynamics)을 최우선으로 존중하되, 이를 철저히 수치 최적화로 통제하는 '입법자(Lawgiver)'의 철학을 바탕으로 설계되었습니다.

기성 솔버(OSQP, ACADO, HPIPM 등)에 의존하지 않고, 최하단 자동 미분(Auto Differentiation) 엔진부터 최상단 SQP-IPM 솔버까지 단일 C++ 스택으로 직접 구축한 하드코어 제어 아키텍처입니다.

> ⚠️ **Current Architecture Status (Honest Engineering):**
> 현재 제어 스택은 다항 시간(Polynomial Time) 내에 60차원 이상의 하드 제약 조건을 뚫어내는 **Primal-Dual IPM(내부점 기법)**으로 성공적으로 전환되었습니다. 
> 현재 KKT 행렬 분해는 메모리 안정성이 검증된 `Dense Matrix (가우스 소거법 + Partial Pivoting)` 기반으로 구동되며, 진정한 $O(NNZ)$ 스케일링을 달성하기 위한 `정적 CSR(Compressed Sparse Row) 엔진`은 **Layer 2**에 구현 및 독립 테스트가 완료되어 IPM 스택과의 최종 통합을 앞두고 있습니다.

## ✨ Core Philosophy

1. **Zero Dynamic Allocation (정적 메모리 설계):** `std::vector`나 `new` 키워드를 철저히 배제. 힙 메모리 파편화(Heap Fragmentation)를 원천 차단하여 임베디드(RTOS) 및 차량용 MCU 환경에서의 극한의 실시간성(Deterministic Timing) 보장.
2. **Physics is the Law (물리 법칙 최우선):** 단순한 기구학(Kinematics)을 넘어, 비선형 타이어 거동을 반영한 6-State Dynamic Bicycle Model 적용.
3. **Hard Constraints via IPM:** 페널티 기반의 꼼수(Soft Constraint)를 버리고, 조향각 및 가속도의 물리적 한계치를 로그 방벽(Log Barrier)으로 완벽히 통제.

## 🏗️ System Architecture (The 7 Layers)

본 아키텍처는 외부 의존성 없이 아래의 7개 계층으로 견고하게 결합되어 있습니다.

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Layer 0** | `AutoDiff (Dual Numbers)` | 이중수(Dual Numbers) 기반 전진 모드 자동 미분 엔진. 무오차 자코비안/헤시안 추출. |
| **Layer 1** | `StaticMatrix Engine` | 템플릿 메타프로그래밍을 활용한 크기 고정형 Dense 선형대수 엔진. (Partial Pivoting 내장) |
| **Layer 2** | `SparseMatrix Engine` | **[New]** 동적 할당이 배제된 정적 CSR(Compressed Sparse Row) 포맷 및 $O(NNZ)$ SpMV 연산 모듈. |
| **Layer 3** | `RK4 Integrator` | 연속 시간의 물리 모델(ODE)을 이산화하는 4차 룽게-쿠타 적분기. |
| **Layer 4** | `Primal-Dual IPM QP` | KKT 시스템을 분해하는 내부점 기법 솔버. 특이점 폭발 방지를 위한 **티호노프 정규화(Tikhonov Regularization)** 탑재. |
| **Layer 5** | `SQP Solver` | 비선형 최적화 문제를 QP로 분해하는 순차 이차 계획법 엔진. Trust-Region Clamping 및 Damped BFGS 적용. |
| **Layer 6** | `NMPC Core` | 장애물 회피망 구축, Slew Rate Penalty(승차감 확보), Temporal Scaling(제어 근시안 극복)이 적용된 닫힌 루프 제어기. |
| **Layer 7** | `Vehicle Model` | 비선형 타이어 마찰 모델이 적용된 6-State 차량 동역학 플랜트. |

## 🚀 Key Technical Breakthroughs

### 1. Overcoming MPC Myopia (시간 축 축소 기법)
물리적 제동 거리보다 제어기의 시야(Horizon)가 짧을 때 발생하는 오버슈트(Overshoot) 파국을 막기 위해 **Temporal Scaling** 기법을 도입했습니다. 
예측 스텝($N_p$)을 15로 압축하여 $O(N^3)$ Dense 연산 부하를 8배 줄여 실시간성을 확보하고, 대신 제어 주기($dt$)를 0.2초로 확장하여 **총 3.0초의 압도적인 예측 시야**를 확보했습니다.

### 2. Singularity Defense (솔버 폭발 방어망)
하드 제약 조건을 처리할 때 곡률(Hessian)이 0에 수렴하여 조향각이 $10^{44}$ 단위로 폭발(NaN/Inf)하는 수학적 대참사를 방어하기 위해 두 겹의 방탄조끼를 입혔습니다.
* **Tikhonov Regularization:** 대각 성분에 미세한 $10^{-6}$ 정규화를 더해 행렬 특이점(Singularity) 원천 차단.
* **Trust-Region Clamping:** 솔버가 물리적 변화 한계(e.g., 스텝당 $\pm 3.0$)를 벗어나는 $\Delta U$를 제안할 경우 강제 절단.

### 3. Ride Comfort & Parking Singularity (승차감 및 주차 딜레마 극복)
단순한 '점 쫓기' 방식을 탈피하여 목적 함수에 제어 입력 변화율 페널티(Slew Rate Penalty)를 주입, Bang-Bang 제어를 소멸시키고 유체처럼 부드러운 회피 궤적을 도출했습니다. 또한 종단 상태(Terminal State)에서 비홀로노믹 제약으로 인해 제자리 선회를 하는 현상을 타파하기 위해 가중치(Cost Weight) 튜닝을 최적화했습니다.