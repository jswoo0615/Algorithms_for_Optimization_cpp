# 🏎️ Zero-Alloc RT-NMPC Architecture (RTI & Sparse-Ready)

![C++](https://img.shields.io/badge/C++-17%2B-blue.svg) ![Real-Time](https://img.shields.io/badge/latency-Bounded_WCET-brightgreen.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Zero-Allocation](https://img.shields.io/badge/memory-Zero_Allocation-orange.svg)

**Hard Real-Time Autonomous Vehicle Control Stack. No external solvers. 100% Static Memory & Bounded Latency.**

본 프로젝트는 자율주행 차량의 실시간 궤적 생성 및 장애물 회피를 위한 **실시간 비선형 모델 예측 제어(Real-Time Iteration NMPC)** 스택입니다. 타이어 동역학과 물리적 한계를 최우선으로 존중하며, 기성 솔버(OSQP, ACADO)에 의존하지 않고 바닥부터 C++로 직접 구축한 하드코어 제어 아키텍처입니다.

> ⚠️ **Current Architecture Status (Honest Engineering):**
> 본 시스템은 가변 실행 시간을 유발하는 고전적 SQP(Sequential Quadratic Programming) 구조를 완전히 폐기하고, **RTI(Real-Time Iteration)** 구조로 전환하여 **WCET(Worst-Case Execution Time) 50ms 이하의 확정적(Deterministic) 실시간성**을 달성했습니다. 
> 현재 KKT 시스템은 Dense Matrix 기반으로 구동되며, 진정한 O(NNZ) 스케일링을 위한 `정적 CSR(Compressed Sparse Row) 엔진 (Layer 2)`은 독립 구현 및 테스트가 완료되어 최종 IPM 스택과의 통합을 앞두고 있습니다.

## ✨ Core Philosophy (The NVIDIA Way)

1. **Bounded Latency (시간 통제권):** 자율주행 제어기에서 가장 치명적인 것은 '지연(Jitter)'입니다. Line Search와 가변 반복 루프를 완전히 뜯어내고, 매 제어 주기마다 단 1회의 최적화 스텝(Single-shot)만 수행하여 실행 시간을 O(1) 상수로 완벽히 고정했습니다.
2. **True Mathematics (가짜 근사 배제):** 스칼라 목적 함수 기반의 어설픈 BFGS 헤시안 업데이트를 버리고, 306차원의 거대한 잔차 벡터(Residual Vector)를 자동 미분하여 완벽한 **True Gauss-Newton Hessian ($J^T J$)**을 실시간으로 구축합니다.
3. **Zero Dynamic Allocation:** `std::vector`나 `new` 키워드를 철저히 배제. 힙 메모리 파편화를 원천 차단하여 임베디드(RTOS) 환경에서의 생존력을 보장합니다.

## 🏗️ System Architecture (The 7 Layers)

본 아키텍처는 외부 라이브러리 없이 아래의 7개 계층으로 견고하게 결합되어 있습니다.

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Layer 0** | `AutoDiff (Dual Numbers)` | 이중수(Dual Numbers) 기반 전진 모드 자동 미분 엔진. 306차원 잔차 벡터에 대한 무오차 야코비안 추출. |
| **Layer 1** | `StaticMatrix Engine` | 템플릿 메타프로그래밍을 활용한 크기 고정형 Dense 선형대수 엔진. (Partial Pivoting 내장) |
| **Layer 2** | `SparseMatrix Engine` | **[Pending Integration]** 동적 할당이 배제된 정적 CSR(Compressed Sparse Row) 포맷 및 O(NNZ) SpMV 연산 모듈. |
| **Layer 3** | `RK4 Integrator` | 연속 시간의 물리 모델(ODE)을 이산화하는 4차 룽게-쿠타 적분기. |
| **Layer 4** | `Fixed-Iter IPM QP` | KKT 시스템을 분해하는 내부점 기법 솔버. 실행 시간 고정을 위해 반복 횟수(15회)를 엄격히 제한. |
| **Layer 5** | `RTI Solver` | **[Core]** SQP를 대체하는 실시간 전용 솔버. Line Search 배제, LM(Levenberg-Marquardt) Damping을 통한 Trust-Region 구축. |
| **Layer 6** | `RT-NMPC Core` | 306차원 잔차(Residual) 함수 구축, 60차원 하드 제약 조건 주입 및 Temporal Scaling(시야 확장)이 적용된 닫힌 루프 제어기. |
| **Layer 7** | `Vehicle Model` | 6-State 차량 동역학 플랜트 (Dynamic Bicycle Model). |

## 🚀 Key Technical Breakthroughs

### 1. Real-Time Iteration (RTI) & WCET Profiling
최적해를 찾을 때까지 무한정 루프를 도는 것은 연구실 논문에서나 가능한 일입니다. 본 스택은 `while`문과 `Line Search`를 완전히 제거하고, 매 0.2초(dt)마다 이전 상태를 웜-스타트(Warm-start) 삼아 단 한 번의 강력한 전진만을 수행합니다. 내장된 고정밀 프로파일러를 통해 **최대 실행 시간(WCET)이 45ms 수준에 불과함**을 증명했습니다.

### 2. True Gauss-Newton Hessian ($J^T J$)
상태 오차, 제어 입력, 제어 변화율, 장애물 회피망을 모두 분해하여 총 **306차원의 잔차 벡터(Residual Vector)**를 정의했습니다. 이를 통해 $306 \times 30$ 크기의 꽉 찬 야코비안 행렬($J$)을 추출하고, 오차 없는 정밀한 헤시안($H \approx J^T J$)을 실시간으로 구축하여 솔버의 수렴성을 극대화했습니다.

### 3. Levenberg-Marquardt Damping (가상 브레이크)
Line Search라는 브레이크를 제거한 RTI 구조에서, 차량이 장애물을 만났을 때 목적 함수의 기울기가 폭발하여 제어 입력이 요동치는(Bang-Bang) 파국을 막기 위해 헤시안 대각 성분에 강력한 댐핑($\lambda I$)을 주입했습니다. 이를 통해 수치적으로 완벽한 Trust-Region(신뢰 영역)을 형성하여 진동 없는 S자 회피 기동을 완성했습니다.