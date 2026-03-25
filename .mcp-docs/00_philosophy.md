# 🏛️ NMPC Architect: The Constitution (2026 Revision)

## 1. 정체성 및 비전 (Identity & Vision)
* **Persona:** 10년의 차량 동역학 직관을 최적화 알고리즘으로 집행하는 **'입법자(Legislator)'**.
* **Timeline:** 2024.03.25 넥센타이어 입사 후 **2년(731일)의 신뢰성**을 확보함. 2028년 현대차그룹(HMG) 자율주행/모션제어 핵심 부서로의 **'Value > Risk'** 점프를 목표로 한다.
* **Base Engine:** `Algorithms_for_Optimization_cpp` 리포지토리를 단순 학습용이 아닌 **'실전형 NMPC Full Stack'**으로 진화시킨다.

## 2. 핵심 설계 원칙 (Hard Constraints)
모든 코드 생성 및 리팩토링 시 에이전트는 아래 규칙을 **절대 준수**해야 한다.
* **[Memory] Zero Dynamic Allocation:** 제어 루프 내 `malloc`, `new`, `std::vector` 사용을 엄격히 금지한다. 모든 메모리는 `std::array` 기반의 **정적/스택 할당**이어야 한다.
* **[Performance] Deterministic Latency:** Worst-case 연산 시간을 보장해야 한다. 캐시 효율을 위한 **Data-Oriented Design**과 **SIMD 최적화**를 지향한다.
* **[Language] Modern C++ (C++20/23):** `concepts`, `ranges`, `std::span`을 적극 활용하여 런타임 오버헤드를 제로화한다.

## 3. 7단계 레이어 아키텍처 (7-Layer Stack)
모든 코드는 아래 레이어 중 하나에 명확히 속해야 하며, 상위 레이어는 하위 레이어의 인터페이스에만 의존한다.
* **Layer 0-2:** Scalar(Dual Number), Matrix Engine(Static), Linear Algebra Ops
* **Layer 3-4:** Integrator(RK4), **Nonlinear Solver(SQP, LM, RT-QP)**
* **Layer 5-6:** Estimator(EKF/MHE), **NMPC Controller**
* **Layer 7:** **Vehicle/Tire Dynamics (Magic Formula via LUT)**

## 4. 아키텍처 철학 (Values)
* **Soft Constraints & Slack Penalty:** 시스템이 물리적 한계에서 폭발하지 않도록 유연한 벌점 체계를 도입한다.
* **KKT Monitor:** 수렴 여부를 수치적 지표($\nabla L$, Feasibility)로 입증한다.
* **Obstacle Avoidance:** 장애물을 수학적 **Potential Field** 또는 **Constraint**로 정의하여 솔버 레벨에서 직접 해결한다.

## 5. AI 에이전트(Roo Code) 업무 수칙
* 에이전트는 코드를 제안하기 전, 해당 코드가 **'정적 메모리 원칙'**을 준수하는지 자가 검토 프로세스를 거친다.
* 기존 `Algorithms_for_Optimization_cpp` 코드를 분석하여 **NMPC 적합성**을 평가하고 개선안을 제시한다.
* 모든 결과물은 "2028년 현대차 면접관을 압도할 수 있는 수준"의 기술적 깊이를 유지한다.

---

### 🛠️ 에이전트에게 내릴 '첫 번째 명령'

위 파일을 저장하셨다면, Roo Code 채팅창에 다음과 같이 입력하여 **'락(Lock)'**을 거십시오.

> "현재 프로젝트 루트의 `.mcp-docs/00_philosophy.md`를 읽어라. 나는 오늘로 현직 2주년을 채운 **NMPC Architect**다. 앞으로 내가 요청하는 모든 코딩 작업에서 이 문서의 **정적 메모리 원칙**과 **7-Layer 구조**를 최우선으로 적용하라. 준비가 되었다면, 내 리포지토리의 최적화 알고리즘 중 하나를 골라 이 원칙에 따라 리팩토링할 첫 번째 후보를 제안하라."