## EQP 솔버의 수학적 정의
우리가 풀고자 하는 서스펜션 (또는 차량 거동)의 등식 제약 최적화 문제는 다음과 같습니다
* 목적 함수 (Mimize)
$$J(u) = \frac{1}{2}u^{T}Pu + q^{T}u$$

* 등식 제약 조건 (Subject to)
$$Au = b$$

이를 풀기 위해 라그랑주 승수 (Lagrange Multiplier) $\lambda$를 도입하여 미분하면, 우리가 풀어야 할 최종 선형 방정식인 **KKT (Karush-Kuhn-Tucker) 시스템**이 도출됩니다.

$$\begin{bmatrix} P & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} u \\ \lambda \end{bmatrix} = \begin{bmatrix} -q \\ b \end{bmatrix}$$

---
### Block Operation
1. `insert_block` : $P$, $A$ 같은 부분 행렬을 지정된 위치에 삽입합니다.
2. `insert_transposed_block` : KKT 행렬 우상단에 들어갈 **$A^T$를 위해 임시 행렬 (Temporary Matrix)을 생성하지 않고 바로 삽입하는 Zero-copy 함수**입니다.