## 1. 행렬 곱셈 (Matrix Multiplication)
Column-major 메모리 구조에서 캐시 히트율 (Cache Locality)을 극대화하기 위해 J-K-I 루프 순서를 사용한 행렬곱 $C = A \times B$ 연산입니다.

* 수식
$$C_{ij} = \sum_{k=0}^{Cols-1}A_{ik}B_{kj}$$

* 특징 : 행렬 $B$의 스칼라 원소 $B_{kj}$를 고정한 상태로, 행렬 $A$의 $k$번째 열 (Column)과 결과 행렬 $C$의 $j$번째 열을 매칭하여 누산합니다.


## 2. LU 분해 및 해법 (Doolittle Algorithm)
정방 행렬 $A$를 하삼각 행렬 $L$ (대각 원소가 1)과 상삼각 행렬 $U$로 분해합니다. $A = LU$ 구조를 통해 선형 방정식 $Ax = b$를 빠르게 풉니다. 코드에서는 메모리 절약을 위해 $A$ 행렬 자리에 $L$과 $U$를 덮어쓰는 (In-place) 방식을 사용했습니다.

* U 행렬 갱신 (상삼각)
$$U_{ij} = A_{ij} - \sum_{k=0}^{i-1}L_{ik}U_{kj} \quad (j\geq i)$$

* L 행렬 갱신 (하삼각)
$$L_{ji} = \frac{1}{U_{ii}}(A_{ji} - \sum_{k=0}^{i-1}L_{jk}U_{ki}) \quad (j > i)$$

* 해법 (Forward & Backward Substitution)
1. $Ly = b$ (Forward Substitution)  
전진 대입은 대각선을 기준으로 왼쪽 아래에만 숫자가 있는 **하삼각 행렬 (Lower Triangular Matrix)**을  풀 때 사용합니다. 방정식은 $Ly = b$ 형태를 가집니다.

**수학적 전개**  
행렬식을 풀어 쓰면 다음과 같습니다
$$L_{00}y_{0} = b_{0}$$
$$L_{10}y_{0} + L_{11}y_{1} = b_{1}$$
$$L_{20}y_{0} + L_{21}y_{1} + L_{22}y_{2} = b_{2}$$

위에서부터 아래로 $(i = 0 \rightarrow N - 1)$ 순차적으로 해를 구할 수 있습니다. $y_{0}$는 첫 번째 식에서 바로 나오고, 그 값을 두 번째 식에 대입하여 $y_{1}$을 구하는 식입니다.

일반식은 다음과 같습니다.
$$y_{i} = \frac{1}{L_{ii}}(b_{i} - \sum_{j=0}^{i-1})L_{ij}y_{j}$$

Doolittle LU 분해에서는 $L$의 대각 원소 $L_{ii} = 1$이므로 나눗셈이 생략됩니다.

`LU_solve` 기준
```c++
// Forward Substitution : Ly = b
for (size_t i = 0; i < Rows; ++i) {
    T sum = static_cast<T>(0);
    for (size_t j = 0; j < i; ++j) {
        sum += (*this)(static_cast<int>(i), static_cast<int>(j)) * y(j);
    }
    y(i) = b(i) - sum;  // L_{ii}가 1이므로 나눗셈 생략
}
```
* 특징 : 이미 계산된 과거의 결과 (`y(j)`)를 누적하여 현재의 답 (`y(i)`)을 확정 짓습니다.

2. $Ux = y$ (Backward Substitution)
후진 대입은 대각선을 기준으로 오른쪽 위에만 숫자가 있는 **상삼각 행렬 (Upper Triangular Matrix)** 을 풀 때 사용합니다. 방정식은 $Ux = y$ 형태를 가집니다.

**수학적 전개**    
방정식의 맨 아래쪽부터 보면 다음과 같습니다
$$U_{N-1, N-1}x_{N-1} = y_{N-1}$$
$$U_{N-2, N-2}x_{N-2} + U_{N-2, N-1}x_{N-1} = y_{N-2}$$

이번에는 맨 아래 $(i = N-1 \rightarrow 0)$에서 시작하여 거꾸로 위로 올라가며 해를 구합니다.

일반식은 다음과 같습니다.
$$x_{i} = \frac{1}{U_{ii}}(y_{i} - \sum_{j=i+1}^{N-1}U_{ij}x_{j})$$

`LU_solve` 기준
```c++
// Back substitution : Ux = y
for (int i = static_cast<int>(Rows) - 1; i >= 0; --i) {
    T sum = static_cast<T>(0);
    for (size_t j = static_cast<size_t>(i) + 1; j < Cols; ++j) {
        sum += (*this)(i, static_cast<int>(j)) * x(j);
    }
    x(static_cast<size_t>(i)) = (y(static_cast<size_t>(i)) - sum) / (*this)(i, i);
}
```

## 3. Cholesky 분해 $(A = LL^{T})$
행렬이 대칭 양의 정부호 (Symmetric Positive Definite)일 때 사용하는 분해법입니다. NMPC 최적화 (SQP, Newton Method) 과정에서 헤시안 (Hessian) 행렬의 역행렬을 구할 때 주로 사용되며, LU 분해보다 연산량이 절반으로 줄어듭니다.

* 대각 원소 계산
$$L_{jj} = \sqrt{A_{jj} - \sum_{k=0}^{j-1}}L^{2}_{jk}$$

* 비대각 원소 계산
$$L_{ij} = \frac{1}{L_{jj}}(A_{ij} - \sum_{k=0}^{j-1}L_{ik}L_{jk}) \quad (i > j)$$

1. $Ly = b$ (Forward Substitution)
LU 분해의 하삼각 행렬과 같지만, **대각 원소가 1이 아니라는 점**이 다릅니다. 따라서 계산 후 반드시 $L_{ii}$로 나누어 주어야 합니다.  

$$y_{i} = \frac{1}{L_{ii}}(b_{i} - \sum_{k=0}^{i-1}L_{ik}y_{k})$$

2. $L^T x = y$ (Backward Substitution)
수식으로는 상삼각 행렬을 푸는 것과 같지만, 메모리 상에는 상삼각 행렬이 따로 없고 오직 $L$만 존재합니다. 따라서 $L$의 행과 열 인덱스를 뒤집어서 $(L_{ki})$ 참조해야 합니다.

$$x_{i} = \frac{1}{L_{ii}}(y_{i} - \sum_{k=i+1}^{Rows-1}L_{ki}x_{k})$$


## 4. LDLT 분해
제곱근 연산은 CPU 사이클을 많이 소모하므로, 이를 피하기 위해 대각 행렬 $D$를 도입한 $A = LDL^T$ 분해입니다.
* 대각 행렬 $D$ 갱신
$$D_{jj} = A_{jj} - \sum_{k=0}^{j-1}L_{jk}^{2}D_{kk}$$

* 하삼각 행렬 $L$ 갱신
$$L_{ij} = \frac{1}{D_{jj}}(A_{ij} - \sum_{k=0}^{j-1}L_{ik}L_{jk}D_{kk}) \quad (i > j)$$

* 해법 : $Lz = b, Dy = z, L^Tx = y$ 순서로 전진 및 후진 대입을 수행합니다.

1. $Lz = b$ (Forward Substitution)
여기서 $L$은 대각 원소가 무조건 1입니다. 따라서 나눗셈이 발생하지 않습니다
$$z_{i} = b_{i} - \sum_{k=0}^{i-1} L_{ik}z_{k}$$

2. $Dy = z$ (Diagonal Scaling)
가장 단순한 단계입니다. 대각 행렬 $D$의 원소로 나누어 스케일을 맞춥니다
$$y_{i} = \frac{z_i}{D_{ii}}$$

3. $L^T = y$ (Backward Substitution)
대각 원소가 1이므로 나눗셈 없이 뺄셈과 곱셈만으로 끝납니다
$$x_{i} = y_{i} - \sum_{k=i+1}^{Rows-1}L_{ki}x_{k}$$

## 5. QR 분해
행렬 $A$를 직교 행렬 $Q (Q^TQ = I)$와 상삼각 행렬 $R$로 분해합니다. 최소제곱법 (Least Squares)이나 제어 시스템의 MHE (Moving Horizon Estimation)에서 과결정 시스템 (Overdetermined System, $Rows \geq Cols$)을 풀 때 수치적으로 가장 안정적인 방법입니다.

### A. Modified Gram-Schmidt (MGS)
원래의 Gram-Schmidt 직교화 과정에서 발생하는 수치적 오차 누적을 줄이기 위해 순서를 변경한 알고리즘입니다.
* R 행렬 대각 원소 (Norm 계산)
$$R_{ii} = \sqrt{\sum_{k=0}^{Rows-1}A_{ki}^{2}}$$

* Q 행렬 벡터 정규화
$$Q_{*i} = \frac{A_{*i}}{R_{ii}}$$

* 나머지 열 직교화 투영
$$R_{ij} = Q^{T}_{*i}A_{*j}$$
$$A_{*j} \leftarrow A_{*j} - R_{ij}Q_{*i}$$

### B. Householder Reflection
MGS보다 연산량은 약간 많지만 수치적 안정성이 극대화된 방법입니다. 거울에 반사시키듯 벡터를 회전시켜 삼각 행렬을 만듭니다.
* Householder 벡터 $v$ 및 $\tau$ 계산
$$v = A_{*i} + sign(A_{ii})||A_{*i}|| e_{1}$$
$$\tau = \frac{2}{||v||^{2}}$$

* 행렬 A 업데이트 (반사 적용)
$$A \leftarrow (I - \tau v v^T)A$$