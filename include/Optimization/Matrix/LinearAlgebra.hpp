#ifndef OPTIMIZATION_LINEAR_ALGEBRA_HPP_
#define OPTIMIZATION_LINEAR_ALGEBRA_HPP_

// ============================================================
// Layer 2: Master Facade Header
// 철저하게 역할별로 분할된 서브 모듈들을 단일 진입점으로 묶어냅니다.
// 하위 클래스는 이 파일만 include 하면 모든 연산을 사용할 수 있습니다.
// ============================================================

#include "Optimization/Matrix/LinearAlgebra_Cholesky.hpp"
#include "Optimization/Matrix/LinearAlgebra_Core.hpp"
#include "Optimization/Matrix/LinearAlgebra_LDLT.hpp"
#include "Optimization/Matrix/LinearAlgebra_LU.hpp"
#include "Optimization/Matrix/LinearAlgebra_NMPC.hpp"
#include "Optimization/Matrix/LinearAlgebra_QR.hpp"
#include "Optimization/Matrix/MathTraits.hpp"

#endif  // OPTIMIZATION_LINEAR_ALGEBRA_HPP_