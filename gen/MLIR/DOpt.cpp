//===-- DOpt.cpp - Toy High Level Optimizer -------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the D dialect.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "Dialect.h"
#include <numeric>
using namespace mlir;
using namespace D;

/// Fold integers.
OpFoldResult IntegerOp::fold(ArrayRef<Attribute> operands) { return value(); }

/// Fold floats.
OpFoldResult FloatOp::fold(ArrayRef<Attribute> operands) { return value(); }

/// Fold doubles.
OpFoldResult DoubleOp::fold(ArrayRef<Attribute> operands) { return value(); }

/// Fold struct constants.
/*OpFoldResult StructConstantOp::fold(ArrayRef<Attribute> operands) {
  return value();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
  auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index().getZExtValue();
  return structAttr.getValue()[elementIndex];
}
*/
#endif
