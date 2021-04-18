//===-- MLIR/Dialect.h - Generate Statements MLIR code ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the IR Dialect for D Programming Language.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
//#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Types.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/*namespace mlir {
namespace D {
namespace detail {
struct StructTypeStorage;
} // end namespace detail
} // end namespace toy
} // end namespace mlir*/

/// Include the auto-generated header file containing the declaration of the D
/// dialect.
#include "Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "Ops.h.inc"

//===----------------------------------------------------------------------===//
// D Dialect New Types
//===----------------------------------------------------------------------===//
/*
/// Create a local enumeration with all of the types that are defined by D.
namespace DTypes {
enum Types {
  Struct = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
} // end namespace ToyTypes
*/
/// This class defines the D struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
/*class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == DTypes::Struct; }

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};

} // end namespace toy
} // end namespace mlir
*/
#endif // LDC_MLIR_ENABLED
