//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
#if LDC_MLIR_ENABLED

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "gen/MLIR/Dialect.h"
#include "gen/logger.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::D;



/*Operation *DDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
  Type originalType = type;
  Operation *op = nullptr;

  if (type.isa<TensorType>())
    type = type.cast<RankedTensorType>().getElementType();

  if (type.isa<StructType>())
    op = builder.create<StructConstantOp>(loc, originalType,
                                          value.cast<ArrayAttr>());
  else if (type.isF16() || type.isF32())
    op = builder.create<D::FloatOp>(loc, originalType,
                                    value.cast<DenseElementsAttr>());
  else if (type.isF64())
    op = builder.create<D::DoubleOp>(loc, originalType,
                                     value.cast<DenseElementsAttr>());
  else if (type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
           type.isInteger(32) || type.isInteger(64) || type.isInteger(128))
    op = builder.create<D::IntegerOp>(loc, originalType,
                                      value.cast<DenseElementsAttr>());

  return op;
}*/

//===----------------------------------------------------------------------===//
// D Operations
//===----------------------------------------------------------------------===//

/// Verify that the given attribute value is valid for the given type.
/*static LogicalResult verifyConstantForType(Type type, Attribute opaqueValue,
                                           Operation *op) {
  if (type.isa<TensorType>()) {
    // Check that the value is a elements attribute.
    auto attrValue = opaqueValue.dyn_cast<DenseElementsAttr>();
    if (!attrValue)
      return op->emitError("constant of TensorType must be initialized by "
                           "a DenseElementsAttr, got ")
             << opaqueValue;

    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = type.dyn_cast<RankedTensorType>();
    if (!resultType)
      return success();

    // Check that the rank of the attribute type matches the rank of the
    // constant result type.
    auto attrType = attrValue.getType().cast<TensorType>();
    if (attrType.getRank() != resultType.getRank()) {
      return op->emitOpError("return type must match the one of the attached "
                             "value attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        return op->emitOpError(
                   "return type shape mismatches its attribute at dimension ")
               << dim << ": " << attrType.getShape()[dim]
               << " != " << resultType.getShape()[dim];
      }
    }
    return success();
  }
  auto resultType = type.cast<StructType>();
  llvm::ArrayRef<Type> resultElementTypes = resultType.getElementTypes();

  // Verify that the initializer is an Array.
  auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitError("constant of StructType must be initialized by an "
                         "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid.
  llvm::ArrayRef<Attribute> attrElementValues = attrValue.getValue();
  for (const auto &it : llvm::zip(resultElementTypes, attrElementValues))
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return failure();
  return success();
}

static LogicalResult verify(StructConstantOp op) {
  return verifyConstantForType(op.getResult().getType(), op.value(), op);
}

//===----------------------------------------------------------------------===//
// StructAccessOp

void StructAccessOp::build(Builder *b, OperationState &state, Value input,
                           size_t index) {
  // Extract the result type from the input type.
  StructType structTy = input.getType().cast<StructType>();
  assert(index < structTy.getNumElementTypes());
  Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b->getI64IntegerAttr(index));
}

static LogicalResult verify(StructAccessOp op) {
  StructType structTy = op.input().getType().cast<StructType>();
  size_t index = op.index().getZExtValue();
  if (index >= structTy.getNumElementTypes())
    return op.emitOpError()
           << "index should be within the range of the input struct type";
  Type resultType = op.getResult().getType();
  if (resultType != structTy.getElementTypes()[index])
    return op.emitOpError() << "must have the same result type as the struct "
                               "element referred to by the index";
  return success();
}

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace D {
namespace detail {
/// This class represents the internal storage of the D `StructType`.
struct StructTypeStorage : public TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<Type> elementTypes;
};
} // end namespace detail
} // end namespace toy
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first two parameters are the context to unique in and the
  // kind of the type. The parameters after the type kind are forwarded to the
  // storage instance.
  MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, DTypes::Struct, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

/// Parse an instance of a type registered to the toy dialect.
Type DDialect::parseType(DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<Type, 1> elementTypes;
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<TensorType>() && !elementType.isa<StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void DDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
*/
//===----------------------------------------------------------------------===//
// Dialect Ops

/*void D::SubOp::build(Builder *b, OperationState &state, Value lhs, Value
 * rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::SubFOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void MulOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::MulFOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::DivFOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::DivSOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::DivUOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::ModSOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::ModUOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::ModFOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::AndOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::OrOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::XorOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  if (lhs.getType() == rhs.getType())
    state.addTypes(lhs.getType());
  else
    state.addTypes(NoneType::get(b->getContext()));
  state.addOperands({lhs, rhs});
}

void D::CallOp::build(Builder *b, OperationState &state, llvm::StringRef callee,
                      llvm::ArrayRef<Type> types,
                      llvm::ArrayRef<Value> arguments) {
  state.addTypes(types);
  state.addOperands(arguments);
  state.addAttribute("callee", b->getSymbolRefAttr(callee));
}

void D::IntegerOp::build(Builder *builder, OperationState &state, Type type,
                         int value, int size = 0) {
  if (type.isInteger(size)) {
    auto dataType = builder->getIntegerType(size);
    auto shapedType = RankedTensorType::get({}, dataType);
    auto dataAttribute = DenseElementsAttr::get(shapedType, value);
    IntegerOp::build(builder, state, dataAttribute);
  } else {
    IF_LOG Logger::println("Unable to get the Attribute for %d", value);
  }
}

void D::FloatOp::build(Builder *builder, OperationState &state, Type type,
                       float value) {
  if (type.isF16()) {
    auto shapedType = RankedTensorType::get({}, builder->getF16Type());
    auto dataAttribute = DenseElementsAttr::get(shapedType, value);
    FloatOp::build(builder, state, dataAttribute);
  } else if (type.isF32()) {
    auto shapedType = RankedTensorType::get(1, builder->getF32Type());
    auto dataAttribute = DenseElementsAttr::get(shapedType, value);
    FloatOp::build(builder, state, dataAttribute);
  } else {
    IF_LOG Logger::println("Unable to get the Attribute for %f", value);
  }
}

void D::DoubleOp::build(Builder *builder, OperationState &state, Type type,
                        double value) {
  if (type.isF64()) {
    auto shapedType = RankedTensorType::get(1, builder->getF64Type());
    auto dataAttribute = DenseElementsAttr::get(shapedType, value);
    FloatOp::build(builder, state, dataAttribute);
  } else {
    IF_LOG Logger::println("Unable to get the Attribute for %f", value);
  }
}*/

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// DDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void DDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();
  //addTypes<StructType>();
}

#endif // LDC_MLIR_ENABLED
