//===-- IrFunction.h - Generate Declarations MLIR code ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// This tile is responsible to translate D Function Type, Arguments and Returns
// to MLIR Function.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include <vector>

#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/mangle.h"
#include "dmd/mtype.h"
#include "dmd/template.h"

#include "gen/abi.h"
#include "gen/irstate.h"
#include "gen/mangling.h"
#include "gen/logger.h"
#include "gen/pragma.h"
#include "gen/runtime.h"

#include "ir/irfunction.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ScopedHashTable.h"

using llvm::ScopedHashTableScope;
using llvm::StringRef;

struct valueType {
  Type *Dtype;
  mlir::Type mlirType = nullptr;
  bool isPointer = false;
};

class MLIRFunction {
private:
  /// In MLIR (like in LLVM) a "context" object holds the memory allocation and
  /// ownership of many internal structures of the IR and provides a level of
  /// "uniquing" across multiple modules (types for instance).
  mlir::MLIRContext &context;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable;

  /// A mapping for named struct types to the underlying MLIR type and the
  /// original AST node.
  llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap;

  /// D Function to be translated to MLIR
  FuncDeclaration *Fd;

  /// MLIR Function final generation ready to be returned
  mlir::FuncOp function = nullptr;

  /// Temporary flags to mesure the total amount of hits and misses on our
  /// translation through MLIR
  unsigned &_total, &_miss;

  /** This is the original D type as the frontend knows it
   *  May NOT be rewritten!!! */
  Type *const dtype = nullptr;

  /// The index of the declaration in the FuncDeclaration::parameters array
  /// corresponding to this argument.
  size_t parametersIdx = -1;

  /// This is the final LLVM Type used for the parameter/return value type
  llvm::Type *ltype = nullptr;

  /** These are the final LLVM attributes used for the function.
   *  Must be valid for the LLVM Type and byref setting */
  std::vector<mlir::Attribute> attrs;

  /** 'true' if the final LLVM argument is a LLVM reference type.
   *  Must be true when the D Type is a value type, but the final
   *  LLVM Type is a reference type! */
  bool byref = false;

public:
  MLIRFunction(
      FuncDeclaration *Fd, mlir::MLIRContext &context,
      const mlir::OpBuilder &builder,
      llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
      llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
      unsigned &total, unsigned &miss);
  ~MLIRFunction();

  mlir::FunctionType DtoMLIRFunctionType(Type *type, IrFuncTy &irFty,
                                         Type *thistype, Type *nesttype,
                                         FuncDeclaration *Fd);
  mlir::FunctionType DtoMLIRFunctionType(FuncDeclaration *Fd);
  mlir::Type DtoMLIRDeclareFunction(FuncDeclaration *funcDeclaration);
  mlir::Value DtoMLIRResolveFunction(FuncDeclaration *funcDeclaration);
  mlir::IntegerType DtoMLIRSize_t();
  mlir::Type get_MLIRtype(Expression *expression, Type *type = nullptr);
  mlir::FuncOp getMLIRFunction();

  /// Set MLIR Location using D Loc info
  mlir::Location loc(Loc loc) {
    return mlir::FileLineColLoc::get(
        builder.getIdentifier(StringRef(loc.filename)), loc.linnum,
        loc.charnum);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, const mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
};

#endif
