//===-- MLIR/MLIRStatements.h - Generate Statements MLIR code ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Generates MLIR code for one or more D Statements and return nullptr if it
// wasn't able to identify a given statement.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "dmd/statement.h"
#include "dmd/statement.h"
#include "dmd/expression.h"

#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/MLIR/MLIRGen.h"
#include "gen/MLIR/MLIRDeclaration.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ScopedHashTable.h"

using llvm::ScopedHashTableScope;
using llvm::StringRef;

class MLIRStatements {
private:
  Module *module;

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

  /// Class to deal with all declarations.
  MLIRDeclaration declaration;

  unsigned &_total, &_miss;
  unsigned decl_total = 0, decl_miss = 0;

public:
  MLIRStatements(
      Module *m, mlir::MLIRContext &context, const mlir::OpBuilder &builder_,
      llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
      llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
      unsigned &total, unsigned &miss);
  ~MLIRStatements();
  void mlirGen(IfStatement *ifStatement);
  mlir::Value mlirGen(Statement *statement);
  mlir::Value mlirGen(ExpStatement *expStatement);
  void mlirGen(ForStatement *forStatement);
  mlir::LogicalResult mlirGen(ReturnStatement *returnStatement);
  mlir::LogicalResult genStatements(FuncDeclaration *funcDeclaration);
  std::vector<mlir::Value> mlirGen(CompoundStatement *compoundStatement);
  std::vector<mlir::Value> mlirGen(ScopeStatement *scopeStatement);

  mlir::Location loc(Loc loc) {
    return builder.getFileLineColLoc(
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

#endif // LDC_MLIR_ENABLED
