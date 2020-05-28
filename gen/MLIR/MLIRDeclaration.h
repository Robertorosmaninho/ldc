//==-- MLIR/MLIRDeclaration.h - Generate Declarations MLIR code --*- C++ -*-==//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Generates MLIR code for one or more D Declarations and return nullptr if it
// wasn't able to identify a given declaration.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "dmd/statement.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/init.h"
#include "dmd/template.h"
#include "dmd/visitor.h"

#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/pragma.h"
#include "gen/MLIR/Dialect.h"
#include "gen/MLIR/IrFunction.h"
#include "gen/MLIR/MLIRGen.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

using llvm::StringRef;
using llvm::ScopedHashTableScope;

class MLIRDeclaration {
private:
  Module *module;
  mlir::Value *stmt = nullptr;

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

  /// Temporary flags to mesure the total amount of hits and misses on our
  /// translation through MLIR
  unsigned &_total, &_miss;

public:
  MLIRDeclaration(
      Module *m, mlir::MLIRContext &context, mlir::OpBuilder builder,
      llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
      llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
      unsigned &total, unsigned &miss);
  ~MLIRDeclaration();

  mlir::DenseElementsAttr getConstantAttr(Expression *expression);
  std::pair<mlir::ArrayAttr, mlir::Type>
  getConstantAttr(StructLiteralExp *structLiteralExp);
  llvm::Optional<size_t> getMemberIndex(Expression* expression);
  StructDeclaration* getStructFor(Expression *expression);
  mlir::Value mlirGen(Declaration* declaration);
  mlir::LogicalResult mlirGen(StructDeclaration* structDeclaration, bool generated);
  mlir::Value mlirGen(VarDeclaration* varDeclaration);

  static mlir::Value DtoAssignMLIR(mlir::Location Loc, mlir::Value lhs,
      mlir::Value rhs, StringRef lhs_name, StringRef rhs_name, int op,
      bool canSkipPostblitm, Type* t1, Type* t2);
  mlir::Value DtoMLIRSymbolAddress(mlir::Location loc, Type* type,
      Declaration* declaration);
  mlir::Type get_MLIRtype(Expression* expression, Type* type = nullptr);

  //Expression
  mlir::Value mlirGen(AddExp *addExp = nullptr, AddAssignExp *addAssignExp = nullptr);
  mlir::Value mlirGen(AndExp *andExp = nullptr, AndAssignExp *andAssignExp = nullptr);
  mlir::Value mlirGen(ArrayLiteralExp *arrayLiteralExp);
  mlir::Value mlirGen(AssignExp *assignExp); //Not perfect yet
  mlir::Value mlirGen(CallExp *callExp);
  mlir::Value mlirGen(CastExp *castExp);
  mlir::Value mlirGen(ConstructExp *constructExp);
  mlir::Value mlirGen(DeclarationExp* declarationExp);
  mlir::Value mlirGen(DivExp *divExp = nullptr, DivAssignExp *divAssignExp = nullptr);
  mlir::Value mlirGen(DotVarExp *dotVarExp);
  mlir::Value mlirGen(Expression *expression, int func);
  mlir::Value mlirGen(Expression *expression, mlir::Block *block = nullptr);
  mlir::Value mlirGen(IntegerExp *integerExp);
  mlir::Value mlirGen(MinExp *minExp = nullptr, MinAssignExp *minAssignExp = nullptr);
  mlir::Value mlirGen(ModExp *modExp = nullptr, ModAssignExp *modAssignExp = nullptr);
  mlir::Value mlirGen(MulExp *mulExp = nullptr, MulAssignExp *mulAssignExp = nullptr);
  mlir::Value mlirGen(OrExp *orExp = nullptr, OrAssignExp *orAssignExp = nullptr);
  mlir::Value mlirGen(PostExp *postExp);
  mlir::Value mlirGen(RealExp *realExp);
  mlir::Value mlirGen(StringExp *stringExp);
  mlir::Value mlirGen(StructLiteralExp* structLiteralExp);
  mlir::Value mlirGen(VarExp *varExp);
  mlir::Value mlirGen(XorExp *xorExp = nullptr, XorAssignExp *xorAssignExp = nullptr);
  void mlirGen(TemplateInstance *templateInstance);

  ///Set MLIR Location using D Loc info
  mlir::Location loc(Loc loc){
    return builder.getFileLineColLoc(builder.getIdentifier(
        StringRef(loc.filename)),loc.linnum, loc.charnum);
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
