//===-- MLIRStatments.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "mlir/Dialect/StandardOps/Ops.h"
#include "MLIRStatements.h"

namespace llvm{
using llvm::StringRef;
}

MLIRStatements::MLIRStatements(
    Module *m, mlir::MLIRContext &context, const mlir::OpBuilder &builder_,
    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
    llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
    unsigned &total, unsigned &miss)
    : module(m), context(context), builder(builder_), symbolTable(symbolTable),
      structMap(structMap),
      declaration(MLIRDeclaration(m, context, builder_, symbolTable, structMap,
                                  decl_total, decl_miss)),
      _total(total), _miss(miss) {}

MLIRStatements::~MLIRStatements() = default; //Default Destructor

mlir::Value MLIRStatements::mlirGen(ExpStatement *expStmt) {
  IF_LOG Logger::println("MLIRCodeGen: ExpStatement to MLIR: '%s'",
                         expStmt->toChars());
  LOG_SCOPE

  if (DeclarationExp *decl_exp = expStmt->exp->isDeclarationExp()) {
    return declaration.mlirGen(decl_exp, builder.getInsertionBlock());
  } else if (Expression *e = expStmt->exp) {
    return declaration.mlirGen(e, builder.getInsertionBlock());
  } else {
    _miss++;
    IF_LOG Logger::println("Unable to recoganize: '%s'",
                           expStmt->exp->toChars());
    return nullptr;
  }

}

void MLIRStatements::mlirGen(ForStatement *forStatement) {
  IF_LOG Logger::println("MLIRCodeGen: ForStatement to MLIR: '%s'",
                         forStatement->toChars());
  LOG_SCOPE

  mlir::Location location = loc(forStatement->loc);

  // When we create an block mlir automatically change the insert point, but
  // we have to keep it to insert the if operation inside it's own block an
  // then we can write on each successor block.
  mlir::Block *insert = builder.getInsertionBlock();

  mlir::Block *condition =
      builder.createBlock(insert->getParent(), insert->getParent()->end());
  mlir::Block *forbody = builder.createBlock(condition);
  mlir::Block *increment = builder.createBlock(forbody);

  mlir::Block *endfor = builder.createBlock(condition);

  mlir::ValueRange args = {}; //operands
  // Writing a branch instruction on predecessor of condition block
  builder.setInsertionPointToEnd(insert);
  builder.create<mlir::BranchOp>(location, condition, args);

  builder.setInsertionPointToStart(condition);
  // Getting Value for Condition
  mlir::Value cond;
  //mlir::CmpIOp cmpIOp;
  if (forStatement->condition)
    cond = declaration.mlirGen(forStatement->condition, condition);

  //Writing a branch instruction on predecessor of condition block
  builder.setInsertionPointToEnd(condition);
  builder.create<mlir::CondBranchOp>(location, cond, forbody, args, endfor, args);

  builder.setInsertionPointToStart(forbody);
  if(auto body = forStatement->_body)
    mlirGen(body);

  //Writing a branch instruction on predecessor of condition block
  //builder.setInsertionPointToEnd(condition);
  builder.create<mlir::BranchOp>(location, increment, args);

  builder.setInsertionPointToStart(increment);

  if(auto inc = forStatement->increment)
    declaration.mlirGen(inc, increment);

  //Writing a branch instruction on predecessor of condition block
  //builder.setInsertionPointToEnd(condition);
  builder.create<mlir::BranchOp>(location, condition, args);

  builder.setInsertionPointToStart(endfor);

}

void MLIRStatements::mlirGen(IfStatement *ifStatement){
  IF_LOG Logger::println("MLIRCodeGen: IfStatement to MLIR: '%s'",
                         ifStatement->toChars());
  LOG_SCOPE

  unsigned if_total = 0, if_miss = 0;
  // Builing the object to get the Value for an expression
  MLIRDeclaration mlirDeclaration(module, context, builder, symbolTable,
                                  structMap, if_total, if_miss);

  // Marks if a new direct branch is needed. This happens when we need to
  // connect the end_if of an "else if" into the his successor end_if
  bool gen_new_br = false;

  //Getting Value for Condition
  mlir::Value cond = mlirDeclaration.mlirGen(ifStatement->condition);

  mlir::Location location = loc(ifStatement->loc);

  // When we create an block mlir automatically change the insert point, but
  // we have to keep it to insert the if operation inside it's own block an
  // then we can write on each successor block.
  mlir::Block *insert = builder.getInsertionBlock();

  // Creating two blocks if, else and end
  mlir::Block *if_then = builder.createBlock(cond.getParentRegion(),
                                             cond.getParentRegion()->end());
  mlir::Block *if_else = nullptr;
  if(ifStatement->elsebody)
    if_else = builder.createBlock(cond.getParentRegion(),
                                  cond.getParentRegion()->end());
  mlir::Block *end_if = builder.createBlock(cond.getParentRegion(),
                                            cond.getParentRegion()->end());

  // Getting back to the old insertion point
  builder.setInsertionPointAfter(&insert->back());

  // TODO: Make args to block generic -> phi nodes
  mlir::ValueRange args = {}; // Args to block
  if(ifStatement->elsebody)
    builder.create<mlir::CondBranchOp>(location, cond, if_then, args, if_else,args);
  else
    builder.create<mlir::CondBranchOp>(location, cond, if_then, args, end_if,args);


  // After create the branch operation we can fill each block with their
  // operations
  builder.setInsertionPointToStart(if_then);
  if (ExpStatement *expStatement = ifStatement->ifbody->isExpStatement())
    mlirGen(expStatement);
  else if (ScopeStatement *scopeStatement =
      ifStatement->ifbody->isScopeStatement())
    mlirGen(scopeStatement);
  else if (IfStatement *nested_if = ifStatement->ifbody->isIfStatement())
    mlirGen(nested_if);
  else if (ReturnStatement *Return = ifStatement->ifbody->isReturnStatement())
    mlirGen(Return);
  else
    _miss++;

  //Writing a branch instruction on each block (if, else) to (end)

  builder.create<mlir::BranchOp>(location, end_if, args); //args = {}

  if (ifStatement->elsebody){
    builder.setInsertionPointToStart(if_else);
    if (ExpStatement * expStatement = ifStatement->elsebody->isExpStatement())
      mlirGen(expStatement);
    else if (ScopeStatement *scopeStatement =
        ifStatement->elsebody->isScopeStatement())
      auto _result = mlirGen(scopeStatement);
    else if (IfStatement* elseif = ifStatement->elsebody->isIfStatement()){
      gen_new_br = true;
      mlirGen(elseif);
    }
    else
    _miss++;
  }

  if (gen_new_br){
    builder.create<mlir::BranchOp>(location, end_if, args); //args = {}
  } else if (ifStatement->elsebody) {
    builder.setInsertionPointToEnd(if_else);
    builder.create<mlir::BranchOp>(location, end_if, args); //args = {}
  }

  //Setting the insertion point to the block before if_then and else
  builder.setInsertionPointToStart(end_if);

  _total += if_total;
  _miss += if_miss;

}

mlir::LogicalResult MLIRStatements::mlirGen(ReturnStatement *returnStatement){
  IF_LOG Logger::println("MLIRCodeGen - Return Stmt: '%s'",
                         returnStatement->toChars());
  LOG_SCOPE

  mlir::Location location = loc(returnStatement->loc);

  if(returnStatement->exp->hasCode()) {
    auto expr = declaration.mlirGen(returnStatement->exp, builder.getInsertionBlock());
    if(!expr)
      return mlir::failure();
    auto returnOp = builder.create<mlir::ReturnOp>(location, expr.getType(),
                                                   mlir::ValueRange(expr));

    //Assuming that the function only returns one value
    returnOp.setOperand(0, expr);
  }else{
    builder.create<mlir::ReturnOp>(location);
  }

  return mlir::success();
}

std::vector<mlir::Value> MLIRStatements::mlirGen(CompoundStatement *compoundStatement){
  IF_LOG Logger::println("MLIRCodeGen - CompundStatement: '%s'",
                         compoundStatement->toChars());
  LOG_SCOPE

  std::vector<mlir::Value> arrayValue;

  for (auto stmt : *compoundStatement->statements){
    _total++;
    if (CompoundStatement *compoundStatement = stmt->isCompoundStatement()) {
      arrayValue = mlirGen(compoundStatement); // Try again
    } else if (ExpStatement *expStmt = stmt->isExpStatement()) {
      arrayValue.push_back(mlirGen(expStmt));
    } else if (ReturnStatement *returnStatement = stmt->isReturnStatement()){
      mlirGen(returnStatement);
    } else if (IfStatement *ifStatement = stmt->isIfStatement()) {
      mlirGen(ifStatement);
    } else if (ForStatement *forStatement = stmt->isForStatement()) {
      mlirGen(forStatement);
    } else if (UnrolledLoopStatement *unrolledLoopStatement =
                                              stmt->isUnrolledLoopStatement()) {
      mlirGen(unrolledLoopStatement);
    } else if (ScopeStatement *scopeStatement = stmt->isScopeStatement()){
      mlirGen(scopeStatement->statement->isCompoundStatement());
    } else {
      _miss++;
      IF_LOG Logger::println("Statament doesn't match with any implemented "
                             "CompoundStatement implemented: '%s' : "
                             "'%hhu'", stmt->toChars(), stmt->stmt);
    }
  }
  return arrayValue;
}

std::vector<mlir::Value> MLIRStatements::mlirGen(ScopeStatement *scopeStatement){
  IF_LOG Logger::println("MLIRCodeGen - ScopeStatement: \n'%s'",
                         scopeStatement->toChars());
  LOG_SCOPE
  std::vector<mlir::Value> arrayValue;

  if (auto *compoundStatement = scopeStatement->statement->isCompoundStatement()) {
    arrayValue = mlirGen(compoundStatement);
  } else if (ExpStatement* expStatement =
      scopeStatement->statement->isExpStatement()) {
    arrayValue.push_back(mlirGen(scopeStatement->statement->isExpStatement()));
  } else if (IfStatement *ifStatement = scopeStatement->statement->isIfStatement()) {
    mlirGen(ifStatement);
  } else if (ForStatement *forStatement = scopeStatement->statement->isForStatement()) {
    mlirGen(forStatement);
  } else if (UnrolledLoopStatement *unrolledLoopStatement =
      scopeStatement->statement->isUnrolledLoopStatement()){
    mlirGen(unrolledLoopStatement);
  } else {
    _miss++;
  }

  return arrayValue;
}

mlir::Value MLIRStatements::mlirGen(Statement* stm) {
  _total++;

  if (ExpStatement* expStatement = stm->isExpStatement())
    return mlirGen(expStatement);
  else if (CompoundStatement* compoundStatement = stm->isCompoundStatement())
    mlirGen(compoundStatement);
  else if (ScopeStatement* scopeStatement = stm->isScopeStatement())
    mlirGen(scopeStatement);
  else if (ReturnStatement* returnStatement = stm->isReturnStatement())
    mlirGen(returnStatement);
  else if (IfStatement* ifStatement = stm->isIfStatement())
    mlirGen(ifStatement);
  else if (ForStatement* forStatement = stm->isForStatement())
    mlirGen(forStatement);
  else if (UnrolledLoopStatement* unrolledLoopStatement = stm->isUnrolledLoopStatement())
    mlirGen(unrolledLoopStatement);
  else {
  IF_LOG Logger::println("Statament doesn't match with any implemented "
                         "function: '%s'",stm->toChars());
  _miss++;
  }
  return nullptr;
}

mlir::LogicalResult MLIRStatements::genStatements(FuncDeclaration *funcDeclaration){
  _total++;
  if(CompoundStatement *compoundStatment =
                                  funcDeclaration->fbody->isCompoundStatement()){
    mlirGen(compoundStatment);
    _total += decl_total;
    _miss += decl_miss;
    return mlir::success();
  }
  _miss++;
  _total += decl_total;
  _miss += decl_miss;
  return mlir::failure();
}

#endif
