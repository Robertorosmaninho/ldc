//===-- MLIRDeclaration.cpp -----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED
#include "dmd/ast_node.h"
#include "dmd/aliasthis.h"
#include "dmd/expression.h"
#include "dmd/root/object.h"
#include "dmd/root/root.h"
#include "dmd/attrib.h"
#include "dmd/ctfe.h"
#include "dmd/enum.h"
#include "dmd/errors.h"
#include "dmd/hdrgen.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/ldcbindings.h"
#include "dmd/module.h"
#include "dmd/mtype.h"
#include "dmd/root/port.h"
#include "dmd/root/rmem.h"
#include "dmd/template.h"
#include "gen/aa.h"
#include "gen/abi.h"
#include "gen/arrays.h"
#include "gen/binops.h"
#include "gen/classes.h"
#include "gen/complex.h"
#include "gen/coverage.h"
#include "gen/dvalue.h"
#include "gen/functions.h"
#include "gen/funcgenstate.h"
#include "gen/inlineir.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/nested.h"
#include "gen/optimizer.h"
#include "gen/pragma.h"
#include "gen/runtime.h"
#include "gen/scope_exit.h"
#include "gen/structs.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "gen/warnings.h"
#include "ir/irfunction.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <fstream>
#include <math.h>
#include <stack>
#include <stdio.h>
#include "dmd/visitor.h"
#include "root/dcompat.h"
#include "root/port.h"
#include "globals.h"

#include "MLIRDeclaration.h"
#include "llvm/ADT/APFloat.h"
#include "dmd/visitor.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

MLIRDeclaration::MLIRDeclaration(
    Module *m, mlir::MLIRContext &context, mlir::OpBuilder builder,
    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
    llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
    unsigned &total, unsigned &miss)
    : module(m), context(context), builder(builder), symbolTable(symbolTable),
      structMap(structMap), _total(total), _miss(miss) {} // Constructor

MLIRDeclaration::~MLIRDeclaration() = default;

mlir::DenseElementsAttr MLIRDeclaration::getConstantAttr(Expression *exp) {
  // The type of this attribute is tensor of 64-bit floating-point with no
  // shape.
  auto type = get_MLIRtype(exp);
  auto dataType = mlir::RankedTensorType::get(1, type);

  // This is the actual attribute that holds the list of values for this
  // tensor literal.
  if (auto number = exp->isIntegerExp()) {
    return mlir::DenseElementsAttr::get(dataType,
                                        llvm::makeArrayRef(number->value));
  } else if (auto real = exp->isRealExp()) {
    if (type.isF64()) {
      double value = [](double Double) { return Double; }(real->value);
      return mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(value));
    } else if (type.isF16() || type.isF32()) {
      float value = [](double Double) { return Double; }(real->value);
      return mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(value));
    }
  }

  _miss++;
  fatal();
}

std::pair<mlir::ArrayAttr, mlir::Type>
MLIRDeclaration::getConstantAttr(StructLiteralExp *lit) {

  /// Emit a constant for a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `toy.struct_constant`
  /// operation. This function returns the generated constant, along with the
  /// corresponding struct type.
  std::vector<mlir::Attribute> attrElements;
  std::vector<mlir::Type> typeElements;

  IF_LOG Logger::println("MLIRCodeGen - Getting ConstantAttr for Struct: "
                         "'%s'",
                         lit->toChars());

  for (auto &var : *lit->elements) {
    if (auto number = var->isIntegerExp()) {
      attrElements.push_back(getConstantAttr(number));
      mlir::Type type = get_MLIRtype(number);
      typeElements.push_back(mlir::RankedTensorType::get(1, type));
    } else if (auto real = var->isRealExp()) {
      mlir::Type type = get_MLIRtype(real);
      attrElements.push_back(getConstantAttr(real));
      typeElements.push_back(mlir::RankedTensorType::get(1, type));
    } else if (var->type->ty == Tsarray) {
      mlir::Value value = mlirGen(var);
      attrElements.push_back(value.getDefiningOp()->getAttr("value"));
      typeElements.push_back(value.getType());
    } else {
      Logger::println("Unable to attach '%s' of type '%s' to '%s'",
                      var->toChars(), var->type->toChars(), lit->toChars());
      _miss++;
      fatal();
    }
  }

  mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
  mlir::Type dataType = mlir::D::StructType::get(typeElements);
  return std::make_pair(dataAttr, dataType);
}

llvm::Optional<size_t> MLIRDeclaration::getMemberIndex(Expression *expression) {
  assert(expression->isDotVarExp());

  // Lookup the struct node for the LHS.
  auto dotVarExp = expression->isDotVarExp();
  auto Struct = static_cast<StructLiteralExp *>(dotVarExp->e1);

  Logger::println("Getting member index from '%s'", Struct->toChars());
  StructDeclaration *Sd = getStructFor(Struct);

  if (!Sd)
    return llvm::None;

  // Get the name from the index.
  auto var = dotVarExp->var;
  if (!var)
    return llvm::None;

  auto members = Sd->members;
  std::vector<Dsymbol *> aux;
  for (auto dsymbol : *members)
    aux.push_back(dsymbol);

  llvm::ArrayRef<Dsymbol *> structVars(aux);
  auto it = llvm::find_if(
      structVars, [&](auto &Var) { return Var->toChars() == var->toChars(); });

  if (it == structVars.end())
    return llvm::None;
  return it - structVars.begin();
}

StructDeclaration *MLIRDeclaration::getStructFor(Expression *expression) {
  IF_LOG Logger::println("MLIRCodeGen - GetStructFor: '%s'",
                         expression->toChars());
  llvm::StringRef structName;
  if (auto *lit = static_cast<StructLiteralExp *>(expression)) {
    auto varIt = symbolTable.lookup(lit->toChars());
    if (!varIt)
      return nullptr;
    structName = expression->type->toChars();
  } else if (auto *acess = static_cast<DotVarExp *>(expression)) {

    // The name being accessed should be in var.
    auto *name = acess->var;
    if (!name)
      return nullptr;
    auto *parentStruct = getStructFor(acess->e1);
    if (!parentStruct)
      return nullptr;

    // Get the element within the struct corresponding to the name.
    Dsymbol *decl = nullptr;
    for (auto &var : *parentStruct->members) {
      if (var->toChars() == name->toChars()) {
        decl = var;
        break;
      }
    }
    if (!decl)
      return nullptr;
    structName = decl->getType()->toChars();
  }

  if (structName.empty())
    return nullptr;

  // If the struct name was valid, check for an entry in the struct map.
  auto structIt = structMap.find(structName);
  if (structIt == structMap.end())
    return nullptr;
  return structIt->second.second;
}

mlir::Value MLIRDeclaration::mlirGen(Declaration *declaration) {
  IF_LOG Logger::println("MLIRCodeGen - Declaration: '%s'",
                         declaration->toChars());
  LOG_SCOPE

  if (StructDeclaration *structDeclaration =
          declaration->isStructDeclaration()) {
    if (failed(mlirGen(structDeclaration, 0)))
      return nullptr;
  } else if (auto varDeclaration = declaration->isVarDeclaration()) {
    return mlirGen(varDeclaration);
  } else {
    IF_LOG Logger::println("Unable to recoganize Declaration: '%s'",
                           declaration->toChars());
    _miss++;
    return nullptr;
  }
}

mlir::LogicalResult
MLIRDeclaration::mlirGen(StructDeclaration *structDeclaration, bool generated) {
  IF_LOG Logger::println("MLIRCodeGen - StructDeclaration: '%s'",
                         structDeclaration->toChars());

  if (structMap.count(structDeclaration->toChars()) && !generated)
    return mlir::emitError(loc(structDeclaration->loc))
           << "error: struct "
              "type with name '"
           << structDeclaration->toChars() << "' already exists";

  auto variables = structDeclaration->members;
  std::vector<mlir::Type> elementTypes;
  std::vector<mlir::Attribute> attrElements;
  elementTypes.reserve(variables->size());
  attrElements.reserve(variables->size());
  for (auto variable : *variables) {
    Logger::println("Getting Type of '%s'", variable->toChars());
    if (variable->hasStaticCtorOrDtor())
      return mlir::emitError(loc(structDeclaration->loc))
             << "error: "
                "variables within a struct definition must not have "
                "initializers";

    Type *type0;

    if (!variable->getType())
      if (auto varDecl = variable->isVarDeclaration())
        type0 = varDecl->type;
      else
        fatal();
    else
      type0 = variable->getType();

    mlir::Type type = get_MLIRtype(nullptr, type0);
    if (!type || type.template isa<mlir::NoneType>())
      return mlir::failure();

    if (!type.template isa<mlir::RankedTensorType>())
      type = mlir::RankedTensorType::get(1, type);

    elementTypes.push_back(type);
  }

  if (!generated) {
    structMap.try_emplace(structDeclaration->toChars(),
                          mlir::D::StructType::get(elementTypes),
                          structDeclaration);
    return mlir::success();
  }

  if (symbolTable.lookup(structDeclaration->toChars()))
    return mlir::success();

  for (auto varDecl : structDeclaration->fields) {
    auto var = getConstantAttr(varDecl->_init->isExpInitializer()->exp);
    attrElements.push_back(var);
  }
  mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
  mlir::Value Struct = builder.create<mlir::D::StructConstantOp>(
      loc(structDeclaration->loc), mlir::D::StructType::get(elementTypes),
      dataAttr);
  declare(structDeclaration->toChars(), Struct);

  return mlir::success();
}

mlir::Value MLIRDeclaration::mlirGen(VarDeclaration *vd) {
  IF_LOG Logger::println("MLIRCodeGen - VarDeclaration: '%s' | %s",
      vd->toChars(), vd->type->toChars());
  _total++;
  assert(!vd->aliassym && "Aliases are handled in DtoDeclarationExp.");

  IF_LOG Logger::println("DtoVarDeclaration(vdtype = %s)", vd->type->toChars());
  LOG_SCOPE

  if (vd->nestedrefs.length) {
    IF_LOG Logger::println(
        "has nestedref set (referenced by nested function/delegate)");

    // A variable may not be really nested even if nextedrefs is not empty
    // in case it is referenced by a function inside __traits(compile) or
    // typeof.
    // assert(vd->ir->irLocal && "irLocal is expected to be already set by
    // DtoCreateNestedContext");
  }

  mlir::Type type = get_MLIRtype(nullptr, vd->type);
  type.dump();
  if (!type.template isa<mlir::RankedTensorType>())
    type = mlir::RankedTensorType::get(1, type);

  if (vd->_init) {
    if (ExpInitializer *ex = vd->_init->isExpInitializer()) {
      // TODO: Refactor this so that it doesn't look like toElem has no
      // effect.
      Logger::println("MLIRCodeGen - ExpInitializer: '%s'", ex->toChars());
      LOG_SCOPE
      auto value = mlirGen(ex->exp);
      mlir::Attribute a1 = value.getDefiningOp()->getAttr("value");
      mlir::Attribute a2 = mlir::DenseElementsAttr::get(type
          .cast<mlir::RankedTensorType>(), 0);
      mlir::Attribute a3 = mlir::DenseElementsAttr::get(
          a1.getType().cast<mlir::RankedTensorType>(), 0);
      if (a1.getType() != a2.getType() && a1 == a3) {
        if (value.getType() != type)
          value.setType(type);
        if (value.getDefiningOp()->getAttr("value").getType() != type) {
          int val = ex->exp->isBlitExp()->e2->toInteger();
          mlir::ShapedType typeC = type.cast<mlir::RankedTensorType>();
          auto attr = mlir::DenseElementsAttr::get(typeC, val);
          value.getDefiningOp()->removeAttr(
              mlir::Identifier::get("value", &context));
          value.getDefiningOp()->setAttr("value", attr);
        }
      }
      return value;
    }
  } else {
    IF_LOG Logger::println("Unable to recoganize VarDeclaration: '%s'",
                           vd->toChars());
  }
_miss++;
return nullptr;
}

mlir::Value MLIRDeclaration::DtoAssignMLIR(mlir::Location Loc, mlir::Value lhs,
                                           mlir::Value rhs, StringRef lhs_name,
                                           StringRef rhs_name, int op,
                                           bool canSkipPostblit, Type *t1,
                                           Type *t2) {
  IF_LOG Logger::println("MLIRCodeGen - DtoAssignMLIR:");
  IF_LOG Logger::println("lhs: %s @ '%s'", lhs_name.data(), t1->toChars());
  IF_LOG Logger::println("rhs: %s @ '%s'", rhs_name.data(), t2->toChars());
  LOG_SCOPE;

  assert(t1->ty != Tvoid && "Cannot assign values of type void.");

  if (t1->ty == Tbool) {
    IF_LOG Logger::println("DtoAssignMLIR == Tbool"); // TODO: DtoStoreZextI8
  } else if (t1->ty == Tstruct) {
    IF_LOG Logger::println("DtoAssignMLIR == Tstruct");
    fatal();
  }

  // lhs = rhs; TODO: Verify whats it's better
  return rhs;
}

////////////////////////////////////////////////////////////////////////////////
// Expressions to be evaluated

mlir::Value MLIRDeclaration::mlirGen(AddExp *addExp,
                                     AddAssignExp *addAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (addAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - AddExp: '%s' + '%s' @ %s", addAssignExp->e1->toChars(),
        addAssignExp->e2->toChars(), addAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location addAssignLoc = loc(addAssignExp->loc);
    location = &addAssignLoc;
    e1 = mlirGen(addAssignExp->e1);
    e2 = mlirGen(addAssignExp->e2);
  } else if (addExp) {
    IF_LOG Logger::println("MLIRCodeGen - AddExp: '%s' + '%s' @ %s",
                           addExp->e1->toChars(), addExp->e2->toChars(),
                           addExp->type->toChars());
    LOG_SCOPE
    mlir::Location addExpLoc = loc(addExp->loc);
    location = &addExpLoc;
    e1 = mlirGen(addExp->e1);
    e2 = mlirGen(addExp->e2);
  } else {
    _miss++;
    return nullptr;
  }
  auto tensor = e1.getType().cast<mlir::TensorType>();
  auto type = tensor.getElementType();
  if (type.isF16() || type.isF32() || type.isF64()) {
    result = builder.create<mlir::D::AddFOp>(*location, e1, e2).getResult();
  } else if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
             type.isInteger(64)) {
    result = builder.create<mlir::D::AddOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(AndExp *andExp,
                                     AndAssignExp *andAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (andAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - AndExp: '%s' + '%s' @ %s", andAssignExp->e1->toChars(),
        andAssignExp->e2->toChars(), andAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location andAssignLoc = loc(andAssignExp->loc);
    location = &andAssignLoc;
    e1 = mlirGen(andAssignExp->e1);
    e2 = mlirGen(andAssignExp->e2);
  } else if (andExp) {
    IF_LOG Logger::println("MLIRCodeGen - AndExp: '%s' + '%s' @ %s",
                           andExp->e1->toChars(), andExp->e2->toChars(),
                           andExp->type->toChars());
    LOG_SCOPE
    mlir::Location andExpLoc = loc(andExp->loc);
    location = &andExpLoc;
    e1 = mlirGen(andExp->e1);
    e2 = mlirGen(andExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  if ((e1.getType().isInteger(8) || e1.getType().isInteger(16) ||
       e1.getType().isInteger(32) || e1.getType().isInteger(64)) &&
      (e2.getType().isInteger(8) || e2.getType().isInteger(16) ||
       e2.getType().isInteger(32) || e2.getType().isInteger(64))) {
    result = builder.create<mlir::D::AndOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}



void collectZeroInsideInt(std::vector<mlir::APInt> &data, int size, int dim) {
    for (int i = 0; i < dim; i++)
      data.push_back(mlir::APInt(size, 0));
}

void collectDataInt(Expression *e, std::vector<mlir::APInt> &data, int size) {
  if (e->isArrayLiteralExp()) {
    auto elements = e->isArrayLiteralExp()->elements;

    if (elements->size() > 1) {
      for (auto ex : *elements) {
        collectDataInt(ex, data, size);
      }
      return;
    }

    auto element = elements->front()->toInteger();
    auto insert = mlir::APInt(size, element, !elements->front()->type->isunsigned());
    data.push_back(insert);
  } else {

    auto element = e->toInteger();
    auto insert = mlir::APInt(size, element, !e->type->isunsigned());
    data.push_back(insert);
  }
}

void collectDataFloat(Expression *e, std::vector<mlir::APFloat> &data,
                      mlir::Type elementType) {
  if (e->isArrayLiteralExp()) {
    auto elements = e->isArrayLiteralExp()->elements;

    if (elements->size() > 1) {
      for (auto ex : *elements) {
        collectDataFloat(ex, data, elementType);
      }
      return;
    }

    if (elementType.isF64()) {
      mlir::APFloat apFloat((double)elements->front()->toReal());
      data.push_back(apFloat);
    } else {
      mlir::APFloat apFloat((float)elements->front()->toReal());
      data.push_back(apFloat);
    }
  } else {

    if (elementType.isF64()) {
      mlir::APFloat apFloat((double)e->toReal());
      data.push_back(apFloat);
    } else {
      mlir::APFloat apFloat((float)e->toReal());
      data.push_back(apFloat);
    }
  }
}

mlir::Value MLIRDeclaration::mlirGen(ArrayLiteralExp *arrayLiteralExp) {
  IF_LOG Logger::println("MLIRCodeGen - ArrayLiteralExp: '%s'",
                         arrayLiteralExp->toChars());
  LOG_SCOPE
  _total++;

  // IF_LOG Logger::println("Basis: '%s'",arrayLiteralExp->basis->toChars());
  IF_LOG Logger::println("Elements: '%s'",
                         arrayLiteralExp->elements->toChars());

  // The type of this attribute is tensor the type of the literal with it's
  // shape .
  auto type = get_MLIRtype(arrayLiteralExp).cast<mlir::RankedTensorType>();
  auto elementType = type.getElementType();

  bool isFloat =
      elementType.isF16() || elementType.isF32() || elementType.isF64();

  int size = 0;
  if (elementType.isInteger(1))
    size = 1;
  else if (elementType.isInteger(8))
    size = 8;
  else if (elementType.isInteger(16))
    size = 16;
  else if (elementType.isInteger(32))
    size = 32;
  else if (elementType.isInteger(64))
    size = 64;
  else if (!isFloat)
    IF_LOG Logger::println("MLIR doesn't support integer of type different "
                           "from 1,8,16,32,64");

  std::vector<mlir::APInt> data;
  std::vector<mlir::APFloat> dataF; // Used by float and double
  mlir::Location Loc = loc(arrayLiteralExp->loc);


  Logger::println("Elements: %s", arrayLiteralExp->toChars());
  if (elementType.isInteger(size)) {
    collectDataInt(arrayLiteralExp, data, size);
  } else if (isFloat) {
    collectDataFloat(arrayLiteralExp, dataF, elementType);
  } else {
    _miss++;
    return nullptr;
  }

  // For now lets assume one-dimensional arrays
  std::vector<int64_t> dims = type.getShape();

  // Getting the shape of the tensor. Ex.: tensor<4xf64>
  auto dataType = mlir::RankedTensorType::get(dims, elementType);

  // Set the actual attribute that holds the list of values for this
  // tensor literal and build the operation.
  if (elementType.isInteger(size)) {
    auto dataAttributes = mlir::DenseElementsAttr::get(dataType, data);
    return builder.create<mlir::D::IntegerOp>(Loc, dataAttributes);
  } else if (isFloat) {
    auto dataAttributes = mlir::DenseElementsAttr::get(dataType, dataF);
    if (elementType.isF64())
      return builder.create<mlir::D::DoubleOp>(Loc, dataAttributes);
    else
      return builder.create<mlir::D::FloatOp>(Loc, dataAttributes);
  } else {
    _miss++;
    Logger::println("Unable to build ArrayLiteralExp: '%s'",
                    arrayLiteralExp->toChars());
  }
  return nullptr;
}

mlir::Value MLIRDeclaration::mlirGen(AssignExp *assignExp) {
  _total++;
  IF_LOG Logger::print(
      "AssignExp::toElem: %s | (%s)(%s = %s)\n", assignExp->toChars(),
      assignExp->type->toChars(), assignExp->e1->type->toChars(),
      assignExp->e2->type ? assignExp->e2->type->toChars() : nullptr);

  if (auto ale = assignExp->e1->isArrayLengthExp()) {
    Logger::println("performing array.length assignment");
    fatal();
    /*DLValue arrval(ale->e1->type, DtoLVal(ale->e1));
    DValue *newlen = toElem(e->e2);
    DSliceValue *slice =
        DtoResizeDynArray(e->loc, arrval.type, &arrval, DtoRVal(newlen));
    DtoStore(DtoRVal(slice), DtoLVal(&arrval));
    result = newlen;*/
    return nullptr;
  }

  if (assignExp->memset & referenceInit) {
    assert(assignExp->op == TOKconstruct || assignExp->op == TOKblit);
    assert(assignExp->e1->op == TOKvar);

    Declaration *d = static_cast<VarExp *>(assignExp->e1)->var;
    if (d->storage_class & (STCref | STCout)) {
      Logger::println("performing ref variable initialization");
      mlir::Value rhs = mlirGen(assignExp->e2);
      // mlir::Value *lhs = mlirGen(assignExp->e1);
      mlir::Value value;

      if (!rhs) {
        _miss++;
        fatal();
      }
      // TODO: Create test to evaluate this case and transform it into dialect
      // operation
      mlir::OperationState result(loc(assignExp->loc), "assign");
      result.addTypes(rhs.getType()); // TODO: type
      result.addOperands(rhs);
      value = builder.createOperation(result)->getResult(0);

      if (failed(declare(assignExp->e1->toChars(), rhs))) {
        _miss++;
        fatal();
      }
      return value;
    }
  }

  // This matches the logic in AssignExp::semantic.
  // TODO: Should be cached in the frontend to avoid issues with the code
  // getting out of sync?
  bool lvalueElem = false;
  if ((assignExp->e2->op == TOKslice &&
       static_cast<UnaExp *>(assignExp->e2)->e1->isLvalue()) ||
      (assignExp->e2->op == TOKcast &&
       static_cast<UnaExp *>(assignExp->e2)->e1->isLvalue()) ||
      (assignExp->e2->op != TOKslice && assignExp->e2->isLvalue())) {
    lvalueElem = true;
  }

  Type *t1 = assignExp->e1->type->toBasetype();
  Type *t2 = assignExp->e2->type->toBasetype();

  if (!((assignExp->e1->type->toBasetype()->ty) == Tstruct) &&
      !(assignExp->e2->op == TOKint64) &&
      !(assignExp->op == TOKconstruct || assignExp->op == TOKblit) &&
      !(assignExp->e1->op == TOKslice))
    return DtoAssignMLIR(loc(assignExp->loc), mlirGen(assignExp->e1),
                         mlirGen(assignExp->e2), assignExp->e1->toChars(),
                         assignExp->e2->toChars(), assignExp->op, !lvalueElem,
                         t1, t2);

  // check if it is a declared variable
  mlir::Value lhs = nullptr;
  mlir::Value rhs = nullptr;

  Logger::println("e1: %s \ne2: %s", assignExp->e1->toChars(),
      assignExp->e2->toChars());


  // The front-end sometimes rewrites a static-array-lhs to a slice, e.g.,
  // when initializing a static array with an array literal.
  // Use the static array as lhs in that case.
  mlir::Value rewrittenLhsStaticArray = nullptr;
  if(auto se = assignExp->e1->isSliceExp()) {
    Type *sliceeBaseType = se->e1->type->toBasetype();
    if (se->lwr == nullptr && sliceeBaseType->ty == Tsarray &&
        se->type->toBasetype()->nextOf() == sliceeBaseType->nextOf()) {
      auto typeC = mlir::RankedTensorType::get(
          0, get_MLIRtype(nullptr, sliceeBaseType->nextOf()));
      auto ty = sliceeBaseType->nextOf()->ty;
      auto Loc = loc(assignExp->loc);
      if (ty == Tfloat32) {
        auto dataAttribute = mlir::DenseIntElementsAttr::get(typeC, (float)assignExp->e2->toReal());
        lhs = builder.create<mlir::D::FloatOp>(Loc, dataAttribute);
      } else if (ty == Tfloat64 || ty == Tfloat80) {
        auto dataAttribute = mlir::DenseIntElementsAttr::get(typeC, (float)assignExp->e2->toReal());
        lhs = builder.create<mlir::D::DoubleOp>(Loc, dataAttribute);
      } else if (ty == Tint8 || ty == Tint16 || ty == Tint32 || ty == Tint64 ||
               ty == Tint128) {
        auto dataAttribute =
            mlir::DenseIntElementsAttr::get(typeC, (int)assignExp->e2->toInteger());
        lhs = builder.create<mlir::D::IntegerOp>(Loc, dataAttribute);
       declare(assignExp->e1->isSliceExp()->e1->toChars(), lhs);
      }
    }
      //rewrittenLhsStaticArray = nullptr;
  }

  if (auto var = assignExp->e1->isVarExp())
    lhs = mlirGen(var);
  else if (auto dot = assignExp->e1->isDotVarExp())
    lhs = mlirGen(dot);
  else if (auto index = assignExp->e1->isIndexExp()) {
    mlirGen(index);
    //fatal();
  } else if (lhs != nullptr) {
    return lhs;
  } else {
    Logger::println("assign e1: '%s' of type: '%s' and op : '%d'",
                    assignExp->e1->toChars(), assignExp->e1->type->toChars(),
                    assignExp->e1->op);

    _miss++;
    fatal();
  }

  if (lhs != nullptr) {
    rhs = mlirGen(assignExp->e2);
    if (failed(declare(assignExp->e1->toChars(), rhs))) {
      // Replace the value on symbol table
      symbolTable.insert(assignExp->e1->toChars(), rhs);
    }
    Logger::println("DtoMLIRAssign()");
    Logger::println("lhs: %s", assignExp->e1->toChars());
    Logger::println("rhs: %s", assignExp->e2->toChars());
    return rhs;
  } else {
    _miss++;
    IF_LOG Logger::println("Failed to assign '%s' to '%s'",
                           assignExp->e2->toChars(), assignExp->e1->toChars());
    fatal();
  }

  IF_LOG Logger::println("Unable to translate AssignExp: '%s'",
                         assignExp->toChars());
  _miss++;
  fatal();
}

mlir::Value MLIRDeclaration::mlirGen(CallExp *callExp) {
  IF_LOG Logger::println("MLIRCodeGen - CallExp: '%s'", callExp->toChars());
  LOG_SCOPE
  _total++;

  VarDeclaration *delayedDtorVar = nullptr;
  Expression *delayedDtorExp = nullptr;
  auto Loc = loc(callExp->loc);

  // Check if we are about to construct a just declared temporary. DMD
  // unfortunately rewrites this as
  //   MyStruct(myArgs) => (MyStruct tmp; tmp).this(myArgs),
  // which would lead us to invoke the dtor even if the ctor throws. To
  // work around this, we hold on to the cleanup and push it only after
  // making the function call.
  //
  // The correct fix for this (DMD issue 13095) would have been to adapt
  // the AST, but we are stuck with this as DMD also patched over it with
  // a similar hack.
  if (callExp->f && callExp->f->isCtorDeclaration()) {
    if (auto dve = callExp->e1->isDotVarExp())
      if (auto ce = dve->e1->isCommaExp())
        if (ce->e1->op == TOKdeclaration)
          if (auto ve = ce->e2->isVarExp())
            if (auto vd = ve->var->isVarDeclaration())
              if (vd->needsScopeDtor()) {
                Logger::println("Delaying edtor");
                delayedDtorVar = vd;
                delayedDtorExp = vd->edtor;
                vd->edtor = nullptr;
              }
  }

  // get the calle
  mlir::Value fnval;
  if (callExp->directcall) {
    // TODO: Do this as an extra parameter to DotVarExp implementation.
    auto dve = callExp->e1->isDotVarExp();
    assert(dve);
    FuncDeclaration *fdecl = dve->var->isFuncDeclaration();
    assert(fdecl);
    MLIRFunction mlirFunc(fdecl, context, builder, symbolTable, structMap,
                          _total, _miss);
    mlirFunc.DtoMLIRDeclareFunction(fdecl); //TODO: This does not work yet.
    //TODO: Create DtoRVal and DtoLVal
  } else {
    fnval = mlirGen(callExp->e1);
  }

  std::vector<mlir::Type> types;
  StringRef functionName = callExp->f->mangleString;

  // Codegen the operands first.
  llvm::SmallVector<mlir::Value, 4> operands;
  for (auto exp : *callExp->arguments) {
    auto arg = mlirGen(exp);
    if (!arg) {
      _miss++;
      return nullptr;
    }
    operands.push_back(arg);
  }
  // Get the return type
  auto Fd = callExp->f;
  if (!Fd->returns) {
    auto type = get_MLIRtype(nullptr, Fd->type);
    TypeFunction *funcType = static_cast<TypeFunction *>(Fd->type);
    auto ty = funcType->next->ty;
    if (ty != Tvector && ty != Tarray && ty != Tsarray && ty != Taarray &&
        ty != Tvoid) {
      auto dataType = mlir::RankedTensorType::get(1, type);
      types.push_back(dataType);
    } else if (type.template isa<mlir::TensorType>()){
      auto tensorType = type.cast<mlir::TensorType>();
      types.push_back(tensorType);
    }
  }

  std::vector<mlir::Type> _types;
  for (auto arg : operands)
    _types.push_back(arg.getType());

  auto funcType = mlir::FunctionType::get(_types, {}, &context);
  llvm::ArrayRef<mlir::Type> ret_type(types);
  return builder.create<mlir::D::CallOp>(Loc, funcType, functionName,
                                         operands);
}

mlir::Value MLIRDeclaration::mlirGen(CastExp *castExp) {
  IF_LOG Logger::print("MLIRCodeGen - CastExp: %s @ %s\n", castExp->toChars(),
                       castExp->e1->toChars());
  LOG_SCOPE;

  Logger::println("Cast from %s to %s", castExp->type->toChars(),
                  castExp->to->toChars());

  // Getting mlir location
  mlir::Location location = loc(castExp->loc);

  // get the value to cast
  mlir::Value result;
  result = mlirGen(castExp->e1);

  // handle cast to void (usually created by frontend to avoid "has no effect"
  // error)
  if (castExp->to == Type::tvoid) {
    result = nullptr;
    return result;
  }
  // cast it to the 'to' type, if necessary
  if (castExp->to->equals(castExp->e1->type)) {
    return result;
  }

  // paint the type, if necessary
  if (!castExp->type->equals(castExp->to)) {
    // result = DtoPaintType(castExp->loc, result, castExp->type); //TODO:
    IF_LOG Logger::println("Unable to paint the type, function not "
                           "implemented");
    _miss++;
  }

  int size = 1;
  if (auto isTensor = result.getType().template isa<mlir::TensorType>())
    size = result.getType().cast<mlir::TensorType>().getNumElements();

  auto singletype = get_MLIRtype(castExp);
  auto type = mlir::RankedTensorType::get(size, singletype);

  return builder.create<mlir::D::CastOp>(location, result, type);
}

mlir::Value MLIRDeclaration::mlirGen(ConstructExp *constructExp) {
  IF_LOG Logger::println("MLIRCodeGen - ConstructExp: '%s'",
                         constructExp->toChars());
  LOG_SCOPE
  _total++;
  // mlir::Value *lhs = mlirGen(constructExp->e1);
  mlir::Value rhs = mlirGen(constructExp->e2);

  Logger::println("Declaring '%s' = '%s' on SymbolTable",
                  constructExp->e1->toChars(), constructExp->e2->toChars());
  if (failed(declare(constructExp->e1->toChars(), rhs))) {
    _miss++;
    return nullptr;
  }
  return rhs;
}

mlir::Value MLIRDeclaration::mlirGen(DeclarationExp *decl_exp) {
  IF_LOG Logger::println("MLIRCodeGen - DeclExp: '%s'", decl_exp->toChars());
  LOG_SCOPE
  Dsymbol *dsym = decl_exp->declaration;
  _total++;

  if (VarDeclaration *vd = dsym->isVarDeclaration())
    return mlirGen(vd);
  else if (StructDeclaration *structDeclaration = dsym->isStructDeclaration())
    if (failed(mlirGen(structDeclaration, 0)))
      return nullptr;

  IF_LOG Logger::println("Unable to recoganize DeclarationExp: '%s'",
                         dsym->toChars());
  _miss++;
  return nullptr;
}

mlir::Value MLIRDeclaration::mlirGen(DivExp *divExp,
                                     DivAssignExp *divAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;
  bool isSigned = true;

  if (divAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - DivExp: '%s' + '%s' @ %s", divAssignExp->e1->toChars(),
        divAssignExp->e2->toChars(), divAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location divAssignLoc = loc(divAssignExp->loc);
    location = &divAssignLoc;
    e1 = mlirGen(divAssignExp->e1);
    e2 = mlirGen(divAssignExp->e2);
    if (divAssignExp->e1->type->isunsigned() ||
        divAssignExp->e1->type->isunsigned())
      isSigned = false;
  } else if (divExp) {
    IF_LOG Logger::println("MLIRCodeGen - DivExp: '%s' + '%s' @ %s",
                           divExp->e1->toChars(), divExp->e2->toChars(),
                           divExp->type->toChars());
    LOG_SCOPE
    mlir::Location divExpLoc = loc(divExp->loc);
    location = &divExpLoc;
    e1 = mlirGen(divExp->e1);
    e2 = mlirGen(divExp->e2);
    if (divExp->e1->type->isunsigned() || divExp->e1->type->isunsigned())
      isSigned = false;
  } else {
    _miss++;
    return nullptr;
  }

  if ((e1.getType().isF16() || e1.getType().isF32() || e1.getType().isF64()) &&
      (e2.getType().isF16() || e2.getType().isF32() || e2.getType().isF64())) {
    result = builder.create<mlir::D::DivFOp>(*location, e1, e2).getResult();
  } else if ((e1.getType().isInteger(8) || e1.getType().isInteger(16) ||
              e1.getType().isInteger(32) || e1.getType().isInteger(64)) &&
             (e2.getType().isInteger(8) || e2.getType().isInteger(16) ||
              e2.getType().isInteger(32) || e2.getType().isInteger(64))) {
    if (isSigned)
      result = builder.create<mlir::D::DivSOp>(*location, e1, e2).getResult();
    else
      result = builder.create<mlir::D::DivUOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(DotVarExp *dotVarExp) {
  Logger::println("DotVarExp::toElem: %s @ %s", dotVarExp->toChars(),
                  dotVarExp->type->toChars());

  auto location = loc(dotVarExp->loc);
  Type *e1type = dotVarExp->e1->type->toBasetype();

  if(auto value = symbolTable.lookup(dotVarExp->toChars()))
    return value;

  if (VarDeclaration *vd = dotVarExp->var->isVarDeclaration()) {
    //  LLValue *arrptr;
    // indexing struct pointer
    if (e1type->ty == Tpointer) {
      assert(e1type->nextOf()->ty == Tstruct);
      Logger::println("dotVarExp is VarDeclaration");
      // TypeStruct *ts = static_cast<TypeStruct *>(e1type->nextOf());
      // arrptr = DtoIndexAggregate(DtoRVal(l), ts->sym, vd);
    }
    // indexing normal struct
    else if (e1type->ty == Tstruct) {
//      auto *ts = static_cast<TypeStruct *>(e1type);
      llvm::Optional<size_t> accessIndex = getMemberIndex(dotVarExp);
      if (!accessIndex) {
        emitError(location, "invalid access into struct expression");
        return nullptr;
      }
      auto lhs = mlirGen(dotVarExp->e1);
      return builder.create<mlir::D::StructAccessOp>(location, lhs,
                                                     *accessIndex);
    }
    // indexing class
    else if (e1type->ty == Tclass) {
      TypeClass *tc = static_cast<TypeClass *>(e1type);
      //    arrptr = DtoIndexAggregate(DtoRVal(l), tc->sym, vd);
    } else {
      llvm_unreachable("Unknown DotVarExp type for VarDeclaration.");
    }

    //  Logger::cout() << "mem: " << *arrptr << '\n';
    // result = new DLValue(e->type, DtoBitCast(arrptr, DtoPtrToType(e->type)));
  } else if (FuncDeclaration *fdecl = dotVarExp->var->isFuncDeclaration()) {
    Logger::println("acess to Fd: %s", fdecl->toChars());
    /*DtoResolveFunction(fdecl);

    // This is a bit more convoluted than it would need to be, because it
    // has to take templated interface methods into account, for which
    // isFinalFunc is not necessarily true.
    // Also, private/package methods are always non-virtual.
    const bool nonFinal = !fdecl->isFinalFunc() &&
                          (fdecl->isAbstract() || fdecl->isVirtual()) &&
                          fdecl->prot().kind != Prot::private_ &&
                          fdecl->prot().kind != Prot::package_;

    // Get the actual function value to call.
    LLValue *funcval = nullptr;
    if (nonFinal) {
      funcval = DtoVirtualFunctionPointer(l, fdecl, e->toChars());
    } else {
      funcval = DtoCallee(fdecl);
    }
    assert(funcval);

    LLValue *vthis = (DtoIsInMemoryOnly(l->type) ? DtoLVal(l) : DtoRVal(l));
    result = new DFuncValue(fdecl, funcval, vthis);*/
  } else {
    llvm_unreachable("Unknown target for VarDeclaration.");
  }
}

mlir::Value MLIRDeclaration::mlirGen(Expression *expression, int func) {
  CmpExp *cmpExp = static_cast<CmpExp *>(expression);
  IF_LOG Logger::println("MLIRCodeGen - CmpExp: '%s'", cmpExp->toChars());
  _total++;

  mlir::Location location = loc(cmpExp->loc);
  mlir::Value e1 = mlirGen(cmpExp->e1);
  mlir::Value e2 = mlirGen(cmpExp->e2);
  Type *t = cmpExp->e1->type->toBasetype();
  mlir::CmpIPredicate predicate;
  mlir::CmpFPredicate predicateF;

  bool isFloat =
      e1.getType().isF64() || e1.getType().isF32() || e1.getType().isBF16();

  switch (cmpExp->op) {
  case TOKlt:
    if (isFloat)
      predicateF = mlir::CmpFPredicate::ULT;
    else
      predicate =
          t->isunsigned() ? mlir::CmpIPredicate::ult : mlir::CmpIPredicate::slt;
    break;
  case TOKle:
    if (isFloat)
      predicateF = mlir::CmpFPredicate::ULE;
    else
      predicate =
          t->isunsigned() ? mlir::CmpIPredicate::ule : mlir::CmpIPredicate::sle;
    break;
  case TOKgt:
    if (isFloat)
      predicateF = mlir::CmpFPredicate::UGT;
    else
      predicate =
          t->isunsigned() ? mlir::CmpIPredicate::ugt : mlir::CmpIPredicate::sgt;
    break;
  case TOKge:
    if (isFloat)
      predicateF = mlir::CmpFPredicate::UGE;
    else
      predicate =
          t->isunsigned() ? mlir::CmpIPredicate::uge : mlir::CmpIPredicate::sge;
    break;
  case TOKequal:
    if (isFloat)
      predicateF = mlir::CmpFPredicate::UEQ;
    else
      predicate = mlir::CmpIPredicate::eq;
    break;
  case TOKnotequal:
    if (isFloat)
      predicateF = mlir::CmpFPredicate::UNE;
    else
      predicate = mlir::CmpIPredicate::ne;
    break;
  default:
    _miss++;
    IF_LOG Logger::println("Invalid comparison operation");
    break;
  }

  if (isFloat)
    return builder.create<mlir::CmpFOp>(location, predicateF, e1, e2);
  else
    return builder.create<mlir::CmpIOp>(location, predicate, e1, e2);
}

mlir::Attribute MLIRDeclaration::mlirGen(IndexExp *indexExp) {
  IF_LOG Logger::print("MLIRCodeGen - IndexExp: %s @ %s\n", indexExp->toChars(),
                       indexExp->type->toChars());
  LOG_SCOPE;

  Logger::println("IndexExp: %s \ne1: %s \ne2: %s", indexExp->toChars(),
      indexExp->e1->toChars(), indexExp->e2->toChars());

  mlir::Attribute l;
  if (indexExp->e1->isIndexExp())
    l = mlirGen(indexExp->e1->isIndexExp());
  else
    l = mlirGen(indexExp->e1).getDefiningOp()->getAttr("value");

  if (l != nullptr) {
    auto index = mlirGen(indexExp->e2);
    l.dump();
    if (l.template isa<mlir::ArrayAttr>()) {
      auto array = l.cast<mlir::ArrayAttr>();
      auto llvmarray = array.getValue();
      return llvmarray[index];
    } else {
      auto val = index.getDefiningOp()->getAttr("value");
      mlir:: //O problema agora é que array3[3][3] = 0 e não
      // [[0,0,0],[0,0,0],[0,0,0]] então não é possível acessar essa posição.
      // O certo é arrumar um jeito de criar essa matrix. Mas como? Talvez
      // imitando o jeito que coletamos os dados do arrayliteral expression.
      // testar fazer explicitamente array3[3][3] = [[0,0,0],[0,0,0],[0,0,0]]
      // e então tentar copiar os dados.
    }
  } else if (l == nullptr) {
    // do something
  }

  Type *e1type = indexExp->e1->type->toBasetype();
  // Fala Roberto, 2 problemas: vc tem que resolver a recursão nesse index
  // pro array[][] virar array[] e vc poder trabalhar com isso. Além disso,
  // array3 não foi declarado na symboltable pq tá reconhecendo a expressão
  // da declaração array[3][3] -> array[][] -> array[] como sliceExp e vc não
  // dá suporte a isso. Por fim, creio que a solução pra alterar o valor do
  // index é pegar o valor do attribute e se possível iterar até o elemento e
  // muda-lo lá, o problema que eu vejo nisso é que no nivel mais inferior do
  // index array[i] nós não teremos toda a informação necessaria, ou seja,
  // aqui deve retornar só o valuor do atributo, o subarray definido em
  // array[i], e o constructExp ou varDecl, enfim, deve ser responsavel por
  // atualizar esse valor, acho que esse pode ser o caminho correto.
  // Boa sorte e concentre-se da próxima vez!

  //p->arrays.push_back(l); // if $ is used it must be an array so this is fine.
  auto r = mlirGen(indexExp->e2);
 // l.dump();
  r.dump();
  //p->arrays.pop_back();
  //Logger::println("VALUE: %s", indexExp->e1->isIndexExp()->e1->toChars());
 // auto value = symbolTable.lookup(indexExp->e1->isIndexExp()->e1->toChars());
 // if (value != nullptr)
 //   value.dump();


 // result = new DLValue(e->type, DtoBitCast(arrptr, DtoPtrToType(e->type)));
}

mlir::Value MLIRDeclaration::mlirGen(IntegerExp *integerExp) {
  _total++;
  dinteger_t dint = integerExp->value;
  Logger::println("MLIRGen - Integer: '%lu' | size: %c | type: %s", dint,
      integerExp->size, integerExp->type->toChars());
  mlir::Location location = builder.getUnknownLoc();
  if (integerExp->loc.filename == NULL)
    location = builder.getUnknownLoc();
  else
    location = loc(integerExp->loc);

  auto basetype = integerExp->type->toBasetype();
  mlir::RankedTensorType shapedType;
  mlir::DenseElementsAttr dataAttribute;
  if (basetype->ty == Tbool) {
    shapedType = mlir::RankedTensorType::get(1, builder.getIntegerType(1));
    dataAttribute = mlir::DenseElementsAttr::get(shapedType, (bool)dint);
  } else if (basetype->ty == Tint8 || basetype->ty == Tuns8) {
    shapedType = mlir::RankedTensorType::get(1, builder.getIntegerType(8));
    dataAttribute = mlir::DenseElementsAttr::get(shapedType, (char)dint);
  } else if (basetype->ty == Tint16 || basetype->ty == Tuns16) {
    shapedType = mlir::RankedTensorType::get(1, builder.getIntegerType(16));
    dataAttribute = mlir::DenseElementsAttr::get(shapedType, (short)dint);
  } else if (basetype->ty == Tint32 || basetype->ty == Tuns32) {
    shapedType = mlir::RankedTensorType::get(1, builder.getIntegerType(32));
    dataAttribute = mlir::DenseElementsAttr::get(shapedType, (int)dint);
  } else if (basetype->ty == Tint64 || basetype->ty == Tuns64) {
    shapedType = mlir::RankedTensorType::get(1, builder.getIntegerType(64));
    dataAttribute = mlir::DenseElementsAttr::get(shapedType, (long)dint);
  } else if (basetype->ty == Tint128 || basetype->ty == Tuns128) {
    shapedType = mlir::RankedTensorType::get(1, builder.getIntegerType(128));
    dataAttribute = mlir::DenseElementsAttr::get(shapedType, (dint));
  } else {
    _miss++;
    fatal();
  }
  return builder.create<mlir::D::IntegerOp>(location, dataAttribute);
}

mlir::Value MLIRDeclaration::mlirGen(MinExp *minExp,
                                     MinAssignExp *minAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (minAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - MinExp: '%s' + '%s' @ %s", minAssignExp->e1->toChars(),
        minAssignExp->e2->toChars(), minAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location minAssignLoc = loc(minAssignExp->loc);
    location = &minAssignLoc;
    e1 = mlirGen(minAssignExp->e1);
    e2 = mlirGen(minAssignExp->e2);
  } else if (minExp) {
    IF_LOG Logger::println("MLIRCodeGen - MinExp: '%s' + '%s' @ %s",
                           minExp->e1->toChars(), minExp->e2->toChars(),
                           minExp->type->toChars());
    LOG_SCOPE
    mlir::Location minExpLoc = loc(minExp->loc);
    location = &minExpLoc;
    e1 = mlirGen(minExp->e1);
    e2 = mlirGen(minExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  if ((e1.getType().isF16() || e1.getType().isF32() || e1.getType().isF64()) &&
      (e2.getType().isF16() || e2.getType().isF32() || e2.getType().isF64())) {
    result = builder.create<mlir::D::SubFOp>(*location, e1, e2).getResult();
  } else if ((e1.getType().isInteger(8) || e1.getType().isInteger(16) ||
              e1.getType().isInteger(32) || e1.getType().isInteger(64)) &&
             (e2.getType().isInteger(8) || e2.getType().isInteger(16) ||
              e2.getType().isInteger(32) || e2.getType().isInteger(64))) {
    result = builder.create<mlir::D::SubOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(ModExp *modExp,
                                     ModAssignExp *modAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;
  bool isSigned = true;

  if (modAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - ModExp: '%s' + '%s' @ %s", modAssignExp->e1->toChars(),
        modAssignExp->e2->toChars(), modAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location modAssignLoc = loc(modAssignExp->loc);
    location = &modAssignLoc;
    e1 = mlirGen(modAssignExp->e1);
    e2 = mlirGen(modAssignExp->e2);
    if (modAssignExp->e1->type->isunsigned() ||
        modAssignExp->e1->type->isunsigned())
      isSigned = false;
  } else if (modExp) {
    IF_LOG Logger::println("MLIRCodeGen - ModExp: '%s' + '%s' @ %s",
                           modExp->e1->toChars(), modExp->e2->toChars(),
                           modExp->type->toChars());
    LOG_SCOPE
    mlir::Location modExpLoc = loc(modExp->loc);
    location = &modExpLoc;
    e1 = mlirGen(modExp->e1);
    e2 = mlirGen(modExp->e2);
    if (modExp->e1->type->isunsigned() || modExp->e1->type->isunsigned())
      isSigned = false;
  } else {
    _miss++;
    return nullptr;
  }

  if ((e1.getType().isF16() || e1.getType().isF32() || e1.getType().isF64()) &&
      (e2.getType().isF16() || e2.getType().isF32() || e2.getType().isF64())) {
    result = builder.create<mlir::D::ModFOp>(*location, e1, e2).getResult();
  } else if ((e1.getType().isInteger(8) || e1.getType().isInteger(16) ||
              e1.getType().isInteger(32) || e1.getType().isInteger(64)) &&
             (e2.getType().isInteger(8) || e2.getType().isInteger(16) ||
              e2.getType().isInteger(32) || e2.getType().isInteger(64))) {
    if (isSigned)
      result = builder.create<mlir::D::ModSOp>(*location, e1, e2).getResult();
    else
      result = builder.create<mlir::D::ModUOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(MulExp *mulExp,
                                     MulAssignExp *mulAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (mulAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - MulExp: '%s' + '%s' @ %s", mulAssignExp->e1->toChars(),
        mulAssignExp->e2->toChars(), mulAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location mulAssignLoc = loc(mulAssignExp->loc);
    location = &mulAssignLoc;
    e1 = mlirGen(mulAssignExp->e1);
    e2 = mlirGen(mulAssignExp->e2);
  } else if (mulExp) {
    IF_LOG Logger::println("MLIRCodeGen - MulExp: '%s' + '%s' @ %s",
                           mulExp->e1->toChars(), mulExp->e2->toChars(),
                           mulExp->type->toChars());
    LOG_SCOPE
    mlir::Location mulExpLoc = loc(mulExp->loc);
    location = &mulExpLoc;
    e1 = mlirGen(mulExp->e1);
    e2 = mlirGen(mulExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  auto tensor = e1.getType().cast<mlir::TensorType>();
  auto type = tensor.getElementType();
  if ((type.isF16() || type.isF32() || type.isF64()) &&
      (type.isF16() || type.isF32() || type.isF64())) {
    result = builder.create<mlir::D::MulFOp>(*location, e1, e2).getResult();
  } else if ((type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
              type.isInteger(64)) &&
             (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
              type.isInteger(64))) {
    result = builder.create<mlir::D::MulOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(NewExp *newExp) {
  IF_LOG Logger::print("MLIRCodeGen - NewExp %s @ %s\n", newExp->toChars(),
                       newExp->type->toChars());
  LOG_SCOPE;

  int result;
  bool isArgprefixHandled = false;

  assert(newExp->newtype);
  Type *ntype = newExp->newtype->toBasetype();

  // new class
  if (ntype->ty == Tclass) {
    Logger::println("new class");
   // result = DtoNewClass(e->loc, static_cast<TypeClass *>(ntype), e);
    isArgprefixHandled = true; // by DtoNewClass()
  }
  // new dynamic array
  else if (ntype->ty == Tarray) {
    IF_LOG Logger::println("new dynamic array: %s", newExp->newtype->toChars());
    assert(newExp->argprefix == NULL);
    // get dim
    assert(newExp->arguments);
    assert(newExp->arguments->length >= 1);
    if (newExp->arguments->length == 1) {
      auto sz = mlirGen((*newExp->arguments)[0]);
      // allocate & init
      //result = DtoNewDynArray(e->loc, e->newtype, sz, true);
    }} /*else {
      size_t ndims = e->arguments->length;
      std::vector<DValue *> dims;
      dims.reserve(ndims);
      for (auto arg : *e->arguments) {
        dims.push_back(toElem(arg));
      }
      result = DtoNewMulDimDynArray(e->loc, e->newtype, &dims[0], ndims);
    }
  }
  // new static array
  else if (ntype->ty == Tsarray) {
    llvm_unreachable("Static array new should decay to dynamic array.");
  }
  // new struct
  else if (ntype->ty == Tstruct) {
    IF_LOG Logger::println("new struct on heap: %s\n", e->newtype->toChars());

    TypeStruct *ts = static_cast<TypeStruct *>(ntype);

    // allocate
    LLValue *mem = nullptr;
    if (e->allocator) {
      // custom allocator
      DtoResolveFunction(e->allocator);
      DFuncValue dfn(e->allocator, DtoCallee(e->allocator));
      DValue *res = DtoCallFunction(e->loc, nullptr, &dfn, e->newargs);
      mem = DtoBitCast(DtoRVal(res), DtoType(ntype->pointerTo()),
                       ".newstruct_custom");
    } else {
      // default allocator
      mem = DtoNewStruct(e->loc, ts);
    }

    if (!e->member && e->arguments) {
      IF_LOG Logger::println("Constructing using literal");
      write_struct_literal(e->loc, mem, ts->sym, e->arguments);
    } else {
      // set nested context
      if (ts->sym->isNested() && ts->sym->vthis) {
        DtoResolveNestedContext(e->loc, ts->sym, mem);
      }

      // call constructor
      if (e->member) {
        // evaluate argprefix
        if (e->argprefix) {
          toElemDtor(e->argprefix);
          isArgprefixHandled = true;
        }

        IF_LOG Logger::println("Calling constructor");
        assert(e->arguments != NULL);
        DtoResolveFunction(e->member);
        DFuncValue dfn(e->member, DtoCallee(e->member), mem);
        DtoCallFunction(e->loc, ts, &dfn, e->arguments);
      }
    }

    result = new DImValue(e->type, mem);
  }
  // new basic type
  else {
    IF_LOG Logger::println("basic type on heap: %s\n", e->newtype->toChars());
    assert(e->argprefix == NULL);

    // allocate
    LLValue *mem = DtoNew(e->loc, e->newtype);
    DLValue tmpvar(e->newtype, mem);

    Expression *exp = nullptr;
    if (!e->arguments || e->arguments->length == 0) {
      IF_LOG Logger::println("default initializer\n");
      // static arrays never appear here, so using the defaultInit is ok!
      exp = defaultInit(e->newtype, e->loc);
    } else {
      IF_LOG Logger::println("uniform constructor\n");
      assert(e->arguments->length == 1);
      exp = (*e->arguments)[0];
    }

    // try to construct it in-place
    if (!toInPlaceConstruction(&tmpvar, exp))
      DtoAssign(e->loc, &tmpvar, toElem(exp), TOKblit);

    // return as pointer-to
    result = new DImValue(e->type, mem);
  }

  (void)isArgprefixHandled;
  assert(e->argprefix == NULL || isArgprefixHandled);*/
}

mlir::Value MLIRDeclaration::mlirGen(OrExp *orExp, OrAssignExp *orAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (orAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - OrExp: '%s' + '%s' @ %s", orAssignExp->e1->toChars(),
        orAssignExp->e2->toChars(), orAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location orAssignLoc = loc(orAssignExp->loc);
    location = &orAssignLoc;
    e1 = mlirGen(orAssignExp->e1);
    e2 = mlirGen(orAssignExp->e2);
  } else if (orExp) {
    IF_LOG Logger::println("MLIRCodeGen - OrExp: '%s' + '%s' @ %s",
                           orExp->e1->toChars(), orExp->e2->toChars(),
                           orExp->type->toChars());
    LOG_SCOPE
    mlir::Location orExpLoc = loc(orExp->loc);
    location = &orExpLoc;
    e1 = mlirGen(orExp->e1);
    e2 = mlirGen(orExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  if ((e1.getType().isInteger(8) || e1.getType().isInteger(16) ||
       e1.getType().isInteger(32) || e1.getType().isInteger(64)) &&
      (e2.getType().isInteger(8) || e2.getType().isInteger(16) ||
       e2.getType().isInteger(32) || e2.getType().isInteger(64))) {
    result = builder.create<mlir::D::OrOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(PostExp *postExp) {
  IF_LOG Logger::print("MLIRGen - PostExp: %s @ %s\n", postExp->toChars(),
                       postExp->type->toChars());
  LOG_SCOPE;

  mlir::Value e1 = mlirGen(postExp->e1);

  if (e1 == nullptr) {
    _miss++;
    IF_LOG Logger::println("Unable to build PostExp '%s'", postExp->toChars());
    return nullptr;
  }

  if (e1.getType().isInteger(128))
    IF_LOG Logger::println("MLIR doesn't support integer of type different "
                           "from 1,8,16,32,64");
  mlir::Value e2 = nullptr;
  mlir::Location location = loc(postExp->loc);
  auto shapedType = e1.getType().cast<mlir::RankedTensorType>();
  auto elementType = shapedType.getElementType();
  auto dataAttribute = mlir::DenseElementsAttr::get(shapedType, 1);
  if (elementType.isF32() || elementType.isF16())
    e2 = builder.create<mlir::D::FloatOp>(location, dataAttribute);
  else if (elementType.isF64())
    e2 = builder.create<mlir::D::DoubleOp>(location, dataAttribute);
  else
    e2 = builder.create<mlir::D::IntegerOp>(location, dataAttribute);

  if (postExp->op == TOKplusplus) {
    if (elementType.isF32() || elementType.isF16() || elementType.isF64())
      return builder.create<mlir::D::AddFOp>(location, e1, e2);
    else
      return builder.create<mlir::AddIOp>(location, e1, e2);
  } else if (postExp->op == TOKminusminus) {
    if (elementType.isF32() || elementType.isF16() || elementType.isF64())
      return builder.create<mlir::D::SubFOp>(location, e1, e2);
    else
      return builder.create<mlir::D::SubOp>(location, e1, e2);
  }
  _miss++;
  return nullptr;
}

mlir::Value MLIRDeclaration::mlirGen(RealExp *realExp) {
  _total++;
  real_t dfloat = realExp->value;
  IF_LOG Logger::println("MLIRCodeGen - RealExp: '%Lf'", dfloat);
  mlir::Location Loc = loc(realExp->loc);
  auto type = get_MLIRtype(realExp);
  if (type.isF64())
    return builder.create<mlir::D::DoubleOp>(Loc, type, dfloat);
  else
    return builder.create<mlir::D::FloatOp>(Loc, type, dfloat);
}

mlir::Value MLIRDeclaration::mlirGen(StringExp *stringExp) {
  IF_LOG Logger::println("MLIRCodeGen - StringExp: '%s': '%s'",
                         stringExp->toChars(), stringExp->type->toChars());
  auto Loc = loc(stringExp->loc);
  auto rt =
      mlir::RankedTensorType::get(stringExp->len, builder.getIntegerType(8));
  mlir::StringAttr attr = builder.getStringAttr(StringRef(stringExp->toChars()));
  auto array = llvm::ArrayRef<mlir::Attribute>(attr);
  auto type = llvm::ArrayRef<mlir::Type>(rt);
  mlir::ValueRange range;
  return builder.create<mlir::D::StringOp>(Loc, rt, attr);
}

mlir::Value MLIRDeclaration::mlirGen(StructLiteralExp *structLiteralExp) {
  IF_LOG Logger::println("MLIRCodeGen - StructLiteralExp: '%s'",
                         structLiteralExp->toChars());

  mlir::ArrayAttr dataAttr;
  mlir::Type dataType;
  std::tie(dataAttr, dataType) = getConstantAttr(structLiteralExp);

  // Build the MLIR op `toy.struct_constant`. This invokes the
  // `StructConstantOp::build` method.
  return builder.create<mlir::D::StructConstantOp>(loc(structLiteralExp->loc),
                                                   dataType, dataAttr);
}

mlir::Value MLIRDeclaration::mlirGen(VarExp *varExp) {
  _total++;
  IF_LOG Logger::println("MLIRCodeGen - VarExp: '%s'", varExp->toChars());
  LOG_SCOPE

  assert(varExp->var);

 // if (strncmp(varExp->toChars(), "writeln", 7) == 0)
 //   fatal();

  auto var = symbolTable.lookup(varExp->var->toChars());
  if (var)
    return var;

 /* if (auto fd = varExp->var->isFuncLiteralDeclaration()) {
    Logger::println("Build genFuncLiteral");
    fd = fd->toAliasFunc();
    MLIRFunction func = MLIRFunction()
  }*/
  if (auto em = varExp->var->isEnumMember()) {
    IF_LOG Logger::println("Create temporary for enum member");
    // Return the value of the enum member instead of trying to take its
    // address (impossible because we don't emit them as variables)
    // In most cases, the front-end constfolds a VarExp of an EnumMember,
    // leaving the AST free of EnumMembers. However in rare cases,
    // EnumMembers remain and thus we have to deal with them here.
    // See DMD issues 16022 and 16100.
    // TODO: return toElem(em->value(), p) -> Expression, bool
  }

  mlir::Type type = get_MLIRtype(varExp, varExp->type);
  if (type.isIntOrFloat()) {
    IF_LOG Logger::println("Undeclared VarExp: '%s' | '%u'", varExp->toChars(),
                           varExp->op);
    auto shapedType =
        mlir::RankedTensorType::get(1, builder.getIntegerType(32));
    auto dataAttribute = mlir::DenseElementsAttr::get(shapedType, 0);
    return builder.create<mlir::D::IntegerOp>(loc(varExp->loc), dataAttribute);
  } else if (varExp->type->ty == Tstruct) {
    if (StructLiteralExp *sle = varExp->isStructLiteralExp()) {
      return mlirGen(sle);
    }

    auto structDecl = structMap.lookup(varExp->var->toChars());
    if (!structDecl.second)
      return nullptr;

    if (!symbolTable.lookup(varExp->var->toChars()))
      mlirGen(structDecl.second, 1);

    return symbolTable.lookup(varExp->var->toChars());
  }

  return DtoMLIRSymbolAddress(loc(varExp->loc), varExp->type, varExp->var);
}

mlir::Value MLIRDeclaration::DtoMLIRSymbolAddress(mlir::Location loc,
                                                  Type *type,
                                                  Declaration *declaration) {
  IF_LOG Logger::println("DtoMLIRSymbolAddress ('%s' of type '%s')",
                         declaration->toChars(), declaration->type->toChars());
  LOG_SCOPE

  if (VarDeclaration *vd = declaration->isVarDeclaration()) {
    // The magic variable __ctfe is always false at runtime
    if (vd->ident == Id::ctfe)
      return mlirGen(vd); // new DConstValue(type, DtoConstBool(false));

    // this is an error! must be accessed with DotVarExp
    if (vd->needThis()) {
      error(vd->loc, "need `this` to access member `%s`", vd->toChars());
      _miss++;
      fatal();
    }

    // _arguments
    if (vd->ident == Id::_arguments) {
      Logger::println("Id::_arguments");
      Logger::println("Build arguments traanslation");
      _miss++;
      fatal();
    }

    //Try codegen
    mlir::Value decl = nullptr;
//    decl = mlirGen(declaration->isVarDeclaration());
    if (decl != nullptr)
      return decl;
    Logger::println("VarDeclaration - Missing MLIRGen!");
    _miss++;
    fatal();
  }

  if (FuncLiteralDeclaration *funcLiteralDeclaration =
          declaration->isFuncLiteralDeclaration()) {
    Logger::println("FuncLiteralDeclaration - Missing MLIRGen!");
    _miss++;
    fatal();
  }

  if (FuncDeclaration *funcDeclaration = declaration->isFuncDeclaration()) {
    Logger::println("FuncDeclaration");
    funcDeclaration = funcDeclaration->toAliasFunc();
    auto Loc = funcDeclaration->loc;
    if (funcDeclaration->llvmInternal == LLVMinline_asm) {
      // TODO: Is this needed? If so, what about other intrinsics?
      error(Loc, "special ldc inline asm is not a normal function");
      fatal();
    } else if (funcDeclaration->llvmInternal == LLVMinline_ir) {
      // TODO: Is this needed? If so, what about other intrinsics?
      error(Loc, "special ldc inline ir is not a normal function");
      fatal();
    }

    MLIRFunction DeclFunc(funcDeclaration, context, builder, symbolTable,
                          structMap, _total, _miss);
    DeclFunc.DtoMLIRResolveFunction(funcDeclaration);

    //  const auto mlirValue =
    //      funcDeclaration->llvmInternal ? DtoMLIRCallee(funcDeclaration) :
    //      nullptr;
    return nullptr;
    //  return new DFuncValue(funcDeclaration, mlirValue);
  }

  if (SymbolDeclaration *symbolDeclaration =
          declaration->isSymbolDeclaration()) {
    // this seems to be the static initialiser for structs
    Logger::println("SymbolDeclaration - Missing MLIRGen!");
    _miss++;
    fatal();
  }
  _miss++;
  llvm_unreachable("Unimplemented VarExp type");
}

mlir::Value MLIRDeclaration::mlirGen(XorExp *xorExp,
                                     XorAssignExp *xorAssignExp) {
  mlir::Value e1, e2, result;
  mlir::Location *location = nullptr;

  if (xorAssignExp) {
    IF_LOG Logger::println(
        "MLIRCodeGen - XorExp: '%s' + '%s' @ %s", xorAssignExp->e1->toChars(),
        xorAssignExp->e2->toChars(), xorAssignExp->type->toChars());
    LOG_SCOPE
    mlir::Location xorAssignLoc = loc(xorAssignExp->loc);
    location = &xorAssignLoc;
    e1 = mlirGen(xorAssignExp->e1);
    e2 = mlirGen(xorAssignExp->e2);
  } else if (xorExp) {
    IF_LOG Logger::println("MLIRCodeGen - XorExp: '%s' + '%s' @ %s",
                           xorExp->e1->toChars(), xorExp->e2->toChars(),
                           xorExp->type->toChars());
    LOG_SCOPE
    mlir::Location xorExpLoc = loc(xorExp->loc);
    location = &xorExpLoc;
    e1 = mlirGen(xorExp->e1);
    e2 = mlirGen(xorExp->e2);
  } else {
    _miss++;
    return nullptr;
  }

  if ((e1.getType().isInteger(8) || e1.getType().isInteger(16) ||
       e1.getType().isInteger(32) || e1.getType().isInteger(64)) &&
      (e2.getType().isInteger(8) || e2.getType().isInteger(16) ||
       e2.getType().isInteger(32) || e2.getType().isInteger(64))) {
    result = builder.create<mlir::D::XorOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
}

mlir::Value MLIRDeclaration::mlirGen(SliceExp *sliceExp) {

}

void MLIRDeclaration::mlirGen(TemplateInstance *decl) {
  IF_LOG Logger::println("MLIRCodeGen - TemplateInstance: '%s'",
                         decl->toPrettyChars());
  LOG_SCOPE

  if (decl->ir->isDefined()) {
    Logger::println("Already defined, skipping.");
    return;
  }
  decl->ir->setDefined();

  if (isError(decl)) {
    Logger::println("Has errors, skipping.");
    return;
  }

  if (!decl->members) {
    Logger::println("Has no members, skipping.");
    return;
  }

  // Force codegen if this is a templated function with pragma(inline, true).
  if ((decl->members->length == 1) &&
      ((*decl->members)[0]->isFuncDeclaration()) &&
      ((*decl->members)[0]->isFuncDeclaration()->inlining == PINLINEalways)) {
    Logger::println("needsCodegen() == false, but function is marked with "
                    "pragma(inline, true), so it really does need "
                    "codegen.");
  } else {
    // FIXME: This is #673 all over again.
    if (!decl->needsCodegen()) {
      Logger::println("Does not need codegen, skipping.");
      return;
    }
    if (/*irState->dcomputetarget && */ (decl->tempdecl == Type::rtinfo ||
                                         decl->tempdecl == Type::rtinfoImpl)) {
      // Emitting object.RTInfo(Impl) template instantiations in dcompute
      // modules would require dcompute support for global variables.
      Logger::println("Skipping object.RTInfo(Impl) template instantiations "
                      "in dcompute modules.");
      return;
    }
  }

  for (auto &m : *decl->members) {
    if (m->isDeclaration()) {
      mlirGen(m->isDeclaration());
    } else {
      IF_LOG Logger::println("MLIRGen Has to be implemented for: '%s'",
                             m->toChars());
      _miss++;
    }
  }
}

mlir::Value MLIRDeclaration::mlirGen(Expression *expression,
                                     mlir::Block *block) {
  IF_LOG Logger::println("MLIRCodeGen - Expression: '%s' | '%u' -> '%s'",
                         expression->toChars(), expression->op,
                         expression->type->toChars());
  auto arrayExp = expression->isArrayExp();
  auto arrayLengthExp = expression->isArrayLengthExp();
  auto vectorArrayExp = expression->isVectorArrayExp();
  auto vectorExp = expression->isVectorExp();
  auto assocArrayLiteralExp = expression->isAssocArrayLiteralExp();
  auto catExp = expression->isCatExp();
  auto catAssignExp = expression->isCatAssignExp();
  auto comExp = expression->isComExp();
  auto commaExp = expression->isCommaExp();
  auto compileExp = expression->isCompileExp();
  auto complexExp = expression->isComplexExp();
  auto Const = expression->isConst();
  auto delegateExp = expression->isDelegateExp();
  auto defaultExp = expression->isDefaultInitExp();
  auto delegateFuncPtrExp = expression->isDelegateFuncptrExp();
  auto delegatePtrExp = expression->isDelegatePtrExp();
  auto logical = expression->isLogicalExp();
  auto typeExp = expression->isTypeExp();
  auto typeIdExp = expression->isTypeidExp();

  LOG_SCOPE
  this->_total++;

  if (block != nullptr)
    builder.setInsertionPointToEnd(block);

  int op = expression->op;

  if (VarExp *varExp = expression->isVarExp())
    return mlirGen(varExp);
  else if (DeclarationExp *declarationExp = expression->isDeclarationExp())
    return mlirGen(declarationExp);
  else if (CastExp *castExp = expression->isCastExp())
    return mlirGen(castExp);
  else if (IntegerExp *integerExp = expression->isIntegerExp())
    return mlirGen(integerExp);
  else if (RealExp *realExp = expression->isRealExp())
    return mlirGen(realExp);
  else if (ConstructExp *constructExp = expression->isConstructExp())
    return mlirGen(constructExp);
  else if (AssignExp *assignExp = expression->isAssignExp())
    return mlirGen(assignExp);
  else if (CallExp *callExp = expression->isCallExp())
    return mlirGen(callExp);
  else if (ArrayLiteralExp *arrayLiteralExp = expression->isArrayLiteralExp())
    return mlirGen(arrayLiteralExp);
  else if (op >= 54 && op < 60)
    return mlirGen(expression, 1);
  else if (PostExp *postExp = expression->isPostExp())
    return mlirGen(postExp);
  else if (StringExp *stringExp = expression->isStringExp()) {
    return mlirGen(stringExp); // needs to implement with blocks
  } else if (LogicalExp *logicalExp = expression->isLogicalExp()) {
    _miss++;
    return nullptr; // add mlirGen(logicalExp); //needs to implement with blocks
  } else if (expression->isAddExp() || expression->isAddAssignExp())
    return mlirGen(expression->isAddExp(), expression->isAddAssignExp());
  else if (expression->isMinExp() || expression->isMinAssignExp())
    return mlirGen(expression->isMinExp(), expression->isMinAssignExp());
  else if (expression->isMulExp() || expression->isMulAssignExp())
    return mlirGen(expression->isMulExp(), expression->isMulAssignExp());
  else if (expression->isDivExp() || expression->isDivAssignExp())
    return mlirGen(expression->isDivExp(), expression->isDivAssignExp());
  else if (expression->isModExp() || expression->isModAssignExp())
    return mlirGen(expression->isModExp(), expression->isModAssignExp());
  else if (expression->isAndExp() || expression->isAndAssignExp())
    return mlirGen(expression->isAndExp(), expression->isAndAssignExp());
  else if (expression->isOrExp() || expression->isOrAssignExp())
    return mlirGen(expression->isOrExp(), expression->isOrAssignExp());
  else if (expression->isXorExp() || expression->isXorAssignExp())
    return mlirGen(expression->isXorExp(), expression->isXorAssignExp());
  else if (expression->isBlitExp()) {
    Logger::println("Expression is Blit");
    BinExp* bin = &*expression->isBlitExp();
    if (auto assign = static_cast<AssignExp*>(bin))
      return mlirGen(assign);
    mlirGen(expression->isBlitExp()->e1);
    auto Struct = mlirGen(expression->isBlitExp()->e2);
    declare(expression->isBlitExp()->e1->toChars(), Struct);
    return Struct;
  } else if (expression->isStructLiteralExp()) {
    return mlirGen(expression->isStructLiteralExp());
  } else if (DotVarExp *dotVarExp = expression->isDotVarExp()) {
    return mlirGen(dotVarExp);
  } else if (IndexExp *indexExp = expression->isIndexExp()) {
    mlirGen(indexExp);
  } else if (NewExp *newExp = expression->isNewExp()) {
    Logger::println("NewExpression not Implemented: %s", expression->toChars());
  } else if (SliceExp *sliceExp = expression->isSliceExp()) {
    return mlirGen(sliceExp);
  }

  _miss++;
  IF_LOG Logger::println("Unable to recoganize the Expression: '%s' : '%u': "
                         "'%s'",
                         expression->toChars(), expression->op,
                         expression->type->toChars());
  return nullptr;
}

Type *getNestedType(Expression *e, Type* pType = nullptr) {
  Type *type = nullptr;
  if ( e != nullptr && e->isArrayLiteralExp()) {
    auto elements = e->isArrayLiteralExp()->elements;

    if (elements->size() > 1)
      type = getNestedType(elements->front());
    else
      type = elements->front()->type;
  } else if (pType != nullptr && pType->toBasetype()->isTypeSArray()) {
    auto arraySType = pType->toBasetype()->isTypeSArray();
    if (arraySType->next->isTypeSArray())
      type = getNestedType(nullptr, arraySType->next);
    else
      type = arraySType->next;
  } else if (pType != nullptr && pType->toBasetype()->isTypeDArray()) {
    auto arrayDType = pType->toBasetype()->isTypeDArray();
    if (arrayDType->next->isTypeDArray())
      type = getNestedType(nullptr, arrayDType->next);
    else
      type = arrayDType->next;
  } else if (pType != nullptr && pType->toBasetype()->isTypeAArray()) {
    auto arrayAType = pType->toBasetype()->isTypeAArray();
    if (arrayAType->next->isTypeAArray())
      type = getNestedType(nullptr, arrayAType->next);
    else
      type = arrayAType->next;
  } else {
    type = e->type;
  }
  return type;
}

void getDim(Expression *e, std::vector<int64_t> *dim, Type* type = nullptr) {
  if (e != nullptr)
    Logger::println("Expression: %s", e->toChars());
  else if (type != nullptr)
    Logger::println("Type: %s", type->toChars());
  if (e == nullptr && type->isTypeSArray()) {
    auto typeSArray = type->isTypeSArray();

    if (auto nestedTypeSArray = typeSArray->next->isTypeSArray()) {
      getDim(nullptr, dim, nestedTypeSArray);
      dim->push_back(nestedTypeSArray->dim->toInteger());
      return;
    }

    dim->push_back(typeSArray->dim->toInteger());
  } else if (e->isArrayLiteralExp()){
  auto elements = e->isArrayLiteralExp()->elements;

  if (elements->size() > 1){
    getDim(elements->front(), dim);
    dim->push_back(elements->length);
    return;
  }

  dim->push_back(elements->length);
  }
  return;
}

std::vector<int64_t> getDims(Expression *e, Type *type = nullptr) {
  std::vector<int64_t> dims;
  if (e != nullptr && e->isArrayLiteralExp()) {
    auto array = e->isArrayLiteralExp();
    // Getting the first dimension
    dims.push_back(array->elements->length);

    // Append the nested dimensions to the current level
    if (auto nestedArray = array->elements->front()->isArrayLiteralExp())
      getDim(nestedArray, &dims);
  } else if (type != nullptr && type->isTypeSArray()){
    auto TypeSArray = type->isTypeSArray();
    // Getting the first dimension
    dims.push_back(TypeSArray->dim->toInteger());

    // Append the nested dimensions to the current level
    if (auto nestedArray = TypeSArray->next->isTypeSArray())
      getDim(nullptr, &dims, nestedArray);
  }

  return dims;
}

mlir::Type MLIRDeclaration::get_MLIRtype(Expression *expression = nullptr, Type *type) {
  if (expression == nullptr && type == nullptr)
    return mlir::NoneType::get(&context);

  _total++;

  Type *basetype;
  if (type != nullptr)
    basetype = type->toBasetype();
  else
    basetype = expression->type->toBasetype();

  if (basetype->ty == Tchar || basetype->ty == Twchar ||
      basetype->ty == Tdchar || basetype->ty == Tnull ||
      basetype->ty == Tvoid || basetype->ty == Tnone) {
    return mlir::NoneType::get(&context); // TODO: Build these types on DDialect
  } else if (basetype->ty == Tbool) {
    return builder.getIntegerType(1);
  } else if (basetype->ty == Tint8 || basetype->ty == Tuns8) {
    return builder.getIntegerType(8);
  } else if (basetype->ty == Tint16 || basetype->ty == Tuns16) {
    return builder.getIntegerType(16);
  } else if (basetype->ty == Tint32 || basetype->ty == Tuns32) {
    return builder.getIntegerType(32);
  } else if (basetype->ty == Tint64 || basetype->ty == Tuns64) {
    return builder.getIntegerType(64);
  } else if (basetype->ty == Tint128 || basetype->ty == Tuns128) {
    return builder.getIntegerType(128);
  } else if (basetype->ty == Tfloat32) {
    return builder.getF32Type();
  } else if (basetype->ty == Tfloat64) {
    return builder.getF64Type();
  } else if (basetype->ty == Tfloat80) {
    _miss++; // TODO: Build F80 type on DDialect
  } else if (basetype->ty == Tvector || basetype->ty == Tarray ||
             basetype->ty == Taarray) {
    std::vector<int64_t> dims;
    Type *nestedType = nullptr;
    mlir::RankedTensorType tensor;

    if (expression != nullptr && expression->isArrayLiteralExp()) {
      dims = getDims(expression->isArrayLiteralExp());
      nestedType = getNestedType(expression->isArrayLiteralExp());
      tensor = mlir::RankedTensorType::get(dims, get_MLIRtype(nullptr, nestedType));
    } else if ( basetype->isTypeDArray()) {
      dims = getDims(nullptr, basetype);
      nestedType = getNestedType(nullptr, basetype->isTypeDArray());
      tensor = mlir::RankedTensorType::get(dims, get_MLIRtype(nullptr, nestedType));
    }
    return tensor;
  } else if (basetype->ty == Tsarray) {
    std::vector<int64_t> dims;
    Type *nestedType = nullptr;
    mlir::RankedTensorType tensor;

    if (expression != nullptr && expression->isArrayLiteralExp()) {
      dims = getDims(expression->isArrayLiteralExp());
      nestedType = getNestedType(expression->isArrayLiteralExp());
    } else if ( basetype->isTypeSArray()) {
      dims = getDims(nullptr, basetype);
      nestedType = getNestedType(nullptr, basetype->isTypeSArray());
    }

    tensor =
        mlir::RankedTensorType::get(dims, get_MLIRtype(nullptr, nestedType));
    return tensor;
  } else if (basetype->ty == Tfunction) {
    TypeFunction *typeFunction = static_cast<TypeFunction *>(basetype);
    return get_MLIRtype(nullptr, typeFunction->next);
  } else if (basetype->ty == Tstruct) {
    auto varIt = structMap.lookup(basetype->toChars());
    if (!varIt.first)
      fatal();
    return structMap.lookup(basetype->toChars()).first;
  } else {
    _miss++;
    MLIRDeclaration declaration(module, context, builder, symbolTable,
                                structMap, _total, _miss);
    mlir::Value value = declaration.mlirGen(expression);
    return value.getType();
  }
  _miss++;
  return nullptr;
}

// Create the DSymbol for an MLIR Function with as many argument as the
// provided by Module
mlir::FuncOp MLIRDeclaration::mlirGen(FuncDeclaration *Fd, bool level) {

  // Assuming that the function will only return one value from it's type
  llvm::SmallVector<mlir::Type, 4> ret_types;

  if (!Fd->returns->empty()) {
    auto type = get_MLIRtype(nullptr, Fd->type);
    TypeFunction *funcType = static_cast<TypeFunction *>(Fd->type);
    auto ty = funcType->next->ty;
    if (ty != Tvector && ty != Tarray && ty != Tsarray && ty != Taarray) {
      auto dataType = mlir::RankedTensorType::get(1, type);
      ret_types.push_back(dataType);
    } else {
      auto tensorType = type.cast<mlir::TensorType>();
      ret_types.push_back(tensorType);
    }
  }

  unsigned long size = 0;
  if (Fd->parameters)
    size = Fd->parameters->length;

  // Arguments type is uniformly a generic array.
  llvm::SmallVector<mlir::Type, 4> arg_types;

  if (size) {
    for (auto var : *Fd->parameters) {
      auto type = get_MLIRtype(nullptr, var->type);
      auto ty = var->type->ty;
      if (ty != Tvector && ty != Tarray && ty != Tsarray && ty != Taarray) {
        auto dataType = mlir::RankedTensorType::get(1, type);
        arg_types.emplace_back(dataType);
      } else {
        auto tensorType = type.cast<mlir::TensorType>();
        arg_types.emplace_back(tensorType);
      }
    }
  } else {
    arg_types = llvm::SmallVector<mlir::Type, 4>(0, nullptr);
  }

  auto func_type = builder.getFunctionType(arg_types, ret_types);
  auto function = mlir::FuncOp::create(loc(Fd->loc), StringRef(Fd->toChars()),
                                       func_type, {});
  return function;
}

/// Emit a new function and add it to the MLIR module.
mlir::FuncOp MLIRDeclaration::mlirGen(FuncDeclaration *Fd) {
  // Create a scope in the symbol table to hold variable declarations.
  ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

  // MLIRFunction FuncDecl(Fd, context, builder, symbolTable, structMap,
  // total,
  //                       miss);
  // mlir::Type type = FuncDecl.DtoMLIRFunctionType(Fd, nullptr, nullptr);

  // Create an MLIR function for the given prototype.
  mlir::FuncOp function(mlirGen(Fd, true));
  if (!function)
    return nullptr;

  // Let's start the body of the function now!
  // In MLIR the entry block of the function is special: it must have the same
  // argument list as the function itself.
  auto &entryBlock = *function.addEntryBlock();

  // Set the insertion point in the builder to the beginning of the function
  // body, it will be used throughout the codegen to create operations in this
  // function.
  builder.setInsertionPointToStart(&entryBlock);

  // Initialize the object to be the "visitor"
//  MLIRStatements genStmt(module, context, builder, symbolTable, structMap,
//                         _total, _miss);

  // Setting arguments of a given function
  unsigned long size = 0;
  if (Fd->parameters)
    size = Fd->parameters->length;
  llvm::SmallVector<VarDeclarations *, 4> args(size, Fd->parameters);

  // args.push_back(mlirGen())
  auto &protoArgs = args;

  // Declare all the function arguments in the symbol table.
  for (auto name_value : llvm::zip(protoArgs, entryBlock.getArguments())) {
    if (failed(declare(std::get<0>(name_value)->pop()->toChars(),
                       std::get<1>(name_value))))
      return nullptr;
  }
  // Emit the body of the function.
//  if (mlir::failed(genStmt.genStatements(Fd))) {
//    function.erase();
//    fatal();
//  }
  //  function.getBody().back().back().getParentRegion()->viewGraph();

  // Implicitly return void if no return statement was emitted.
  // (this would possibly help the REPL case later)
/*  auto LastOp = function.getBody().back().back().getName().getStringRef();
  if (LastOp != "std.return" && LastOp != "std.br" &&
      LastOp != "std.cond_br") {

    function.getBody().back().back().dump();
    ReturnStatement *returnStatement = Fd->returns->front();
    if (returnStatement != nullptr)
      genStmt.mlirGen(returnStatement);
    else {
      builder.create<mlir::ReturnOp>(
          function.getBody().back().back().getLoc());
    }
  }*/
  return function;
}

#endif // LDC_MLIR_ENABLED
