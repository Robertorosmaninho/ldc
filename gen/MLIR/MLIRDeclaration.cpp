//===-- MLIRDeclaration.cpp -----------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED
#include "MLIRDeclaration.h"
#include "llvm/ADT/APFloat.h"

MLIRDeclaration::MLIRDeclaration(
    Module *m, mlir::MLIRContext &context, mlir::OpBuilder builder,
    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
    llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
    unsigned &total, unsigned &miss)
    : module(m), context(context), builder(builder), symbolTable(symbolTable),
      structMap(structMap), _total(total), _miss(miss) {} // Constructor

MLIRDeclaration::~MLIRDeclaration() = default;

int isInteger(mlir::Type type) {
  if (type.isInteger(1))
    return 1;
  else if (type.isInteger(8))
    return 8;
  else if (type.isInteger(16))
    return 16;
  else if (type.isInteger(32))
    return 32;
  else if (type.isInteger(64))
    return 64;
  else
    return 0;
}

int isReal(mlir::Type type) {
  if (type.isF16())
    return 16;
  else if (type.isF32())
    return 32;
  else if (type.isF64())
    return 64;
  else
    return 0;
}

void collectIntData(std::vector<mlir::APInt> &data,
                    ArrayLiteralExp *arrayLiteralExp, int size) {
  for (auto element : *arrayLiteralExp->elements) {
    if (element->isArrayLiteralExp())
      collectIntData(data, element->isArrayLiteralExp(), size);
    else
      data.push_back(APInt(size, element->toInteger()));
  }
}

void collectFloatData(std::vector<mlir::APFloat> &data, ArrayLiteralExp
*arrayLiteralExp, bool isDouble) {
  for (auto element : *arrayLiteralExp->elements) {
    if (element->isArrayLiteralExp())
      collectFloatData(data, element->isArrayLiteralExp(), isDouble);
    else if (isDouble)
      data.push_back(APFloat((double)element->toReal()));
    else
      data.push_back(APFloat((float)element->toReal()));
  }
}

mlir::Type hasSameTypeAndDims(mlir::Value e1, mlir::Value e2) {

  auto type1 = e1.getType();
  auto type2 = e2.getType();

  if (auto isTensor = type1.template isa<mlir::TensorType>()) {
    type1 = type1.cast<mlir::TensorType>().getElementType();

    if (auto isTensor2 = type2.template isa<mlir::TensorType>()) {
      type2 = type2.cast<mlir::TensorType>().getElementType();
    }

    // Checking if they have the same dimensions
    assert(e1.getType().cast<mlir::TensorType>().getRank() ==
           e2.getType().cast<mlir::TensorType>().getRank());
  }

  if (isReal(type1) && isReal(type2)) {
    return mlir::FloatType::getF16(e1.getContext());
  } else if (isInteger(type1) && isInteger(type2)) {
    return mlir::IntegerType::get(e1.getContext(), 1);
  }

  return mlir::NoneType::get(e1.getContext());
}

/*mlir::DenseElementsAttr MLIRDeclaration::getConstantAttr(Expression *exp) {
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
*/
/*std::pair<mlir::ArrayAttr, mlir::Type>
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
}*/

mlir::Value MLIRDeclaration::mlirGen(Declaration *declaration) {
  IF_LOG Logger::println("MLIRCodeGen - Declaration: '%s'",
                         declaration->toChars());
  LOG_SCOPE

  /*if (StructDeclaration *structDeclaration =
          declaration->isStructDeclaration()) {
    if (failed(mlirGen(structDeclaration, 0)))
      return nullptr;
  } else */if (auto varDeclaration = declaration->isVarDeclaration()) {
    return mlirGen(varDeclaration);
  } else {
    IF_LOG Logger::println("Unable to recoganize Declaration: '%s'",
                           declaration->toChars());
    _miss++;
    return nullptr;
  }
}

/*mlir::LogicalResult
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
}*/

mlir::Value MLIRDeclaration::mlirGen(VarDeclaration *vd) {
  IF_LOG Logger::println("MLIRCodeGen - VarDeclaration: '%s'", vd->toChars());
  LOG_SCOPE
  _total++;
  // if aliassym is set, this VarDecl is redone as an alias to another symbol
  // this seems to be done to rewrite Tuple!(...) v;
  // as a TupleDecl that contains a bunch of individual VarDecls
  if (vd->aliassym) {
    IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: aliassym");
    // return DtoDeclarationExpMLIR(vd->aliassym, mlir_);
  }
  if (vd->isDataseg()) {
    IF_LOG Logger::println("MLIRCodeGen -  VarDeclaration: dataseg");
    // Declaration_MLIRcodegen(vd, mlir_);
  } else {
    if (vd->nestedrefs.length) {
      IF_LOG Logger::println(
          "has nestedref set (referenced by nested function/delegate)");

      // A variable may not be really nested even if nextedrefs is not empty
      // in case it is referenced by a function inside __traits(compile) or
      // typeof.
      // assert(vd->ir->irLocal && "irLocal is expected to be already set by
      // DtoCreateNestedContext");
    }

    if (vd->_init) {
      if (ExpInitializer *ex = vd->_init->isExpInitializer()) {
        // TODO: Refactor this so that it doesn't look like toElem has no
        // effect.
        Logger::println("MLIRCodeGen - ExpInitializer: '%s'", ex->toChars());
        LOG_SCOPE
        return mlirGen(ex->exp);
      }
    } else {
      IF_LOG Logger::println("Unable to recoganize VarDeclaration: '%s'",
                             vd->toChars());
    }
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

  auto type = e1.getType();
  if (auto isTensor = type.template isa<mlir::TensorType>()) {
    type = type.cast<mlir::TensorType>().getElementType();
  }

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

  if (hasSameTypeAndDims(e1, e2) == mlir::IntegerType::get(&context, 1)) {
    result = builder.create<mlir::D::AndOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
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
  mlir::Type elementType = get_MLIRtype(arrayLiteralExp);
  mlir::TensorType tensorType = nullptr;
  mlir::Location Loc = loc(arrayLiteralExp->loc);

  if (auto isTensor = elementType.template isa<mlir::TensorType>()) {
    tensorType = elementType.cast<mlir::TensorType>();
    elementType = tensorType.getElementType();
  }

  int isFloat = isReal(elementType);
  int size = isInteger(elementType);

  if (!isFloat && !size)
    IF_LOG Logger::println("MLIR doesn't support integer of type different "
                           "from 1,8,16,32,64");

  std::vector<mlir::APInt> data;
  std::vector<mlir::APFloat> dataF; // Used by float and double

  // Set the actual attribute that holds the list of values for this
  // tensor literal and build the operation.
  if (isFloat == 64) {
    collectFloatData(dataF, arrayLiteralExp, isFloat == 64);
    auto dataAttributes = mlir::DenseElementsAttr::get(tensorType, dataF);
    return builder.create<mlir::D::DoubleOp>(Loc, tensorType, dataAttributes);
  } else if (isFloat) {
    collectFloatData(dataF, arrayLiteralExp, isFloat == 64);
    auto dataAttributes = mlir::DenseElementsAttr::get(tensorType, dataF);
    return builder.create<mlir::D::FloatOp>(Loc, tensorType, dataAttributes);
  } else if (size) {
    collectIntData(data, arrayLiteralExp, size);
    auto dataAttributes = mlir::DenseElementsAttr::get(tensorType, data);
    return builder.create<mlir::D::IntegerOp>(Loc, tensorType, dataAttributes);
  } else {
    _miss++;
    Logger::println("Unable to build ArrayLiteralExp: '%s'",
                    arrayLiteralExp->toChars());
  }
}

mlir::Value MLIRDeclaration::mlirGen(AssignExp *assignExp) {
  _total++;
  IF_LOG Logger::print(
      "AssignExp::toElem: %s | (%s)(%s = %s)\n", assignExp->toChars(),
      assignExp->type->toChars(), assignExp->e1->type->toChars(),
      assignExp->e2->type ? assignExp->e2->type->toChars() : nullptr);
  if (static_cast<int>(assignExp->memset) & static_cast<int>
      (MemorySet::referenceInit)) {
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

  if (assignExp->e1->isVarExp() || assignExp->e1->isDotVarExp())
    lhs = mlirGen(assignExp->e1);
  else {
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

/*mlir::Value MLIRDeclaration::mlirGen(CallExp *callExp) {
  IF_LOG Logger::println("MLIRCodeGen - CallExp: '%s'", callExp->toChars());
  LOG_SCOPE
  _total++;

  VarDeclaration *delayedDtorVar = nullptr;
  Expression *delayedDtorExp = nullptr;

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
    if (ty != Tvector && ty != Tarray && ty != Tsarray && ty != Taarray) {
      auto dataType = mlir::RankedTensorType::get(1, type);
      types.push_back(dataType);
    } else {
      auto tensorType = type.cast<mlir::TensorType>();
      types.push_back(tensorType);
    }
  }

  llvm::ArrayRef<mlir::Type> ret_type(types);
  return builder.create<mlir::D::CallOp>(loc(callExp->loc), functionName,
                                         ret_type, operands);
}*/

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
  assert(result != nullptr);
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
  /*else if (StructDeclaration *structDeclaration = dsym->isStructDeclaration())
    if (failed(mlirGen(structDeclaration, 0)))
      return nullptr;*/

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

  if (hasSameTypeAndDims(e1, e2) == mlir::FloatType::getF16(&context)) {
    result = builder.create<mlir::D::DivFOp>(*location, e1, e2).getResult();
  } else if (hasSameTypeAndDims(e1, e2) == mlir::IntegerType::get(&context,1)) {
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
    /*else if (e1type->ty == Tstruct) {
//      auto *ts = static_cast<TypeStruct *>(e1type);
      llvm::Optional<size_t> accessIndex = getMemberIndex(dotVarExp);
      if (!accessIndex) {
        emitError(location, "invalid access into struct expression");
        return nullptr;
      }
      auto lhs = mlirGen(dotVarExp->e1);
      return builder.create<mlir::D::StructAccessOp>(location, lhs,
                                                     *accessIndex);
    }*/
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

  mlir::Type type1 = e1.getType();
  if (auto isTensor = type1.template isa<mlir::TensorType>()) {
    auto tensorType = type1.cast<mlir::TensorType>();

    int rank = tensorType.getRank();
    int dim = tensorType.getDimSize(0);

    if (rank == 1 && dim == 1) {
      e1 = builder.create<mlir::ConstantOp>(location, e1.getDefiningOp()->getAttr("value").cast<mlir::DenseElementsAttr>().getValue(0));
      e2 = builder.create<mlir::ConstantOp>(location, e2.getDefiningOp()->getAttr("value").cast<mlir::DenseElementsAttr>().getValue(0));
    }
  }
  bool isFloat = isReal(type1);

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

mlir::Value MLIRDeclaration::mlirGen(IndexExp *indexExp) {
  IF_LOG Logger::print("MLIRCodeGen -IndexExp: %s @ %s\n", indexExp->toChars(),
                       indexExp->type->toChars());
  LOG_SCOPE;
  auto location = loc(indexExp->loc);

  Logger::print("e1 ou l: %s\n", indexExp->e1->toChars());
  Logger::print("e2 ou r: %s\n",indexExp->e2->toChars());

  // Generation or getting the array
  auto l = mlirGen(indexExp->e1);

  if (!l) {
    IF_LOG Logger::println("Expression: %s", indexExp->toChars());
    llvm_unreachable("Unknown IndexExp target.");
  }

  // At this point we hope that the tensor has only one dimension
  mlir::TensorType tensorType = l.getType().cast<mlir::TensorType>();
  int dims = tensorType.getDimSize(0);

  // Index in D is usually `cast(ulong)i` so we skip the cast part to get the
  // value
  mlir::Operation *op = mlirGen(indexExp->e2->isCastExp()->e1).getDefiningOp();
  auto denseValues = op->getAttr("value").cast<mlir::DenseElementsAttr>().getIntValues();
  mlir::APInt value = denseValues.begin().operator*();

  // Finally we extract the value that we can use
  int index = value.getSExtValue();

  if (index > dims)
    llvm_unreachable("index out of bounds");

  op = l.getDefiningOp();
  if (auto attr = op->getAttr("value").cast<mlir::DenseElementsAttr>()) {
    auto attrValue = attr.getValue(index);
    auto attrType = mlir::RankedTensorType::get(1, tensorType.getElementType());

    int sizeInt = isInteger(tensorType.getElementType());
    int sizeFloat = isReal(tensorType.getElementType());

    if (sizeInt) {
      return builder.create<mlir::D::IntegerOp>(location, attrType, attrValue);
    } else if (sizeFloat == 64) {
      return builder.create<mlir::D::FloatOp>(location, attrType, attrValue);
    } else if (sizeFloat){
      return builder.create<mlir::D::DoubleOp>(location, attrType, attrValue);
    } else {
      _miss++;
      llvm_unreachable("unkown operation");
    }

  }
  else
    llvm_unreachable("unkown operation");
}

mlir::Value MLIRDeclaration::mlirGen(IntegerExp *integerExp) {
  _total++;
  dinteger_t dint = integerExp->value;
  Logger::println("MLIRGen - Integer: '%lu'", dint);
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
  return builder.create<mlir::D::IntegerOp>(location, shapedType,
                                            dataAttribute);
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

  if (hasSameTypeAndDims(e1,e2) == mlir::FloatType::getF16(&context)) {
    result = builder.create<mlir::D::SubFOp>(*location, e1, e2).getResult();
  } else if (hasSameTypeAndDims(e1,e2) == mlir::IntegerType::get(&context, 1)) {
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

  if (hasSameTypeAndDims(e1, e2) == mlir::FloatType::getF16(&context)) {
    result = builder.create<mlir::D::ModFOp>(*location, e1, e2).getResult();
  } else if (hasSameTypeAndDims(e1, e2) == mlir::IntegerType::get(&context, 1)) {
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

  if (hasSameTypeAndDims(e1, e2) == mlir::FloatType::getF16(&context)) {
    result = builder.create<mlir::D::MulFOp>(*location, e1, e2).getResult();
  } else if (hasSameTypeAndDims(e1,e2) == mlir::IntegerType::get(&context, 1)) {
    result = builder.create<mlir::D::MulOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
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

  if (hasSameTypeAndDims(e1, e2) == mlir::IntegerType::get(&context, 1)) {
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
  mlir::Type type1 = e1.getType();

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

  if (auto isTensor = type1.template isa<mlir::TensorType>()) {
    type1 = type1.cast<mlir::TensorType>().getElementType();
  }
  auto shapedType = mlir::RankedTensorType::get(1, type1);
  auto dataAttribute = mlir::DenseElementsAttr::get(shapedType, 1);
  if (type1.isF32() || type1.isF16())
    e2 = builder.create<mlir::D::FloatOp>(location, shapedType, dataAttribute);
  else if (type1.isF64())
    e2 = builder.create<mlir::D::DoubleOp>(location, shapedType, dataAttribute);
  else
    e2 = builder.create<mlir::D::IntegerOp>(location, shapedType, dataAttribute);

  if (postExp->op == TOKplusplus) {
    if (type1.isF32() || type1.isF16() || type1.isF64())
      return builder.create<mlir::D::AddFOp>(location, e1, e2);
    else
      return builder.create<mlir::D::AddOp>(location, e1, e2);
  } else if (postExp->op == TOKminusminus) {
    if (type1.isF32() || type1.isF16() || type1.isF64())
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
  IF_LOG Logger::println("MLIRCodeGen - StringExp: '%s'", stringExp->toChars());

  // TODO: MAKE STRING AS DIALECT TYPE
  mlir::OperationState result(loc(stringExp->loc), "ldc.String");
  result.addAttribute("Value",
                      builder.getStringAttr(StringRef(stringExp->toChars())));
  result.addTypes(mlir::RankedTensorType::get(stringExp->len + 1,
                                              builder.getIntegerType(8)));
  return builder.createOperation(result)->getResult(0);
}

/*mlir::Value MLIRDeclaration::mlirGen(StructLiteralExp *structLiteralExp) {
  IF_LOG Logger::println("MLIRCodeGen - StructLiteralExp: '%s'",
                         structLiteralExp->toChars());

  mlir::ArrayAttr dataAttr;
  mlir::Type dataType;
  std::tie(dataAttr, dataType) = getConstantAttr(structLiteralExp);

  // Build the MLIR op `toy.struct_constant`. This invokes the
  // `StructConstantOp::build` method.
  return builder.create<mlir::D::StructConstantOp>(loc(structLiteralExp->loc),
                                                   dataType, dataAttr);
}*/

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

  /*if (auto fd = varExp->var->isFuncLiteralDeclaration()) {
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
    return builder.create<mlir::D::IntegerOp>(loc(varExp->loc), shapedType, dataAttribute);
  } /*else if (varExp->type->ty == Tstruct) {
    if (StructLiteralExp *sle = varExp->isStructLiteralExp()) {
      return mlirGen(sle);
    }

    auto structDecl = structMap.lookup(varExp->var->toChars());
    if (!structDecl.second)
      return nullptr;

    if (!symbolTable.lookup(varExp->var->toChars()))
      mlirGen(structDecl.second, 1);

    return symbolTable.lookup(varExp->var->toChars());
  }*/

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
    return DeclFunc.getMLIRFunction().getOperation()->getResult(0);
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

  if (hasSameTypeAndDims(e1,e2) == mlir::IntegerType::get(&context, 1)) {
    result = builder.create<mlir::D::XorOp>(*location, e1, e2).getResult();
  } else {
    _miss++;
    return nullptr;
  }

  return result;
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

  // FIXME: This is #673 all over again.
  if (!global.params.linkonceTemplates && !decl->needsCodegen()) {
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
  else if (IndexExp *indexExp = expression->isIndexExp())
    return mlirGen(indexExp);
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
    auto Struct = mlirGen(expression->isBlitExp()->e2);
    declare(expression->isBlitExp()->e1->toChars(), Struct);
    return Struct;
  //} else if (expression->isStructLiteralExp()) {
 //   return mlirGen(expression->isStructLiteralExp());
  } else if (DotVarExp *dotVarExp = expression->isDotVarExp()) {
    return mlirGen(dotVarExp);
  }

  _miss++;
  IF_LOG Logger::println("Unable to recoganize the Expression: '%s' : '%u': "
                         "'%s'",
                         expression->toChars(), expression->op,
                         expression->type->toChars());
  return nullptr;
}

mlir::Type MLIRDeclaration::get_MLIRtype(Expression *expression, Type *type) {
  if (expression == nullptr && type == nullptr)
    return mlir::NoneType::get(&context);

  _total++;

  Type *basetype;
  if (type != nullptr)
    basetype = type->toBasetype();
  else
    basetype = expression->type->toBasetype();

  IF_LOG Logger::println("MLIRCodeGen - Getting type `%s`",
                         basetype->toChars());

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
    //auto e1 = expression->isArrayExp();
    auto e2 = expression->isArrayLiteralExp();
    //auto e3 = expression->isVectorExp();
    //auto e4 = expression->isVectorArrayExp();
    std::vector<int64_t> shape;
    auto size = e2->elements->size();
    shape.push_back(size);

    auto elementType = get_MLIRtype(e2->elements->front());
    if (bool isTensor = elementType.template isa<mlir::TensorType>()) {
      auto tensorType = elementType.cast<mlir::TensorType>();
      auto insideShape = tensorType.getShape();
      for (auto value : insideShape)
        shape.push_back(value);
      elementType = tensorType.getElementType();
    }

    auto shapedType = mlir::RankedTensorType::get(llvm::makeArrayRef(shape),
                                                  elementType);
    return shapedType;
  } else if (basetype->ty == Tsarray) {
    auto size = basetype->isTypeSArray()->dim->toInteger();
    return mlir::RankedTensorType::get(
        size, get_MLIRtype(nullptr, type->isTypeSArray()->next));
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
