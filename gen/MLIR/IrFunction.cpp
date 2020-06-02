//===-- IrFunction.h - Generate Declarations MLIR code ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "IrFunction.h"

MLIRFunction::MLIRFunction(
    FuncDeclaration *Fd, mlir::MLIRContext &context,
    const mlir::OpBuilder &builder,
    llvm::ScopedHashTable<StringRef, mlir::Value> &symbolTable,
    llvm::StringMap<std::pair<mlir::Type, StructDeclaration *>> &structMap,
    unsigned &total, unsigned &miss)
    : context(context), builder(builder), symbolTable(symbolTable),
      structMap(structMap), Fd(Fd), _total(total), _miss(miss) {}

MLIRFunction::~MLIRFunction() = default;

mlir::Value
MLIRFunction::DtoMLIRResolveFunction(FuncDeclaration *funcDeclaration) {
  if ((!global.params.useUnitTests || !funcDeclaration->type) &&
      funcDeclaration->isUnitTestDeclaration()) {
    IF_LOG Logger::println("Ignoring unittest %s",
                           funcDeclaration->toPrettyChars());
    return nullptr; // ignore declaration completely
  }

  Type *type = funcDeclaration->type;
  // If errors occurred compiling it, such as bugzilla 6118
  if (type && type->ty == Tfunction) {
    Type *nexty = static_cast<TypeFunction *>(type)->next;
    if (!nexty || nexty->ty == Terror) {
      return nullptr;
    }
  }

  if (funcDeclaration->parent) {
    if (TemplateInstance *templateInstance =
            funcDeclaration->parent->isTemplateInstance()) {
      if (TemplateDeclaration *templateDeclaration =
              templateInstance->tempdecl->isTemplateDeclaration()) {
        if (templateDeclaration->llvmInternal == LLVMva_arg) {
          Logger::println("magic va_arg found");
          funcDeclaration->llvmInternal = LLVMva_arg;
          return nullptr; // this gets mapped to an instruction so a declaration
          // makes no sense
        }
        if (templateDeclaration->llvmInternal == LLVMva_start) {
          Logger::println("magic va_start found");
          funcDeclaration->llvmInternal = LLVMva_start;
        } else if (templateDeclaration->llvmInternal == LLVMintrinsic) {
          Logger::println("overloaded intrinsic found");
          assert(funcDeclaration->llvmInternal == LLVMintrinsic);
          assert(funcDeclaration->mangleOverride.length);
        } else if (templateDeclaration->llvmInternal == LLVMinline_asm) {
          Logger::println("magic inline asm found");
          TypeFunction *tf = static_cast<TypeFunction *>(funcDeclaration->type);
          if (tf->parameterList.varargs != VARARGvariadic ||
              (funcDeclaration->parameters &&
               funcDeclaration->parameters->length != 0)) {
            templateDeclaration->error(
                "invalid `__asm` declaration, must be a D style "
                "variadic with no explicit parameters");
            fatal();
          }
          funcDeclaration->llvmInternal = LLVMinline_asm;
          return nullptr; // this gets mapped to a special inline asm call, no
          // point in going on.
        } else if (templateDeclaration->llvmInternal == LLVMinline_ir) {
          Logger::println("magic inline ir found");
          funcDeclaration->llvmInternal = LLVMinline_ir;
          funcDeclaration->linkage = LINKc;
          Type *type = funcDeclaration->type;
          assert(type->ty == Tfunction);
          static_cast<TypeFunction *>(type)->linkage = LINKc;

          DtoMLIRFunctionType(funcDeclaration, nullptr, nullptr);
          return nullptr; // this gets mapped to a special inline IR call, no
          // point in going on.
        }
      }
    }
  }

  DtoMLIRFunctionType(funcDeclaration, nullptr, nullptr);

  IF_LOG Logger::println("DtoResolveFunction(%s): %s",
                         funcDeclaration->toPrettyChars(),
                         funcDeclaration->loc.toChars());
  LOG_SCOPE;

  /*// queue declaration unless the function is abstract without body
  if (!funcDeclaration->isAbstract() || funcDeclaration->fbody) {
    DtoMLIRDeclareFunction(funcDeclaration);
  }*/
}

void DtoMLIRAddExtendAttr(Type *type, std::vector<mlir::Attribute> &attrs) {
  // This function uses std dialect to get zext and sext, there's no
  // benefit to have this op as D Dialect
  type = type->toBasetype();
  if (type->isintegral() && type->ty != Tvector && type->size() <= 2) {
    //  mlir::ZeroExtendIOp::
    // attrs.push_back(type->isunsigned() ?
    //                 mlir::SignExtendIOp);
    Logger::println("Zext: %d | Sext: %d", type->isunsigned(),
                    type->isunsigned());
  }
}

mlir::IntegerType MLIRFunction::DtoMLIRSize_t() {
  // the type of size_t does not change once set
  static mlir::IntegerType t = nullptr;
  if (t == nullptr) {
    auto triple = global.params.targetTriple;

    if (triple->isArch64Bit()) {
      t = builder.getIntegerType(64);
    } else if (triple->isArch32Bit()) {
      t = builder.getIntegerType(32);
    } else if (triple->isArch16Bit()) {
      t = builder.getIntegerType(16);
    } else {
      llvm_unreachable("Unsupported size_t width");
    }
  }
  return t;
}

mlir::FunctionType MLIRFunction::DtoMLIRFunctionType(FuncDeclaration *Fd,
                                                     Type *thistype,
                                                     Type *nesttype) {
  assert(Fd->type->ty == Tfunction);
  IF_LOG Logger::println("Getting Function Type for '%s': '%s'", Fd->toChars(),
                         Fd->type->toChars());

  // sanity check
  assert(Fd->type->ty == Tfunction);
  TypeFunction *funcType = static_cast<TypeFunction *>(Fd->type);

  assert(funcType->next && "Encountered function type with invalid return type;"
                           " trying to codegen function ignored by the "
                           "frontend?");

  TargetABI *abi = Fd && DtoIsIntrinsic(Fd) ? TargetABI::getIntrinsic() : gABI;

  // The index of the next argument on the LLVM level.
  unsigned nextLLArgIdx = 0;

  // functionType to be returned
  mlir::FunctionType functionType;

  // Type to be returned
  mlir::Type ret;

  // functionType name to be returned
  std::string ret_name;

  // Arguments
  std::vector<std::pair<mlir::IntegerType, mlir::Type>> args;

  // Logger::println("FuncType: %s  |  NextType: %s",
  // funcType->toChars(),next->toChars());
  const bool isMain = Fd && strncmp(Fd->toChars(), "main", 4) == 0;
  if (isMain) {
    // D and C main functions always return i32, even if declared as returning
    // void.

    ret = mlir::FunctionType::get({}, builder.getIntegerType(32), &context);
    ret_name = "i32";
  } else {
    Type *next = funcType->next;
    const bool byref = funcType->isref && next->toBasetype()->ty != Tvoid;
    std::vector<mlir::Attribute> attrs;

    if (abi->returnInArg(funcType, Fd && Fd->needThis())) {
      // sret return
      std::vector<mlir::Attribute> sretAttrs;
      // sretAttrs.push_back(mlir::Attr)
      Logger::println("StructType - Missing MLIRGen");
      _miss++;
      fatal();
    } else {
      // sext/zext return - TODO: What is this for?
      DtoMLIRAddExtendAttr(byref ? next->pointerTo() : next, attrs);
    }

    ret = get_MLIRtype(nullptr, next);
  }
  ++nextLLArgIdx;

  if (thistype) {
    // Add the this pointer for member functions
    Logger::println("arg_this - Missing MLIRGen");
    _miss++;
    fatal();
  } else if (nesttype) {
    // Add the context pointer for nested functions
    Logger::println("arg_nest - Missing MLIRGen");
    _miss++;
    fatal();
  }

  bool hasObjCSelector = false;
  if (Fd && Fd->linkage == LINKobjc && thistype) {
    if (Fd->selector) {
      hasObjCSelector = true;
    } else if (Fd->parent->isClassDeclaration()) {
      Fd->error("Objective-C `@selector` is missing");
    }
  }

  if (hasObjCSelector) {
    // TODO: make arg_objcselector to match dmd type
    Logger::println("arg_objcSelector  - Missing MLIRGen");
    ++nextLLArgIdx;
  }

  // Non-typesafe variadics (both C and D styles) are also variadics on the LLVM
  // level.
  const bool isLLVMVariadic =
      (funcType->parameterList.varargs == VARARGvariadic);
  if (isLLVMVariadic && funcType->linkage == LINKd) {
    // Add extra `_arguments` parameter for D-style variadic functions.
    Logger::println("arg_arguments: '%s' - Missing MLIRGen",
                    getTypeInfoType()->arrayOf()->toChars());
    ++nextLLArgIdx;
  }

  const size_t numExplicitDArgs = funcType->parameterList.length();

  // if this _Dmain() doesn't have an argument, we force it to have one
  if (isMain && funcType->linkage != LINKc && numExplicitDArgs == 0) {
    Type *mainargs = Type::tchar->arrayOf()->arrayOf();
    Logger::println("Missing mainargs MLIRGen: %s", mainargs->toChars());

    // newIrFty.args.push_back(new IrFuncTyArg(mainargs, false));
    ++nextLLArgIdx;
  }

  auto llvmInternal = Fd->llvmInternal;

  if (ret == nullptr) {
    ret = builder.getNoneType();
    ret_name = "NoneType";
  }

  IF_LOG Logger::cout() << "Final function type: " << ret_name << "\n";

  return functionType;
}

mlir::Type
MLIRFunction::DtoMLIRDeclareFunction(FuncDeclaration *funcDeclaration) {

  IF_LOG Logger::println("DtoDeclareFunction(%s): %s",
                         funcDeclaration->toPrettyChars(),
                         funcDeclaration->loc.toChars());
  LOG_SCOPE;

  if (funcDeclaration->isUnitTestDeclaration() && !global.params.useUnitTests) {
    Logger::println("unit tests not enabled");
    return nullptr;
  }

  // printf("declare function: %s\n", fdecl->toPrettyChars());

  // intrinsic sanity check
  if (DtoIsIntrinsic(funcDeclaration) && funcDeclaration->fbody) {
    error(funcDeclaration->loc, "intrinsics cannot have function bodies");
    fatal();
  }

  // Check if the function is defineAsExternallyAvailable

  // get TypeFunction*
  Type *t = funcDeclaration->type->toBasetype();
  TypeFunction *f = static_cast<TypeFunction *>(t);

  mlir::FuncOp vaFunc = nullptr;
  // if (DtoIsVaIntrinsic(funcDeclaration)) {
  //  vafunc = DtoDeclareVaFunction(funcDeclaration);
  //}

  // Calling convention.
  //
  // DMD treats _Dmain as having C calling convention and this has been
  // hardcoded into druntime, even if the frontend type has D linkage (Bugzilla
  // issue 9028).
  const bool forceC =
      DtoIsIntrinsic(funcDeclaration) || funcDeclaration->isMain();
  const auto link = forceC ? LINKc : f->linkage;

  // mangled name
  mlir::FunctionType functionType = static_cast<mlir::FunctionType>(
      DtoMLIRFunctionType(funcDeclaration, nullptr, nullptr));
  // mlir::FuncOp *func =
}

mlir::Type MLIRFunction::get_MLIRtype(Expression *expression, Type *type) {
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
    mlir::UnrankedTensorType tensor;
    return tensor;
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
  }
  Logger::print("Impossible to infer the type!");
  _miss++;
  fatal();
  return nullptr;
}
#endif // LDC_MLIR_ENABLED
