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

namespace {

void onlyOneMainCheck(FuncDeclaration *fd) {
  if (!fd->fbody) // multiple *declarations* are fine
    return;

  // We'd actually want all possible main functions to be mutually exclusive.
  // Unfortunately, a D main implies a C main, so only check C mains with
  // -betterC.
  if (fd->isMain() || (global.params.betterC && fd->isCMain()) ||
      (global.params.isWindows && (fd->isWinMain() || fd->isDllMain()))) {
    // global - across all modules compiled in this compiler invocation
    static Loc mainLoc;
    if (!mainLoc.filename) {
      mainLoc = fd->loc;
      assert(mainLoc.filename);
    } else {
      const char *otherMainNames =
          global.params.isWindows ? ", `WinMain`, or `DllMain`" : "";
      const char *mainSwitch =
          global.params.addMain ? ", -main switch added another `main()`" : "";
      error(fd->loc,
            "only one `main`%s allowed%s. Previously found `main` at %s",
            otherMainNames, mainSwitch, mainLoc.toChars());
    }
  }
}

} // anonymous namespace

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

          //DtoMLIRFunctionType(funcDeclaration, nullptr, nullptr);
          return nullptr; // this gets mapped to a special inline IR call, no
          // point in going on.
        }
      }
    }
  }

  DtoMLIRFunctionType(funcDeclaration);

  IF_LOG Logger::println("DtoResolveFunction(%s): %s",
                         funcDeclaration->toPrettyChars(),
                         funcDeclaration->loc.toChars());
  LOG_SCOPE;

  // queue declaration unless the function is abstract without body
  if (!funcDeclaration->isAbstract() || funcDeclaration->fbody) {
    DtoMLIRDeclareFunction(funcDeclaration);
  }
}

mlir::FunctionType MLIRFunction::DtoMLIRFunctionType(FuncDeclaration *Fd) {
  // handle C vararg intrinsics
  // if (DtoIsVaIntrinsic(Fd)) {
  //     return DtoVaFunctionType(Fd);
  // }

  Type *dthis = nullptr, *dnest = nullptr;

  if (Fd->ident == Id::ensure || Fd->ident == Id::require) {
    FuncDeclaration *p = Fd->parent->isFuncDeclaration();
    assert(p);
    AggregateDeclaration *ad = p->isMember2();
    (void)ad;
    assert(ad);
    dnest = Type::tvoid->pointerTo();
  } else if (Fd->needThis()) {
    if (AggregateDeclaration *ad = Fd->isMember2()) {
      IF_LOG Logger::println("isMember = this is: %s", ad->type->toChars());
      dthis = ad->type;
      // TODO: LLType *thisty = DtoType(dthis);
      // Logger::cout() << "this llvm type: " << *thisty << '\n';
      // if (ad->isStructDeclaration()) {
      // TODO:   thisty = getPtrToType(thisty);
      // }
    } else {
      IF_LOG Logger::println("chars: %s type: %s kind: %s", Fd->toChars(),
                             Fd->type->toChars(), Fd->kind());
      Fd->error("requires a dual-context, which is not yet supported by LDC");
      if (!global.gag)
        fatal();
      // TODO: Needs to be tested
      return mlir::FunctionType::get({}, {}, &context);
      // return LLFunctionType::get(LLType::getVoidTy(gIR->context()),
      //    /*isVarArg=*/false);
    }
  } else if (Fd->isNested()) {
    dnest = Type::tvoid->pointerTo();
  }

  mlir::FunctionType funcType = DtoMLIRFunctionType(
      Fd->type, getIrFunc(Fd, true)->irFty, dthis, dnest, Fd);

  return funcType;
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
    fatal();
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

mlir::FunctionType
MLIRFunction::DtoMLIRFunctionType(Type *type, IrFuncTy &irFty, Type *thistype,
                                  Type *nesttype, FuncDeclaration *Fd) {
  IF_LOG Logger::println("DtoFunctionType(%s)", type->toChars());
  LOG_SCOPE

  assert(Fd->type->ty == Tfunction);
  IF_LOG Logger::println("Getting Function Type for '%s': '%s'", Fd->toChars(),
                         Fd->type->toChars());

  // sanity check
  assert(Fd->type->ty == Tfunction);
  TypeFunction *f = static_cast<TypeFunction *>(type);

  assert(f->next && "Encountered function type with invalid return type; "
                    "trying to codegen function ignored by the frontend?");

  TargetABI *abi = Fd && DtoIsIntrinsic(Fd) ? TargetABI::getIntrinsic() : gABI;

  // Do not modify irFty yet; this function may be called recursively if any
  // of the argument types refer to this type.
  IrFuncTy newIrFty(f);

  // The index of the next argument on the LLVM level.
  unsigned nextLLArgIdx = 0;

  // functionType to be returned
  mlir::FunctionType functionType;

  // Type to be returned
  mlir::Type ret;

  // functionType name to be returned
  std::string ret_name;

  // Arguments
  std::vector<mlir::Type> args;

  // Logger::println("FuncType: %s  |  NextType: %s",
  // funcType->toChars(),next->toChars());
  const bool isMain = Fd && strncmp(Fd->toChars(), "main", 4) == 0;
  if (isMain) {
    // D and C main functions always return i32, even if declared as returning
    // void.

    ret = mlir::FunctionType::get({}, builder.getIntegerType(32), &context);
    ret_name = "i32";
  } else {
    Type *rt = f->next;
    ret_name = rt->toChars();
    const bool byref = f->isref && rt->toBasetype()->ty != Tvoid;
    std::vector<mlir::Attribute> attrs;

    if (abi->returnInArg(f, Fd && Fd->needThis())) {
      // sret return
      std::vector<mlir::Attribute> sretAttrs;
      // sretAttrs.push_back(mlir::Attr)
      Logger::println("StructType - Missing MLIRGen");
      _miss++;
      fatal();
    } else {
      // sext/zext return - TODO: What is this for?
      DtoMLIRAddExtendAttr(byref ? rt->pointerTo() : rt, attrs);
    }

    ret = get_MLIRtype(nullptr, rt);
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
  const bool isLLVMVariadic = (f->parameterList.varargs == VARARGvariadic);
  if (isLLVMVariadic && f->linkage == LINKd) {
    // Add extra `_arguments` parameter for D-style variadic functions.
    Logger::println("arg_arguments: '%s' - Missing MLIRGen",
                    getTypeInfoType()->arrayOf()->toChars());
    fatal();
    ++nextLLArgIdx;
  }

  const size_t numExplicitDArgs = f->parameterList.length();

  // if this _Dmain() doesn't have an argument, we force it to have one
  if (isMain && f->linkage != LINKc && numExplicitDArgs == 0) {
    Type *mainargs = Type::tchar->arrayOf()->arrayOf();
    auto type0 = get_MLIRtype(nullptr, mainargs);
    if (!type0.template isa<mlir::TensorType>())
      type0 = mlir::RankedTensorType::get(1, type0);
    args.push_back(get_MLIRtype(nullptr, mainargs));
    //++nextLLArgIdx;
  }

  for (size_t i = 0; i < numExplicitDArgs; ++i) {
    Parameter *arg = Parameter::getNth(f->parameterList.parameters, i);

    // Whether the parameter is passed by LLVM value or as a pointer to the
    // alloca/….
    bool passPointer = arg->storageClass & (STCref | STCout);

    Type *loweredDType = arg->type;
    std::vector<mlir::Type> sretAttrs;
    if (arg->storageClass & STClazy) {
      // Lazy arguments are lowered to delegates.
      Logger::println("lazy param");
      // auto ltf = TypeFunction::create(nullptr, arg->type, VARARGnone, LINKd);
      // auto ltd = createTypeDelegate(ltf);
      // loweredDType = merge(ltd);
      fatal();
    } else if (passPointer) {
      // ref/out
      Logger::println("passPointer: ref/out");
      fatal();
      // attrs.addDereferenceableAttr(loweredDType->size());
    } else {
      if (abi->passByVal(f, loweredDType)) {
        // LLVM ByVal parameters are pointers to a copy in the function
        // parameters stack. The caller needs to provide a pointer to the
        // original argument.
        // attrs.addAttribute(LLAttribute::ByVal);
        Logger::println("Needs to implement addAttribute(LLAttribute::ByVal)");
        fatal();
        // if (auto alignment = DtoAlignment(loweredDType))
        //  attrs.addAlignmentAttr(alignment);
        passPointer = true;
      } else {
        // Add sext/zext as needed.
        DtoMLIRAddExtendAttr(loweredDType, attrs);
      }
    }

    auto type0 = get_MLIRtype(nullptr, loweredDType);
    if (!type0.template isa<mlir::TensorType>())
      type0 = mlir::RankedTensorType::get(1, type0);
    sretAttrs.push_back(type0);
    for (unsigned int j = 0; j < nextLLArgIdx; ++j)
      args.push_back(sretAttrs[j]);

    ++nextLLArgIdx;
  }



  functionType = mlir::FunctionType::get(args, {}, &context);

  if (ret == nullptr) {
    ret = builder.getNoneType();
    ret_name = "NoneType";
  }

  IF_LOG Logger::cout() << "Final function type: " << ret_name  << "\n";

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

  // Check if fdecl should be defined too for cross-module inlining.
  // If true, semantic is fully done for fdecl which is needed for some code
  // below (e.g. code that uses fdecl->vthis).
  /*const bool defineAtEnd = && defineAsExternallyAvailable(*funcDeclaration);
  if (defineAtEnd) {
    IF_LOG Logger::println(
          "Function is an externally_available inline candidate.");
  }*/

  // get TypeFunction*
  Type *t = funcDeclaration->type->toBasetype();
  TypeFunction *f = static_cast<TypeFunction *>(t);
  mlir::FuncOp vafunc = nullptr;
  /*if (DtoIsVaIntrinsic(fdecl)) {
    vafunc = DtoDeclareVaFunction(fdecl);
  }*/

  // create IrFunction
  IrFunction *irFunc = getIrFunc(funcDeclaration, true);

  // Calling convention.
  //
  // DMD treats _Dmain as having C calling convention and this has been
  // hardcoded into druntime, even if the frontend type has D linkage (Bugzilla
  // issue 9028).
  const bool forceC =
      DtoIsIntrinsic(funcDeclaration) || funcDeclaration->isMain();
  const auto link = forceC ? LINKc : f->linkage;

  // mangled name
  const auto irMangle = getIRMangledName(funcDeclaration, link);

  // construct function
  mlir::FunctionType functype = DtoMLIRFunctionType(funcDeclaration);
  mlir::FuncOp func = vafunc;
  // LLFunction *func = vafunc ? vafunc : gIR->module.getFunction(irMangle);

  if (!func) {
    // All function declarations are "external" - any other linkage type
    // is set when actually defining the function, except extern_weak.
    auto linkage = llvm::GlobalValue::ExternalLinkage;
    // Apply pragma(LDC_extern_weak)
    if (funcDeclaration->llvmInternal == LLVMextern_weak)
      linkage = llvm::GlobalValue::ExternalWeakLinkage;
    auto point = builder.getInsertionPoint();
    builder.setInsertionPointToEnd(builder.getInsertionBlock());
    func =
        mlir::FuncOp::create(loc(funcDeclaration->loc), irMangle, functype, {});
    builder.setInsertionPoint(builder.getInsertionBlock(), point);
  } else if (func.getType() != functype) {
   // const auto existingTypeString = llvmTypeToString(func.getType());
   // const auto newTypeString = llvmTypeToString(functype);
    error(funcDeclaration->loc,
          "Function type does not match previously declared "
          "function with the same mangled name: `%s`",
          mangleExact(funcDeclaration)
   // errorSupplemental(funcDeclaration->loc, "Previous IR type: %s",
   //                   existingTypeString.c_str());
   // errorSupplemental(funcDeclaration->loc, "New IR type:      %s",
   //                   newTypeString.c_str());
    );
    fatal();
  }

  /*if (global.params.isWindows && fdecl->isExport()) {
    func->setDLLStorageClass(fdecl->isImportedSymbol()
                             ? LLGlobalValue::DLLImportStorageClass
                             : LLGlobalValue::DLLExportStorageClass);
  }*/

  IF_LOG Logger::println("func = %s", func.getName().data());

  // add func to IRFunc
  function = func;

  function.dump();
  //There are not any "NonLazyBind Attribute on MLIR"
  // if (!fdecl->fbody && opts::noPLT) {
    // Add `NonLazyBind` attribute to function declarations,
    // the codegen options allow skipping PLT.
   // func->addFnAttr(LLAttribute::NonLazyBind);
  // }

  // Detect multiple main function definitions, which is disallowed.
  // DMD checks this in the glue code, so we need to do it here as well.
  onlyOneMainCheck(funcDeclaration);

  // Set inlining attribute
  /*if (fdecl->neverInline) { TODO: Inline is not set yet
    irFunc->setNeverInline();
  } else {
    if (fdecl->inlining == PINLINEalways) {
      irFunc->setAlwaysInline();
    } else if (fdecl->inlining == PINLINEnever) {
      irFunc->setNeverInline();
    }
  }*/

  // Do not exist in mlir
  /*if (fdecl->isCrtCtorDtor & 1) {
    AppendFunctionToLLVMGlobalCtorsDtors(func, fdecl->priority, true);
  }
  if (fdecl->isCrtCtorDtor & 2) {
    AppendFunctionToLLVMGlobalCtorsDtors(func, fdecl->priority, false);
  } */

  // name parameters
//  auto iarg = func.args_begin();

  return nullptr;
}

mlir::FuncOp MLIRFunction::getMLIRFunction() {
  assert(function && "MLIR FuncOp has to be generated");

  return function;
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
