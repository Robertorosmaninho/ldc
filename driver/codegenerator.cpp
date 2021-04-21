//===-- codegenerator.cpp -------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "driver/codegenerator.h"

#include "dmd/compiler.h"
#include "dmd/errors.h"
#include "dmd/globals.h"
#include "dmd/id.h"
#include "dmd/module.h"
#include "dmd/scope.h"
#include "driver/cl_options.h"
#include "driver/cl_options_instrumentation.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#if LDC_MLIR_ENABLED
#include "gen/MLIR/Dialect.h"
#include "gen/MLIR/MLIRGen.h"
#include "gen/MLIR/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Target/LLVMIR.h"
#endif
#include "gen/dynamiccompile.h"
#include "gen/logger.h"
#include "gen/modules.h"
#include "gen/runtime.h"
#if LDC_LLVM_VER >= 1100
#include "llvm/IR/LLVMRemarkStreamer.h"
#elif LDC_LLVM_VER >= 900
#include "llvm/IR/RemarkStreamer.h"
#endif
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#if LDC_MLIR_ENABLED
#if LDC_LLVM_VER >= 1200
#include "mlir/IR/BuiltinOps.h"
#else
#include "mlir/IR/Module.h"
#endif
#include "mlir/IR/MLIRContext.h"
#endif

#if LDC_LLVM_VER >= 1000
using std::make_unique;
#else
using llvm::make_unique;
#endif

namespace {

std::unique_ptr<llvm::ToolOutputFile>
createAndSetDiagnosticsOutputFile(IRState &irs, llvm::LLVMContext &ctx,
                                  llvm::StringRef filename) {
  std::unique_ptr<llvm::ToolOutputFile> diagnosticsOutputFile;

  // Set LLVM Diagnostics outputfile if requested
  if (opts::saveOptimizationRecord.getNumOccurrences() > 0) {
    llvm::SmallString<128> diagnosticsFilename;
    if (!opts::saveOptimizationRecord.empty()) {
      diagnosticsFilename = opts::saveOptimizationRecord.getValue();
    } else {
      diagnosticsFilename = filename;
      llvm::sys::path::replace_extension(diagnosticsFilename, "opt.yaml");
    }

    // If there is instrumentation data available, also output function hotness
    const bool withHotness = opts::isUsingPGOProfile();

#if LDC_LLVM_VER >= 900
    auto remarksFileOrError =
#if LDC_LLVM_VER >= 1100
        llvm::setupLLVMOptimizationRemarks(
#else
        llvm::setupOptimizationRemarks(
#endif

        ctx, diagnosticsFilename, "", "", withHotness);
    if (llvm::Error e = remarksFileOrError.takeError()) {
      irs.dmodule->error("Could not create file %s: %s",
                         diagnosticsFilename.c_str(),
                         llvm::toString(std::move(e)).c_str());
      fatal();
    }
    diagnosticsOutputFile = std::move(*remarksFileOrError);
#else
    std::error_code EC;
    diagnosticsOutputFile = make_unique<llvm::ToolOutputFile>(
        diagnosticsFilename, EC, llvm::sys::fs::F_None);
    if (EC) {
      irs.dmodule->error("Could not create file %s: %s",
                         diagnosticsFilename.c_str(), EC.message().c_str());
      fatal();
    }

    ctx.setDiagnosticsOutputFile(

        llvm::make_unique<llvm::yaml::Output>(diagnosticsOutputFile->os()));

    if (withHotness) {
      ctx.setDiagnosticsHotnessRequested(true);
    }
#endif // LDC_LLVM_VER < 900
  }

  return diagnosticsOutputFile;
}

void addLinkerMetadata(llvm::Module &M, const char *name,
                       llvm::ArrayRef<llvm::MDNode *> newOperands) {
  if (newOperands.empty())
    return;

  llvm::NamedMDNode *node = M.getOrInsertNamedMetadata(name);

  // Add the new operands in front of the existing ones, such that linker
  // options of .bc files passed on the cmdline are put _after_ the compiled .d
  // file.

  // Temporarily store metadata nodes that are already present
  llvm::SmallVector<llvm::MDNode *, 5> oldMDNodes;
  for (auto *MD : node->operands())
    oldMDNodes.push_back(MD);

  // Clear the list and add the new metadata nodes.
  node->clearOperands();
  for (auto *MD : newOperands)
    node->addOperand(MD);

  // Re-add metadata nodes that were already present
  for (auto *MD : oldMDNodes)
    node->addOperand(MD);
}

/// Add the "llvm.{linker.options,dependent-libraries}" metadata.
/// If the metadata is already present, merge it with the new data.
void emitLinkerOptions(IRState &irs) {
  llvm::Module &M = irs.module;
  addLinkerMetadata(M, "llvm.linker.options", irs.linkerOptions);
  addLinkerMetadata(M, "llvm.dependent-libraries", irs.linkerDependentLibs);
}

void emitLLVMUsedArray(IRState &irs) {
  if (irs.usedArray.empty()) {
    return;
  }

  auto *i8PtrType = llvm::Type::getInt8PtrTy(irs.context());

  // Convert all elements to i8* (the expected type for llvm.used)
  for (auto &elem : irs.usedArray) {
    elem = llvm::ConstantExpr::getBitCast(elem, i8PtrType);
  }

  auto *arrayType = llvm::ArrayType::get(i8PtrType, irs.usedArray.size());
  auto *llvmUsed = new llvm::GlobalVariable(
      irs.module, arrayType, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(arrayType, irs.usedArray), "llvm.used");
  llvmUsed->setSection("llvm.metadata");
}

void inlineAsmDiagnosticHandler(const llvm::SMDiagnostic &d, void *context,
                                unsigned locCookie) {
  if (d.getKind() == llvm::SourceMgr::DK_Error)
    ++global.errors;

  if (!locCookie) {
    d.print(nullptr, llvm::errs());
    return;
  }

  // replace the `<inline asm>` dummy filename by the LOC of the actual D
  // expression/statement (`myfile.d(123)`)
  const Loc &loc =
      static_cast<IRState *>(context)->getInlineAsmSrcLoc(locCookie);
  const char *filename = loc.toChars(/*showColumns*/ false);

  // keep on using llvm::SMDiagnostic::print() for nice, colorful output
  llvm::SMDiagnostic d2(*d.getSourceMgr(), d.getLoc(), filename, d.getLineNo(),
                        d.getColumnNo(), d.getKind(), d.getMessage(),
                        d.getLineContents(), d.getRanges(), d.getFixIts());
  d2.print(nullptr, llvm::errs());
}

} // anonymous namespace

namespace ldc {
CodeGenerator::CodeGenerator(llvm::LLVMContext &context,
#if LDC_MLIR_ENABLED
                             mlir::MLIRContext &mlirContext,
#endif
                             bool singleObj)
    : context_(context),
#if LDC_MLIR_ENABLED
      mlirContext_(mlirContext),
#endif
      moduleCount_(0), singleObj_(singleObj), ir_(nullptr) {
  // Set the context to discard value names when not generating textual IR.
  if (!global.params.output_ll) {
    context_.setDiscardValueNames(true);
  }
}

CodeGenerator::~CodeGenerator() {
  if (singleObj_ && moduleCount_ > 0) {
    // For singleObj builds, the first object file name is the one for the first
    // source file (e.g., `b.o` for `ldc2 a.o b.d c.d`).
    const char *filename = global.params.objfiles[0];

    // If there are bitcode files passed on the cmdline, add them after all
    // other source files have been added to the (singleobj) module.
    insertBitcodeFiles(ir_->module, ir_->context(), global.params.bitcodeFiles);

    writeAndFreeLLModule(filename);
  }
}

void CodeGenerator::prepareLLModule(Module *m) {
  ++moduleCount_;

  if (singleObj_ && ir_) {
    return;
  }

  assert(!ir_);

  // See http://llvm.org/bugs/show_bug.cgi?id=11479 – just use the source file
  // name, as it should not collide with a symbol name used somewhere in the
  // module.
  ir_ = new IRState(m->srcfile.toChars(), context_);
  ir_->module.setTargetTriple(global.params.targetTriple->str());
  ir_->module.setDataLayout(*gDataLayout);

  // TODO: Make ldc::DIBuilder per-Module to be able to emit several CUs for
  // single-object compilations?
  ir_->DBuilder.EmitCompileUnit(m);

  IrDsymbol::resetAll();
}

void CodeGenerator::finishLLModule(Module *m) {
  if (singleObj_) {
    return;
  }

  // Add bitcode files passed on the cmdline to
  // the first module only, to avoid duplications.
  if (moduleCount_ == 1) {
    insertBitcodeFiles(ir_->module, ir_->context(), global.params.bitcodeFiles);
  }
  writeAndFreeLLModule(m->objfile.toChars());
}

void CodeGenerator::writeAndFreeLLModule(const char *filename) {
  ir_->objc.finalize();

  ir_->DBuilder.Finalize();
  generateBitcodeForDynamicCompile(ir_);

  emitLLVMUsedArray(*ir_);
  emitLinkerOptions(*ir_);

  // Issue #1829: make sure all replaced global variables are replaced
  // everywhere.
  ir_->replaceGlobals();

  // Emit ldc version as llvm.ident metadata.
  llvm::NamedMDNode *IdentMetadata =
      ir_->module.getOrInsertNamedMetadata("llvm.ident");
  std::string Version("ldc version ");
  Version.append(global.ldc_version.ptr, global.ldc_version.length);
  llvm::Metadata *IdentNode[] = {llvm::MDString::get(ir_->context(), Version)};
  IdentMetadata->addOperand(llvm::MDNode::get(ir_->context(), IdentNode));

  //context_.setInlineAsmDiagnosticHandler(inlineAsmDiagnosticHandler, ir_);

  std::unique_ptr<llvm::ToolOutputFile> diagnosticsOutputFile =
      createAndSetDiagnosticsOutputFile(*ir_, context_, filename);

  writeModule(&ir_->module, filename);

  if (diagnosticsOutputFile)
    diagnosticsOutputFile->keep();

  delete ir_;
  ir_ = nullptr;
}

void CodeGenerator::emit(Module *m) {
  bool const loggerWasEnabled = Logger::enabled();
  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::enable();
  }

  IF_LOG Logger::println("CodeGenerator::emit(%s)", m->toPrettyChars());
  LOG_SCOPE;

  if (global.params.verbose_cg) {
    printf("codegen: %s (%s)\n", m->toPrettyChars(), m->srcfile.toChars());
  }

  if (global.errors) {
    Logger::println("Aborting because of errors");
    fatal();
  }

  prepareLLModule(m);

  codegenModule(ir_, m);

  finishLLModule(m);

  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::disable();
  }
}

#if LDC_MLIR_ENABLED
void CodeGenerator::emitMLIR(Module *m) {
  bool const loggerWasEnabled = Logger::enabled();
  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::enable();
  }

  IF_LOG Logger::println("CodeGenerator::emitMLIR(%s)", m->toPrettyChars());
  LOG_SCOPE;

  if (global.params.verbose_cg) {
    printf("codegen: %s (%s)\n", m->toPrettyChars(), m->srcfile.toChars());
  }

  if (global.errors) {
    Logger::println("Aborting because of errors");
    fatal();
  }

  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::OwningModuleRef module = ldc_mlir::mlirGen(mlirContext_, m);
  if (!module) {
    const auto llpath =
        replaceExtensionWith(global.mlir_ext, m->objfile.toChars());
    IF_LOG Logger::println("Error generating MLIR:'%s'", llpath.c_str());
    fatal();
  }

  writeMLIRModule(&module, m->objfile.toChars());

  if (m->llvmForceLogging && !loggerWasEnabled) {
    Logger::disable();
  }
}

int runJit(mlir::ModuleOp module) {
// Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0 ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

void emitLLVMIR(mlir::ModuleOp module, const char *filename) {
  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule =  mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    fatal();
  }

  bool enableOpt = global.params.enableOpt;

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    fatal();
  }

  // llvm::errs() << *llvmModule << "\n";

  if (llvmModule) {
    llvm::SmallString<128> buffer(filename);
    llvm::sys::path::replace_extension(
        buffer, llvm::StringRef(global.bc_ext.ptr, global.bc_ext.length));
    std::error_code errinfo;
    const auto bcpath = std::string(buffer.data(), buffer.size());
    llvm::raw_fd_ostream baba(bcpath, errinfo, llvm::sys::fs::F_None);
    const auto &M = *llvmModule;
    //llvm::WriteBitcodeToFile(M, baba);
    llvm::errs() << *llvmModule << "\n";
  }
}

void CodeGenerator::writeMLIRModule(mlir::OwningModuleRef *module,
                                    const char *filename) {
  // Write MLIR
  if (global.params.output_mlir) {
    const auto mlirpath = replaceExtensionWith(global.mlir_ext, filename);
    Logger::println("Writting MLIR to %s\n", mlirpath.c_str());
    std::error_code errinfo;
    llvm::raw_fd_ostream aos(mlirpath, errinfo, llvm::sys::fs::F_None);

    if (aos.has_error()) {
      error(Loc(), "Cannot write MLIR file '%s': %s", mlirpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }

    mlir::PassManager pm(&mlirContext_);
    //   pm.addPass(mlir::createInlinerPass());

    // Apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);

    bool isLoweringToAffine = global.params.affineDialect;
    bool isLoweringToLLVM = global.params.llvmDialect;
    bool printLLVMIR = global.params.llvmIr;
    bool enableJIT = global.params.runJIT;
    bool enableOpt = global.params.enableOpt;

    if (enableOpt || isLoweringToAffine) {
      // Inline all functions into main and then delete them.
      pm.addPass(mlir::createInlinerPass());

      mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      // Add optimizations if enabled.
      if (enableOpt) {
        optPM.addPass(mlir::createLoopFusionPass());
        optPM.addPass(mlir::createMemRefDataFlowOptPass());
      }
    }

    if (isLoweringToAffine) {
      // Finish lowering the toy IR to the LLVM dialect.
      pm.addPass(mlir::D::createLowerToAffinePass());

      mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      // Add optimizations if enabled.
      if (enableOpt) {
        optPM.addPass(mlir::createLoopFusionPass());
        optPM.addPass(mlir::createMemRefDataFlowOptPass());
      }
    }

    if (isLoweringToLLVM) {
      // Finish lowering the toy IR to the LLVM dialect.
      pm.addPass(mlir::D::createLowerToLLVMPass());
    }

    if (mlir::failed(pm.run(module->get()))) {
      IF_LOG Logger::println("Failed on running passes!");
      return;
    }

    if (!module->get()) {
      IF_LOG Logger::println("Cannot write MLIR file to '%s'",
                             mlirpath.c_str());
      fatal();
    }

    module->get().print(aos);

    if (printLLVMIR) {
      emitLLVMIR(module->get(), filename);
    }

    if (enableJIT) {
      assert(isLoweringToLLVM);
      runJit(module->get());
    }
  }
}

#endif
}
