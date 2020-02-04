//===-- LowerToStandardLoops.cpp ------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
// =============================================================================
//
// This file implements a partial lowering of D operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//===----------------------------------------------------------------------===//
#if LDC_MLIR_ENABLED

#include "gen/MLIR/Dialect.h"
#include "gen/MLIR/Passes.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns - Licensed under the Apache License, Version
// 2.0
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                    PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input a rewriter, an array of memRefOperands corresponding
/// to the operands of the input operation, and the set of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn = function_ref<Value(PatternRewriter &rewriter,
                                           ArrayRef<Value> memRefOperands,
                                           ArrayRef<Value> loopIvs)>;

static void lowerOpToLoops(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create an empty affine loop for each of the dimensions within the shape.
  SmallVector<Value, 4> loopIvs;
  for (auto dim : tensorType.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, /*lb=*/0, dim, /*step=*/1);
    loop.getBody()->clear();
    loopIvs.push_back(loop.getInductionVar());

    // Terminate the loop body and update the rewriter insertion point to the
    // beginning of the loop.
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // Generate a call to the processing function with the rewriter, the memref
  // operands, and the loop induction variables. This function will return the
  // value to store at the current index.
  Value valueToStore = processIteration(rewriter, operands, loopIvs);
  rewriter.create<AffineStoreOp>(loc, valueToStore, alloc,
                                 llvm::makeArrayRef(loopIvs));

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

  template <typename BinaryOp, typename LoweredBinaryOp>
  struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx)
        : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    PatternMatchResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
      auto loc = op->getLoc();
      lowerOpToLoops(
          op, operands, rewriter,
          [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
                ArrayRef<Value> loopIvs) {
            // Generate an adaptor for the remapped operands of the BinaryOp. This
            // allows for using the nice named accessors that are generated by the
            // ODS.
            typename BinaryOp::OperandAdaptor binaryAdaptor(memRefOperands);

            // Generate loads for the element of 'lhs' and 'rhs' at the inner
            // loop.
            auto loadedLhs =
                rewriter.create<AffineLoadOp>(loc, binaryAdaptor.lhs(),
                                              loopIvs);
            auto loadedRhs =
                rewriter.create<AffineLoadOp>(loc, binaryAdaptor.rhs(),
                                              loopIvs);

            // Create the binary operation performed on the loaded values.
            return rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
          });
      return matchSuccess();
    }
  };

  using AddOpLowering = BinaryOpLowering<AddIOp, AddIOp>;
  using AddFOpLowering = BinaryOpLowering<AddFOp, AddFOp>;
  using SubOpLowering = BinaryOpLowering<SubIOp, SubIOp>;
  using SubFOpLowering = BinaryOpLowering<SubFOp, SubFOp>;
  using MulOpLowering = BinaryOpLowering<MulIOp, MulIOp>;
  using MulFOpLowering = BinaryOpLowering<MulFOp, MulFOp>;
  using DivSOpLowering = BinaryOpLowering<SignedDivIOp, SignedDivIOp>;
  using DivUOpLowering = BinaryOpLowering<UnsignedDivIOp, UnsignedDivIOp>;
  using DivFOpLowering = BinaryOpLowering<DivFOp, DivFOp>;
  using ModSOpLowering = BinaryOpLowering<SignedRemIOp, SignedRemIOp>;
  using ModUOpLowering = BinaryOpLowering<UnsignedRemIOp, UnsignedRemIOp>;
  using ModFOpLowering = BinaryOpLowering<RemFOp, RemFOp>;
  using AndOpLowering = BinaryOpLowering<AndOp, AndOp>;
  using OrOpLowering = BinaryOpLowering<OrOp, OrOp>;
  using XorUOpLowering = BinaryOpLowering<XOrOp, XOrOp>;
}

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the D operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the D dialect.
namespace {
  struct DToStandardLoweringPass : public FunctionPass<DToStandardLoweringPass> {
    void runOnFunction() final;
  };
} // end anonymous namespace.

void DToStandardLoweringPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() != "_Dmain")
    return;

  // Verify that the given main has no inputs and results.
  if (function.getNumArguments() || function.getType().getNumResults()) {
    function.emitError("expected 'main' to have 0 inputs and 0 results");
    return signalPassFailure();
  }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to `Standard` dialect.
  target.addLegalDialect<StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. If we actually want
  // a partial lowering, we explicitly mark the operations that don't want
  // to lower as `legal`.

 // target.addIllegalDialect<D::DDialect>();
  target.addLegalOp<D::CastOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, AddFOpLowering,
  SubOpLowering, SubFOpLowering, MulOpLowering, MulFOpLowering,
  DivSOpLowering, DivUOpLowering, DivFOpLowering, ModSOpLowering,
  ModUOpLowering, ModFOpLowering, AndOpLowering, OrOpLowering,
  XorUOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the D IR (e.g. matmul).
std::unique_ptr<Pass> mlir::D::createLowerToStandardPass() {
  return std::make_unique<DToStandardLoweringPass>();
}

#endif
