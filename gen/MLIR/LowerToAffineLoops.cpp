//===-- LowerToAffineLoops.cpp --------------------------------------------===//
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
// DToAffine RewritePatterns
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

  // Make sure to deallocate this alloc at the end of the block.
  // TODO: Analyze the impact of it in control flow
  // This is fine as toy functions have no control flow.
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
                  rewriter.create<AffineLoadOp>(loc, memRefOperands.front(), loopIvs);
              auto loadedRhs =
                  rewriter.create<AffineLoadOp>(loc, memRefOperands.back(),
                      loopIvs);

              // Create the binary operation performed on the loaded values.
              return rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
            });
      return matchSuccess();
    }
  };

  using AddOpLowering = BinaryOpLowering<D::AddOp, AddIOp>;
  using AddFOpLowering = BinaryOpLowering<D::AddFOp, AddFOp>;
  using SubOpLowering = BinaryOpLowering<D::SubOp, SubIOp>;
  using SubFOpLowering = BinaryOpLowering<D::SubFOp, SubFOp>;
  using MulOpLowering = BinaryOpLowering<D::MulOp, MulIOp>;
  using MulFOpLowering = BinaryOpLowering<D::MulFOp, MulFOp>;
  using DivSOpLowering = BinaryOpLowering<D::DivSOp, SignedDivIOp>;
  using DivUOpLowering = BinaryOpLowering<D::DivUOp, UnsignedDivIOp>;
  using DivFOpLowering = BinaryOpLowering<D::DivFOp, DivFOp>;
  using ModSOpLowering = BinaryOpLowering<D::ModSOp, SignedRemIOp>;
  using ModUOpLowering = BinaryOpLowering<D::ModUOp, UnsignedRemIOp>;
  using ModFOpLowering = BinaryOpLowering<D::ModFOp, RemFOp>;
  using AndOpLowering = BinaryOpLowering<D::AndOp, AndOp>;
  using OrOpLowering = BinaryOpLowering<D::OrOp, OrOp>;
  using XorUOpLowering = BinaryOpLowering<D::XorOp, XOrOp>;
}

//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Integer Array operations
//===----------------------------------------------------------------------===//

struct IntegerOpLowering : public OpRewritePattern<D::IntegerOp> {
  using OpRewritePattern<D::IntegerOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(D::IntegerOp op,
                                     PatternRewriter &rewriter) const final {
    Attribute value = op.value();
    Location loc = op.getLoc();
    IntegerAttr constantValue = value.cast<IntegerAttr>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    //auto tensorType = op.getType().cast<TensorType>();
    auto tensorType = mlir::RankedTensorType::get(1, op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    indices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    rewriter.create<AffineStoreOp>(loc, rewriter.create<ConstantOp>(loc,
                                                                    constantValue),
                                   alloc, llvm::makeArrayRef(indices));

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};


//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Integer Array operations
//===----------------------------------------------------------------------===//

struct FloatOpLowering : public OpRewritePattern<D::FloatOp> {
  using OpRewritePattern<D::FloatOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(D::FloatOp op,
                                     PatternRewriter &rewriter) const final {
    Attribute value = op.value();
    Location loc = op.getLoc();
    FloatAttr constantValue = value.cast<FloatAttr>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    //auto tensorType = op.getType().cast<TensorType>();
    auto tensorType = mlir::RankedTensorType::get(1, op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    indices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    rewriter.create<AffineStoreOp>(loc, rewriter.create<ConstantOp>(loc,
                                                                    constantValue),
                                   alloc, llvm::makeArrayRef(indices));

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};


//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Integer Array operations
//===----------------------------------------------------------------------===//

struct DoubleOpLowering : public OpRewritePattern<D::DoubleOp> {
  using OpRewritePattern<D::DoubleOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(D::DoubleOp op,
                                     PatternRewriter &rewriter) const final {
    Attribute value = op.value();
    Location loc = op.getLoc();
    FloatAttr constantValue = value.cast<FloatAttr>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    //auto tensorType = op.getType().cast<TensorType>();
    auto tensorType = mlir::RankedTensorType::get(1, op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    indices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    rewriter.create<AffineStoreOp>(loc, rewriter.create<ConstantOp>(loc,
                                                                    constantValue),
                                   alloc, llvm::makeArrayRef(indices));

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};


//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Integer Array operations
//===----------------------------------------------------------------------===//

struct FloatArrayOpLowering : public OpRewritePattern<D::FloatArrayOp> {
  using OpRewritePattern<D::FloatArrayOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(D::FloatArrayOp op,
                                     PatternRewriter &rewriter) const final {
    Attribute value = op.value();
    Location loc = op.getLoc();
    DenseElementsAttr constantValue = value.cast<DenseElementsAttr>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Integer Array operations
//===----------------------------------------------------------------------===//

struct DoubleArrayOpLowering : public OpRewritePattern<D::DoubleArrayOp> {
  using OpRewritePattern<D::DoubleArrayOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(D::DoubleArrayOp op,
                                     PatternRewriter &rewriter) const final {
    Attribute value = op.value();
    Location loc = op.getLoc();
    DenseElementsAttr constantValue = value.cast<DenseElementsAttr>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// DToAffine RewritePatterns: Integer Array operations
//===----------------------------------------------------------------------===//

struct IntegerArrayOpLowering : public OpRewritePattern<D::IntegerArrayOp> {
  using OpRewritePattern<D::IntegerArrayOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(D::IntegerArrayOp op,
                                     PatternRewriter &rewriter) const final {
    Attribute value = op.value();
    Location loc = op.getLoc();
    DenseElementsAttr constantValue = value.cast<DenseElementsAttr>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<IntegerAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};


//===----------------------------------------------------------------------===//
// DToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the D operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the D dialect.
namespace {
  struct DToAffineLoweringPass : public FunctionPass<DToAffineLoweringPass> {
    void runOnFunction() final;
  };
} // end anonymous namespace.

void DToAffineLoweringPass::runOnFunction() {

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to `Standard` dialect.
  target.addLegalDialect<mlir::AffineOpsDialect, mlir::StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. If we actually want
  // a partial lowering, we explicitly mark the operations that don't want
  // to lower as `legal`.

  target.addIllegalDialect<D::DDialect>();
  target.addLegalOp<D::CastOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  OwningRewritePatternList patterns;
  patterns.insert<IntegerOpLowering, IntegerArrayOpLowering,
  FloatOpLowering, FloatArrayOpLowering, DoubleOpLowering,
  DoubleArrayOpLowering, AddOpLowering, AddFOpLowering,
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
std::unique_ptr<Pass> mlir::D::createLowerToAffinePass() {
  return std::make_unique<DToAffineLoweringPass>();
}

#endif
