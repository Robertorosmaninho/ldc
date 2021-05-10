//=- ValueToBlockArgument.cpp - Pass to transform values to block arguments -=//
//
//
//===----------------------------------------------------------------------===//

#include "gen/MLIR/Dialect.h"
#include "gen/MLIR/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "iostream"
#include <set>

using namespace mlir;
using denseMap = DenseMap<Block *, std::vector<Block*>>;

/// Algorithm to compute the Dominance Frontier an to place phi-functions
/// from Andrew Appel's book "Modern Compiler Implementation in Java":

/* computeDF[n] =
 *   S <- {}
 *   for each node y in succ[n]
 *     if idom(y) ≠ n
 *       S <- S u {y}
 *   for each child c of n in the dominator tree
 *     computeDF[c]
 *     for each element w of DF[c]
 *       if n does not dominate w, or if n = w
 *         S <- S u {w}
 *   DF[n] <- S
 * */

/* Place-phi-Function =
 *   for each node n
 *     for each variable a in A_orig[n]
 *       defsites[a] <- defsites[a] u {n}
 *   for each variable a
 *     W <- defsites[a]
 *     while W not empty
 *       remove some node n from W
 *       for each y in DF[n]
 *         if a ∉ A_pri[y]
 *           insert the statement a <- phi(a,a,...,a) at the top of block y,
 *             where the phi-functions has as many arguments as y has
 *             predecessors
 *           A_phi[Y] <- A_phi[Y] u {a}
 *           if a ∉ A_orig[y]
 *             W <- W u {y}
 * */

namespace {
/// This pass take a value x in block A an add it as a block argument on block B
/// where x is used and an operation may modify it's value.
struct ValueToBlockArgumentPass
    : public PassWrapper<ValueToBlockArgumentPass, OperationPass<FuncOp>> {
  // Entry point for the pass.
  void runOnOperation() override {
    FuncOp op = (FuncOp)getOperation();
    denseMap dominanceFrontier;
    Block* entryPoint = &op->getRegion(0).front();
    auto dom = dominatorTree(&op->getRegion(0));

    /*llvm::errs() << "Teste\n";
    for (auto node : dom) {
      llvm::errs() << decode(node.first) << "\n";
      for (auto c : node.second)
        llvm::errs() << decode(c) << " - ";
      llvm::errs() << "\n";
    }*/

    computeDominanceFrontier(entryPoint, dominanceFrontier, dom);
    placeArguments(dominanceFrontier, op);

    /*llvm::errs() << "\nTeste DF:\n";
    for (auto node : dominanceFrontier) {
      llvm::errs() << decode(node.first) << "\n";
      for (auto c : node.second)
        llvm::errs() << decode(c) << " - ";
      llvm::errs() << "\n";
    }*/
  }

  void DFS(Block* node, std::map<Block*, bool> &visited,
                           std::vector<Block*> &postOrder) {

    auto firstOp = node->front().getName().getStringRef().lower();
    auto lastOp = node->front().getName().getStringRef().lower();

    visited.insert(std::make_pair(node, true));
    for (auto succ : node->getSuccessors()) {
      if (visited.find(succ) == visited.end() || !visited.find(node)->second)
        DFS(succ, visited, postOrder);
    }
    postOrder.push_back(node);
  }

  std::string decode(Block *node) {
    StringRef back, front =  node->front().getName().getStringRef();
    if (node->getOperations().size() > 2)
      back = (++node->rbegin())->getName().getStringRef();
    else
      back = node->back().getName().getStringRef();

    if (front == "D.double" && back == "D.double")
      return "bb0";
    else if (front == "std.cmpf" && back == "std.cond_br")
      return "bb4";
    else if (front == "D.return" && back == "D.return")
      return "bb3";
    else if (front == "D.double" && back == "D.fsub")
      return "bb2";
    else if (front == "D.double" && back == "D.fadd")
      return "bb1";
    else
      return "error";
  }

  std::vector<Block*> reversePostOrder(Block* node) {

    std::map<Block *, bool> visited;
    std::vector<Block *> postOrder;
    std::vector<Block *> reverse;

    DFS(node, visited, postOrder);

    //llvm::errs() << "PostOrder: \n";
    for (int i = postOrder.size() - 1; i >= 0; i--) {
      reverse.push_back(postOrder[i]);
      //llvm::errs() << decode(postOrder[i]) << " -> ";
    }
    //llvm::errs() << "\n";
    return reverse;
  }

  denseMap dominatorTree(Region *region) {
    DenseMap<Block *, std::vector<Block*>> DOM;
    Block *entryPoint = &region->front();
    auto reversePO = reversePostOrder(entryPoint);

    auto iplistToVector = [](llvm::iplist<Block>* v) {
      std::vector<Block*> list;
      for (auto &block : *v)
        list.push_back(&block);
      return list;
    };

    for (Block &block : region->getBlocks())
      DOM.insert(std::make_pair(&block, iplistToVector(&region->getBlocks())));

    auto setIntersection = [](std::vector<Block*> set1,
                              std::vector<Block*> set2) {
      std::vector<Block*> set3;

      std::sort(set1.begin(), set1.end());
      std::sort(set2.begin(), set2.end());

      auto it1 = set1.begin();
      auto it2 = set2.begin();

      while (it1.operator*() == it2.operator*()) {
        set3.push_back(*it1);
        ++it1; ++it2;
      }

      return set3;
    };

    auto compareSets = [](std::vector<Block*> set1, std::vector<Block*> set2) {
      if (set1.size() != set2.size())
        return false;

      for (int i = 0; i < (int)set1.size(); i++)
        if (set1[i] != set2[i])
          return false;

      return true;
    };

    bool changed = true;
    while (changed) {
      changed = false;
      for (auto node : reversePO) {
        std::vector<Block *> newSet;
        //auto node_name = decode(node);
        if (auto singlePred = node->getSinglePredecessor()) {
          //auto singlePred_name = decode(singlePred);
          newSet = DOM[singlePred];
        } else if (!node->isEntryBlock()) {
          auto first = node->getPredecessors().begin().operator*();
          //auto first_name = decode(first);
          for (auto pred : node->getPredecessors()) {
            //auto pred_name = decode(pred);
            newSet = setIntersection(DOM[first], DOM[pred]);
          }
        }
        newSet.push_back(node);
        if (!compareSets(newSet, DOM[node])) {
          DOM[node] = newSet;
          changed = true;
        }
      }
    }

    return DOM;
  }

  std::set<Block*> findChildren(denseMap DOM, Block* target) {
    std::set<Block*> children;
   // auto target_name = decode(target);
    for (auto node : DOM) {
     // auto node_name = decode(node.first);
      if (node.first == target)
        continue;
      for (int i = 0; i < (int)node.second.size(); i++) {
       // auto c_name = decode(node.second[i]);
        if (node.second[i] == target) {
          //auto succ = (node.second[i+1]);
         // auto succ_name = decode(succ);
          children.emplace(node.second[i+1]);
        }
      }
    }
    return children;
  }

  Block* idom(DenseMap<Block *, std::vector<Block*>> dom,
              Block* node) {
      int size = dom[node].size();
      auto vec = dom[node];
      assert(dom[node][size-1] == node);
      if (dom[node].size() > 1)
        return dom[node][size-2];
      return node;
  }

  bool dominates (Block* a, Block* B, denseMap dom) {
    for (auto block : dom[B])
      if (a == block)
        return true;

    return false;
  }

  // Create the dominance frontier
  void computeDominanceFrontier(Block* node, denseMap &DF, denseMap dom) {
    std::vector<Block*> S;
    //llvm::errs() << decode(node)  << " - ";
    //auto node_name = decode(node);

    for (auto y : node->getSuccessors()) {
      //auto y_name = decode(y);
      if (idom(dom, y) != node)
        S.push_back(y);
    }
    for (auto c : findChildren(dom, node)) {
      if (c == node)
        continue;
      //auto c_name = decode(c);
      computeDominanceFrontier(c, DF, dom);
      for (auto w : DF[c]) {
        //auto w_name = decode(w);
        if (!dominates(node, w, dom) || node == w)
          S.push_back(w);
      }
    }

    DF[node] = S;
  }

  void placeArguments(denseMap DF, FuncOp function) {
    for (auto node : DF) {
      if (node.second.empty())
        continue;
      llvm::errs() << "Node: " << decode(node.first) << "\n";

      for (auto block : node.second) {
        for (auto &op : block->getOperations()) {
          llvm::errs() << "Op: ";
          op.dump();
          llvm::errs() << "Users of Op: ";
          for (auto users : op.getUsers()) {
            users->dump();
            if (users->getBlock() == node.first) {
              llvm::errs() << "===============================\n";
            }
          }

          for (auto &operand : op.getOpOperands()) {
            llvm::errs() << "Operand: ";
            operand.get().dump();
            llvm::errs() << "Users of Operands: ";
            for (auto user : operand.get().getUsers()) {
              user->dump();
              if (user->getBlock() == node.first) {
                int i = node.first->getNumArguments();
                node.first->addArgument(operand.get().getType());
                user->replaceUsesOfWith(operand.get(),
                                        node.first->getArgument(i));
                llvm::errs() << "===============================\n";
                node.first->dump();
                llvm::errs() << "===============================\n";
              }
            }
          }
        }
      }
      /*auto p =  (++node.first->getOperations().rbegin());
      p->dump();
      for (auto &a : p->getOpOperands()) {
        llvm::errs() << "Operand: ";
        a.get().dump();
        llvm::errs() << "\nUsers: \n";
        for (auto user : a.get().getUsers())
          user->dump();
      }*/
    }
  }
};
} // end anonymous namespace
namespace mlir {
void registerValueToBlockArgumentPass() {
  PassRegistration<ValueToBlockArgumentPass>("value-to-blockargument",
                                             "Populates block arguments with "
                                             "values to be used that "
                                             "was previous defined");
}
} // namespace mlir

std::unique_ptr<Pass> mlir::D::createValueToBlockArgumentPass() {
  return std::make_unique<ValueToBlockArgumentPass>();
}