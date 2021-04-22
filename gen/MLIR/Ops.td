//===- Ops.td - D dialect operation definitions ----------*- tablegen -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the D Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef D_OPS
#define D_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// Provide a definition of the 'D' dialect in the ODS framework so that we

// can define our operations.
def D_Dialect : Dialect {
  let name = "D";

  //let summary = "Summary"; //#TODO
  //let description = [{Description}]; //#TODO

  let cppNamespace = "::mlir::D";
}

// Base class for D dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class D_Op<string mnemonic, list<OpTrait> traits = []> :
     Op<D_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : D_Op<"add", [NoSideEffect, Commutative]> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two
        integer values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

// AddFOp FloatLike infers the size of float

def AddFOp : D_Op<"fadd", [NoSideEffect, Commutative]> {
    let summary = "element-wise addition operation";
    let description = [{
        The "addf" operation performs element-wise addition between two values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
    ];
}


// SubOp SignlessIntegerLike infers the size of int

def SubOp : D_Op<"sub", [NoSideEffect]> {
    let summary = "element-wise subtraction operation";
    let description = [{
        The "sub" operation performs element-wise subtraction between two
        integer values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

    // Allow building an SubOp with from the two input operands.
    let builders = [
      OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

// SubFOp FloatLike infers the size of float

def SubFOp : D_Op<"fsub"> {
    let summary = "element-wise subtraction operation";
    let description = [{
        The "fsub" operation performs element-wise subtraction between two
        float values.
        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

    // Allow building an SubFOp with from the two input operands.
    let builders = [
      OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def MulOp : D_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    operands. The type of operands are expected to match.
  }];

  let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
  let results = (outs SignlessIntegerLike);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def MulFOp : D_Op<"fmul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    operands. The type of operands are expected to match.
  }];

  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
  let results = (outs FloatLike);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def DivSOp : D_Op<"sdiv"> {
    let summary = "signed element-wise division operation";
    let description = [{
        The "sdiv" operation performs element-wise division between two signed
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a DivSOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def DivUOp : D_Op<"udiv"> {
    let summary = "unsigned element-wise division operation";
    let description = [{
        The "sdiv" operation performs element-wise division between two unsigned
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a DivUOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def DivFOp : D_Op<"fdiv"> {
    let summary = "element-wise division operation";
    let description = [{
        The "fdiv" operation performs element-wise division between two
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

   // Allow building a DivIOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def ModSOp : D_Op<"srem"> {
    let summary = "Returns the signed integer remainder of a division.";
    let description = [{
        The ‘srem‘ instruction returns the remainder from the signed division of
        its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a ModSOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def ModUOp : D_Op<"urem"> {
    let summary = "Returns the unsigned integer remainder of a division.";
    let description = [{
        The ‘urem‘ instruction returns the remainder from the un signed division
        of its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a ModUOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def ModFOp : D_Op<"frem"> {
    let summary = "Returns the remainder of a division.";
    let description = [{
        The ‘frem‘ instruction returns the remainder from the division of its
        two operands. The remainder has the same sign as the dividend.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    let results = (outs FloatLike);

   // Allow building a ModFOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def AndOp : D_Op<"and"> {
    let summary = "Logical 'and' between two operands";
    let description = [{
        The ‘and‘ instruction returns the bitwise logical and of its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a AndOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def OrOp : D_Op<"or"> {
    let summary = "Logical 'or' between two operands";
    let description = [{
        The ‘or‘ instruction returns the bitwise logical and of its two operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a OrOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

def XorOp : D_Op<"xor"> {
    let summary = "Logical 'xor' between two operands";
    let description = [{
        The ‘xor‘ instruction returns the bitwise logical and of its two
        operands.

        The type of the value operands are expected to match.
    }];

    let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike);

   // Allow building a XorOp with from the two input operands.
     let builders = [
       OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
     ];
}

////////////////////////////////////////////////////////////////////////////////
// Call Operation
////////////////////////////////////////////////////////////////////////////////
/*def CallOp : D_Op<"call"> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that was
        already declared on the scope. The callee name is attached
        as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
         %4 = "D.call"(%1, %3) {callee = @my_func}
               : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
        arguments.
    }];

    // The call operation takes a symbol reference attribute as the callee, and
    // inputs for the call.
    let arguments = (ins FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$inputs);

    // The generic call operation returns a single value of AnyType - The type
    // is inferred later.
    let results = (outs AnyType);

    // Add custom build methods for the call operation.
    let builders = [
        OpBuilderDAG<(ins "StringRef":$callee,
                          "ArrayRef<Type> types":$types,
                          "ArrayRef<Value>":$arguments)>
    ];
}
*/
////////////////////////////////////////////////////////////////////////////////
// Constant/Literal Declaration Ops
////////////////////////////////////////////////////////////////////////////////

def IntegerOp : D_Op<"int", [NoSideEffect]> {
  let summary = "Integer Declaration";
  let description = [{
    IntegerOp operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "D.int"() {value = 10 : i32} : () -> i32
    ```
  }];

  // The IntegerOp operation takes an attribute as the only input.
  let arguments = (ins AnyAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyType);

  let builders = [
    // Build a constant with a given constant integer value.
    OpBuilderDAG<(ins "Type":$type, "int":$value, "int":$size)>,

   // OpBuilderDAG<
   //     "Builder *builder, OperationState &result, Attribute value",
   //     [{ build(builder, result, value.getType(), value); }]>
  ];

  let hasFolder = 1;
}


def FloatOp : D_Op<"float", [NoSideEffect]> {
  let summary = "Float Declaration";
  let description = [{
    FloatOp operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "D.float"() {value = 10.00000 : f32} : () -> f32
      %1 = "D.float"() {value = 5.00000 : f16} : () -> f16
    ```
  }];

  // The FloatOp operation takes an attribute as the only input.
  let arguments = (ins AnyAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyType);

  // Add custom build methods for the FloatOp operation. These method
  // populates the `state` that MLIR uses to create operations, i.e. these are
  // used when using `builder.create<FloatOp>(...)`.
  let builders = [
    // Build a constant with a given constant float value.
    OpBuilderDAG<(ins "Type":$type, "float":$value)>

    // Build a float with a given value.
    // OpBuilderDAG<
    //      "Builder *builder, OperationState &result, Attribute value",
    //      [{ build(builder, result, value.getType(), value); }]>
  ];

   let hasFolder = 1;
}

def DoubleOp : D_Op<"double", [NoSideEffect]> {
  let summary = "Double Declaration";
  let description = [{
    DoubleOp operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "D.double"() {value = 10.00000 : f64} : () -> f64
    ```
  }];

  // The DoubleOp operation takes an attribute as the only input.
  let arguments = (ins AnyAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyType);

  // Add custom build methods for the DoubleOp operation. These method
  // populates the `state` that MLIR uses to create operations, i.e. these are
  // used when using `builder.create<DoubleOp>(...)`.
  let builders = [
    // Build a constant with a given constant double value.
    OpBuilderDAG<(ins "Type":$type, "double":$value)>

    // Build a double with a given value.
    //OpBuilderDAG<
    //        "Builder *builder, OperationState &result, Attribute value",
    //        [{ build(builder, result, value.getType(), value); }]>
  ];

   let hasFolder = 1;
}

////////////////////////////////////////////////////////////////////////////////
// Cast Operations
////////////////////////////////////////////////////////////////////////////////

def CastOp : D_Op<"cast"> {
  let summary = "cast type";
  let description = [{
    The "cast" operation converts a Value from one type to another type
    without changing any data elements. If both are ranked
    then the rank should be the same and static dimensions should match. The
    operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins AnyTypeOf<[AnyType, AnyMemRef]>:$input);
  let results = (outs AnyType:$output);

    let builders = [OpBuilderDAG<(ins "Value":$value, "Type":$destType), [{
     build($_builder, $_state, destType, value);
     }]>];

//  let builders = [OpBuilderDAG<
//    "Builder *builder, OperationState &result, Value value, Type destType", [{
//      result.addOperands(value);
//      result.addTypes(destType);
//  }]>];
}


#endif // D_OPS