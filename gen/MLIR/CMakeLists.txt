set(var "")
mlir_tablegen(MLIR
        TARGET
            H
        OUTS
            Ops.h.inc
        FLAG
            -gen-op-decls
        SRCS
            Ops.td)
mlir_tablegen(MLIR
        TARGET
            CPP
        OUTS
            Ops.cpp.inc
        FLAG
            -gen-op-defs
        SRCS
            Ops.td)