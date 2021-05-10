
set(var "")
mlir_tablegen(MLIR
        TARGET
        H
        OUTS
        Ops.h.inc
        FLAG
        -gen-op-decls
        SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/Ops.td)
mlir_tablegen(MLIR
        TARGET
        CPP
        OUTS
        Ops.cpp.inc
        FLAG
        -gen-op-defs
        SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/Ops.td)

mlir_tablegen(MLIR
        TARGET
        Dialect
        OUTS
        Dialect.h.inc
        FLAG
        -gen-dialect-decls
        SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/Ops.td)