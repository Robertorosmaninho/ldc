# - Try to find MLIR project at LLVM
#
# The following are set after configuration is done:
#   MLIR_FOUND   ON if MLIR installation was found
#   MLIR_ROOT_DIR
#   MLIR_INCLUDE_DIR
#   MLIR_INCLUDE_BUILD_DIR
#   MLIR_LIB_DIR
#   MLIR_LIBRARIES
#   MLIR_TABLEGEN  The mlir-tblgen executable

set(MLIR_FOUND OFF)

# We only want to find an MLIR version that is compatible with our LLVM version,
# so for now only look in the same installation dir as LLVM.
find_program(MLIR_TABLEGEN
        NAMES mlir-tblgen
        PATHS ${MLIR_ROOT_DIR}/bin ${LLVM_ROOT_DIR}/bin NO_DEFAULT_PATH
        DOC "Path to mlir-tblgen tool.")

if (NOT MLIR_TABLEGEN)
    message(STATUS "Could not find mlir-tblgen. Try manually setting MLIR_ROOT_DIR or MLIR_TABLEGEN.")
else ()
    set(MLIR_FOUND ON)
    message(STATUS "Found mlir-tblgen: ${MLIR_TABLEGEN}")
    get_filename_component(MLIR_BIN_DIR ${MLIR_TABLEGEN} DIRECTORY CACHE)
    get_filename_component(MLIR_ROOT_DIR "${MLIR_BIN_DIR}/.." ABSOLUTE CACHE)
    set(MLIR_INCLUDE_BUILD_DIR ${MLIR_ROOT_DIR}/tools/mlir/include)
    set(MLIR_LIB_DIR ${MLIR_ROOT_DIR}/lib)
    set(MLIR_INCLUDE_DIR ${MLIR_ROOT_DIR}/../mlir/include)

    # To be done: add the required MLIR libraries. Hopefully we don't have to manually list all MLIR libs.
    if (EXISTS "${MLIR_LIB_DIR}/MLIRIR.lib")
        set(MLIR_LIBRARIES
                ${MLIR_LIB_DIR}/libMLIRAffineOps.lib
                ${MLIR_LIB_DIR}/libMLIRAffineToStandard.lib
                ${MLIR_LIB_DIR}/libMLIRAnalysis.lib
                ${MLIR_LIB_DIR}/libMLIRDialect.lib
                ${MLIR_LIB_DIR}/libMLIRExecutionEngine.lib
                ${MLIR_LIB_DIR}/libMLIRIR.lib
                ${MLIR_LIB_DIR}/libMLIRLLVMIR.lib
                ${MLIR_LIB_DIR}/libMLIRLoopOps.lib
                ${MLIR_LIB_DIR}/libMLIRPass.lib
                ${MLIR_LIB_DIR}/libMLIRStandardOps.lib
                ${MLIR_LIB_DIR}/libMLIRStandardToLLVM.lib
                ${MLIR_LIB_DIR}/libMLIRSupport.lib
                ${MLIR_LIB_DIR}/libMLIRTargetLLVMIR.lib
                ${MLIR_LIB_DIR}/libMLIRTargetLLVMIRModuleTranslation.lib
                ${MLIR_LIB_DIR}/libMLIRTransformUtils.lib
                ${MLIR_LIB_DIR}/libMLIRTransforms.lib
                ${MLIR_LIB_DIR}/libMLIRTranslation.lib
                ${MLIR_LIB_DIR}/libLLVMExecutionEngine.lib
                ${MLIR_LIB_DIR}/libLLVMJITLink.lib
                ${MLIR_LIB_DIR}/libLLVMOrcJIT.lib
                ${MLIR_LIB_DIR}/libLLVMOrcError.lib
                ${MLIR_LIB_DIR}/libLLVMRuntimeDyld.lib
                )
    elseif (EXISTS "${MLIR_LIB_DIR}/libMLIRIR.a")
        set(MLIR_LIBRARIES
                ${MLIR_LIB_DIR}/libMLIRAffineOps.a
                ${MLIR_LIB_DIR}/libMLIRAffineToStandard.a
                ${MLIR_LIB_DIR}/libMLIRAnalysis.a
                ${MLIR_LIB_DIR}/libMLIRDialect.a
                ${MLIR_LIB_DIR}/libMLIRExecutionEngine.a
                ${MLIR_LIB_DIR}/libMLIRIR.a
                ${MLIR_LIB_DIR}/libMLIRLoopToStandard.a
                ${MLIR_LIB_DIR}/libMLIRLLVMIR.a
                ${MLIR_LIB_DIR}/libMLIRLoopOps.a
                ${MLIR_LIB_DIR}/libMLIRPass.a
                ${MLIR_LIB_DIR}/libMLIRStandardOps.a
                ${MLIR_LIB_DIR}/libMLIRStandardToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRSupport.a
                ${MLIR_LIB_DIR}/libMLIRTargetLLVMIR.a
                ${MLIR_LIB_DIR}/libMLIRTargetLLVMIRModuleTranslation.a
                ${MLIR_LIB_DIR}/libMLIRTransformUtils.a
                ${MLIR_LIB_DIR}/libMLIRTransforms.a
                ${MLIR_LIB_DIR}/libMLIRTranslation.a
                ${MLIR_LIB_DIR}/libLLVMExecutionEngine.a
                ${MLIR_LIB_DIR}/libLLVMJITLink.a
                ${MLIR_LIB_DIR}/libLLVMOrcJIT.a
                ${MLIR_LIB_DIR}/libLLVMOrcError.a
                ${MLIR_LIB_DIR}/libLLVMRuntimeDyld.a
                )
    endif ()

    # XXX: This function is untested and will need adjustment.
    function(mlir_tablegen)
        cmake_parse_arguments(
                ARG
                "NAME"
                "TARGET;OUTS;FLAG;SRCS"
                ${ARGN}
        )

        MESSAGE(STATUS "Setting target for Ops_" ${ARG_TARGET})

        set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRCS}
                PARENT_SCOPE)
        #mlir-tblgen ops.td --gen-op-* -I*-o=ops.*.inc
        add_custom_command(
                OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS}
                COMMAND ${MLIR_TABLEGEN} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRCS} -I${MLIR_INCLUDE_DIR} -o=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS}
                ARGS ${ARG_FLAG}
        )
        add_custom_target(Ops_${ARG_TARGET} ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS})
    endfunction()
endif ()
