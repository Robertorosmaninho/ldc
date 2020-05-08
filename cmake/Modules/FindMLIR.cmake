# - Try to find MLIR project at LLVM
#
# The following are set after configuration is done:
#   MLIR_FOUND   ON if MLIR installation was found
#   MLIR_ROOT_DIR
#   MLIR_INCLUDE_DIR
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

if(NOT MLIR_TABLEGEN)
    message(STATUS "Could not find mlir-tblgen. Try manually setting MLIR_ROOT_DIR or MLIR_TABLEGEN.")
else()
    set(MLIR_FOUND ON)
    message(STATUS "Found mlir-tblgen: ${MLIR_TABLEGEN}")
    get_filename_component(MLIR_BIN_DIR ${MLIR_TABLEGEN} DIRECTORY CACHE)
    get_filename_component(MLIR_ROOT_DIR "${MLIR_BIN_DIR}/.." ABSOLUTE CACHE)
    include_directories(${MLIR_ROOT_DIR}/../build/tools/mlir/include)
    set(MLIR_INCLUDE_DIR ${MLIR_ROOT_DIR}/include)
    set(MLIR_LIB_DIR     ${MLIR_ROOT_DIR}/lib)

    # To be done: add the required MLIR libraries. Hopefully we don't have to manually list all MLIR libs.
    set(MLIR_LIBRARIES
            libMLIRAVX512.a
            libMLIRAVX512ToLLVM.a
            libMLIRAffineEDSC.a
            libMLIRAffineOps.a
            libMLIRAffineToStandard.a
            libMLIRAffineTransforms.a
            libMLIRAffineTransformsTestPasses.a
            libMLIRAffineUtils.a
            libMLIRAnalysis.a
            libMLIRCallInterfaces.a
            libMLIRControlFlowInterfaces.a
            libMLIRDerivedAttributeOpInterface.a
            libMLIRDialect.a
            libMLIREDSC.a
            libMLIRExecutionEngine.a
            libMLIRFxpMathOps.a
            libMLIRGPU.a
            libMLIRGPUtoCUDATransforms.a
            libMLIRGPUtoNVVMTransforms.a
            libMLIRGPUtoROCDLTransforms.a
            libMLIRGPUtoSPIRVTransforms.a
            libMLIRGPUtoVulkanTransforms.a
            libMLIRIR.a
            libMLIRInferTypeOpInterface.a
            libMLIRJitRunner.a
            libMLIRLLVMAVX512.a
            libMLIRLLVMIR.a
            libMLIRLLVMIRTransforms.a
            libMLIRLinalg.a
            libMLIRLinalgAnalysis.a
            libMLIRLinalgEDSC.a
            libMLIRLinalgOps.a
            libMLIRLinalgToLLVM.a
            libMLIRLinalgToSPIRVTransforms.a
            libMLIRLinalgTransforms.a
            libMLIRLinalgUtils.a
            libMLIRLoopAnalysis.a
            libMLIRLoopLikeInterface.a
            libMLIRLoopOps.a
            libMLIRLoopOpsTransforms.a
            libMLIRLoopToStandard.a
            libMLIRLoopsToGPU.a
            libMLIRMlirOptLib.a
            libMLIRNVVMIR.a
            libMLIROpenMP.a
            libMLIROptLib.a
            libMLIROptMain.a
            libMLIRParser.a
            libMLIRPass.a
            libMLIRQuant.a
            libMLIRQuantOps.a
            libMLIRQuantizerFxpMathConfig.a
            libMLIRQuantizerSupport.a
            libMLIRQuantizerTransforms.a
            libMLIRROCDLIR.a
            libMLIRSDBM.a
            libMLIRSPIRV.a
            libMLIRSPIRVSerialization.a
            libMLIRSPIRVTestPasses.a
            libMLIRSPIRVTransforms.a
            libMLIRShape.a
            libMLIRSideEffects.a
            libMLIRStandardOps.a
            libMLIRStandardOpsTransforms.a
            libMLIRStandardToLLVM.a
            libMLIRStandardToSPIRVTransforms.a
            libMLIRStandardToStandard.a
            libMLIRSupport.a
            libMLIRTargetAVX512.a
            libMLIRTargetLLVMIR.a
            libMLIRTargetLLVMIRModuleTranslation.a
            libMLIRTargetNVVMIR.a
            libMLIRTargetROCDLIR.a
            libMLIRTestDialect.a
            libMLIRTestIR.a
            libMLIRTestPass.a
            libMLIRTestTransforms.a
            libMLIRTransformUtils.a
            libMLIRTransforms.a
            libMLIRTranslateClParser.a
            libMLIRTranslation.a
            libMLIRVector.a
            libMLIRVectorOps.a
            libMLIRVectorToLLVM.a
            libMLIRVectorToLoops.a
            libMLIRViewLikeInterface.a
            )

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
                COMMAND ${MLIR_TABLEGEN} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRCS} -I${MLIR_INCLUDE_DIR}  -o=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS}
                ARGS ${ARG_FLAG}
        )
        add_custom_target(Ops_${ARG_TARGET} ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS})
    endfunction()
endif()
