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
                ${MLIR_LIB_DIR}/libMLIRAffine.lib
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
               # ${MLIR_LIB_DIR}/libMLIRAffine.a
               # ${MLIR_LIB_DIR}/libMLIRAffineToStandard.a
               # ${MLIR_LIB_DIR}/libMLIRAnalysis.a
               # ${MLIR_LIB_DIR}/libMLIRDialect.a
               # ${MLIR_LIB_DIR}/libMLIRExecutionEngine.a
               # ${MLIR_LIB_DIR}/libMLIRIR.a
               # ${MLIR_LIB_DIR}/libMLIRLoopLikeInterface.a
               # ${MLIR_LIB_DIR}/libMLIRLLVMIR.a
               # #${MLIR_LIB_DIR}/libMLIRLoopOps.a
               # ${MLIR_LIB_DIR}/libMLIRPass.a
               # ${MLIR_LIB_DIR}/libMLIRStandard.a
               # ${MLIR_LIB_DIR}/libMLIRStandardToLLVM.a
               # ${MLIR_LIB_DIR}/libMLIRSupport.a
               # ${MLIR_LIB_DIR}/libMLIRTargetLLVMIRImport.a
               # ${MLIR_LIB_DIR}/libMLIRTargetLLVMIRExport.a
               # #${MLIR_LIB_DIR}/libMLIRTargetLLVMIRModuleTranslation.a
               # ${MLIR_LIB_DIR}/libMLIRTransformUtils.a
               # ${MLIR_LIB_DIR}/libMLIRTransforms.a
               # ${MLIR_LIB_DIR}/libMLIRTranslation.a
               # ${MLIR_LIB_DIR}/libLLVMExecutionEngine.a
               # ${MLIR_LIB_DIR}/libLLVMJITLink.a
               # ${MLIR_LIB_DIR}/libLLVMOrcJIT.a
               # ${MLIR_LIB_DIR}/libLLVMOrcShared.a
               # ${MLIR_LIB_DIR}/libLLVMOrcTargetProcess.a
               # ${MLIR_LIB_DIR}/libLLVMRuntimeDyld.a
                ${MLIR_LIB_DIR}/libDynamicLibraryLib.a
                ${MLIR_LIB_DIR}/libExampleIRTransforms.a
                ${MLIR_LIB_DIR}/libLLVMAggressiveInstCombine.a
                ${MLIR_LIB_DIR}/libLLVMAnalysis.a
                ${MLIR_LIB_DIR}/libLLVMAsmParser.a
                ${MLIR_LIB_DIR}/libLLVMAsmPrinter.a
                ${MLIR_LIB_DIR}/libLLVMBinaryFormat.a
                ${MLIR_LIB_DIR}/libLLVMBitReader.a
                ${MLIR_LIB_DIR}/libLLVMBitWriter.a
                ${MLIR_LIB_DIR}/libLLVMBitstreamReader.a
                ${MLIR_LIB_DIR}/libLLVMCFGuard.a
                ${MLIR_LIB_DIR}/libLLVMCFIVerify.a
                ${MLIR_LIB_DIR}/libLLVMCodeGen.a
                ${MLIR_LIB_DIR}/libLLVMCore.a
                ${MLIR_LIB_DIR}/libLLVMCoroutines.a
                ${MLIR_LIB_DIR}/libLLVMCoverage.a
                ${MLIR_LIB_DIR}/libLLVMDWARFLinker.a
                ${MLIR_LIB_DIR}/libLLVMDebugInfoCodeView.a
                ${MLIR_LIB_DIR}/libLLVMDebugInfoDWARF.a
                ${MLIR_LIB_DIR}/libLLVMDebugInfoGSYM.a
                ${MLIR_LIB_DIR}/libLLVMDebugInfoMSF.a
                ${MLIR_LIB_DIR}/libLLVMDebugInfoPDB.a
                ${MLIR_LIB_DIR}/libLLVMDemangle.a
                ${MLIR_LIB_DIR}/libLLVMDlltoolDriver.a
                ${MLIR_LIB_DIR}/libLLVMExecutionEngine.a
                ${MLIR_LIB_DIR}/libLLVMExegesis.a
                ${MLIR_LIB_DIR}/libLLVMExegesisX86.a
                ${MLIR_LIB_DIR}/libLLVMExtensions.a
                ${MLIR_LIB_DIR}/libLLVMFileCheck.a
                ${MLIR_LIB_DIR}/libLLVMFrontendOpenACC.a
                ${MLIR_LIB_DIR}/libLLVMFrontendOpenMP.a
                ${MLIR_LIB_DIR}/libLLVMFuzzMutate.a
                ${MLIR_LIB_DIR}/libLLVMGlobalISel.a
                ${MLIR_LIB_DIR}/libLLVMIRReader.a
                ${MLIR_LIB_DIR}/libLLVMInstCombine.a
                ${MLIR_LIB_DIR}/libLLVMInstrumentation.a
                ${MLIR_LIB_DIR}/libLLVMInterfaceStub.a
                ${MLIR_LIB_DIR}/libLLVMInterpreter.a
                ${MLIR_LIB_DIR}/libLLVMJITLink.a
                ${MLIR_LIB_DIR}/libLLVMLTO.a
                ${MLIR_LIB_DIR}/libLLVMLibDriver.a
                ${MLIR_LIB_DIR}/libLLVMLineEditor.a
                ${MLIR_LIB_DIR}/libLLVMLinker.a
                ${MLIR_LIB_DIR}/libLLVMMC.a
                ${MLIR_LIB_DIR}/libLLVMMCA.a
                ${MLIR_LIB_DIR}/libLLVMMCDisassembler.a
                ${MLIR_LIB_DIR}/libLLVMMCJIT.a
                ${MLIR_LIB_DIR}/libLLVMMCParser.a
                ${MLIR_LIB_DIR}/libLLVMMIRParser.a
                ${MLIR_LIB_DIR}/libLLVMObjCARCOpts.a
                ${MLIR_LIB_DIR}/libLLVMObject.a
                ${MLIR_LIB_DIR}/libLLVMObjectYAML.a
                ${MLIR_LIB_DIR}/libLLVMOption.a
                ${MLIR_LIB_DIR}/libLLVMOrcJIT.a
                ${MLIR_LIB_DIR}/libLLVMOrcShared.a
                ${MLIR_LIB_DIR}/libLLVMOrcTargetProcess.a
                ${MLIR_LIB_DIR}/libLLVMPasses.a
                ${MLIR_LIB_DIR}/libLLVMProfileData.a
                ${MLIR_LIB_DIR}/libLLVMRemarks.a
                ${MLIR_LIB_DIR}/libLLVMRuntimeDyld.a
                ${MLIR_LIB_DIR}/libLLVMScalarOpts.a
                ${MLIR_LIB_DIR}/libLLVMSelectionDAG.a
                ${MLIR_LIB_DIR}/libLLVMSupport.a
                ${MLIR_LIB_DIR}/libLLVMSymbolize.a
                ${MLIR_LIB_DIR}/libLLVMTableGen.a
                ${MLIR_LIB_DIR}/libLLVMTableGenGlobalISel.a
                ${MLIR_LIB_DIR}/libLLVMTarget.a
                ${MLIR_LIB_DIR}/libLLVMTestingSupport.a
                ${MLIR_LIB_DIR}/libLLVMTextAPI.a
                ${MLIR_LIB_DIR}/libLLVMTransformUtils.a
                ${MLIR_LIB_DIR}/libLLVMVectorize.a
                ${MLIR_LIB_DIR}/libLLVMWindowsManifest.a
                ${MLIR_LIB_DIR}/libLLVMX86AsmParser.a
                ${MLIR_LIB_DIR}/libLLVMX86CodeGen.a
                ${MLIR_LIB_DIR}/libLLVMX86Desc.a
                ${MLIR_LIB_DIR}/libLLVMX86Disassembler.a
                ${MLIR_LIB_DIR}/libLLVMX86Info.a
                ${MLIR_LIB_DIR}/libLLVMXRay.a
                ${MLIR_LIB_DIR}/libLLVMipo.a
                ${MLIR_LIB_DIR}/libLTO.dylib
                ${MLIR_LIB_DIR}/libMLIRAMX.a
                ${MLIR_LIB_DIR}/libMLIRAMXToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRAMXTransforms.a
                ${MLIR_LIB_DIR}/libMLIRAVX512.a
                ${MLIR_LIB_DIR}/libMLIRAVX512ToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRAVX512Transforms.a
                ${MLIR_LIB_DIR}/libMLIRAffine.a
                ${MLIR_LIB_DIR}/libMLIRAffineEDSC.a
                ${MLIR_LIB_DIR}/libMLIRAffineToStandard.a
                ${MLIR_LIB_DIR}/libMLIRAffineTransforms.a
                ${MLIR_LIB_DIR}/libMLIRAffineTransformsTestPasses.a
                ${MLIR_LIB_DIR}/libMLIRAffineUtils.a
                ${MLIR_LIB_DIR}/libMLIRAnalysis.a
                ${MLIR_LIB_DIR}/libMLIRArmNeon.a
                ${MLIR_LIB_DIR}/libMLIRArmNeonToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRArmSVE.a
                ${MLIR_LIB_DIR}/libMLIRArmSVEToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRAsync.a
                ${MLIR_LIB_DIR}/libMLIRAsyncToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRAsyncTransforms.a
                ${MLIR_LIB_DIR}/libMLIRCAPIIR.a
                ${MLIR_LIB_DIR}/libMLIRCallInterfaces.a
                ${MLIR_LIB_DIR}/libMLIRCastInterfaces.a
                ${MLIR_LIB_DIR}/libMLIRComplex.a
                ${MLIR_LIB_DIR}/libMLIRComplexToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRControlFlowInterfaces.a
                ${MLIR_LIB_DIR}/libMLIRCopyOpInterface.a
                ${MLIR_LIB_DIR}/libMLIRDLTI.a
                ${MLIR_LIB_DIR}/libMLIRDataLayoutInterfaces.a
                ${MLIR_LIB_DIR}/libMLIRDerivedAttributeOpInterface.a
                ${MLIR_LIB_DIR}/libMLIRDialect.a
                ${MLIR_LIB_DIR}/libMLIRDialectUtils.a
                ${MLIR_LIB_DIR}/libMLIREDSC.a
                ${MLIR_LIB_DIR}/libMLIRExecutionEngine.a
                ${MLIR_LIB_DIR}/libMLIRGPU.a
                ${MLIR_LIB_DIR}/libMLIRGPUToGPURuntimeTransforms.a
                ${MLIR_LIB_DIR}/libMLIRGPUToNVVMTransforms.a
                ${MLIR_LIB_DIR}/libMLIRGPUToROCDLTransforms.a
                ${MLIR_LIB_DIR}/libMLIRGPUToSPIRV.a
                ${MLIR_LIB_DIR}/libMLIRGPUToVulkanTransforms.a
                ${MLIR_LIB_DIR}/libMLIRIR.a
                ${MLIR_LIB_DIR}/libMLIRInferTypeOpInterface.a
                ${MLIR_LIB_DIR}/libMLIRJitRunner.a
                ${MLIR_LIB_DIR}/libMLIRLLVMArmSVE.a
                ${MLIR_LIB_DIR}/libMLIRLLVMArmSVEToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRLLVMIR.a
                ${MLIR_LIB_DIR}/libMLIRLLVMIRTransforms.a
                ${MLIR_LIB_DIR}/libMLIRLLVMToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRLinalg.a
                ${MLIR_LIB_DIR}/libMLIRLinalgAnalysis.a
                ${MLIR_LIB_DIR}/libMLIRLinalgEDSC.a
                ${MLIR_LIB_DIR}/libMLIRLinalgToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRLinalgToSPIRV.a
                ${MLIR_LIB_DIR}/libMLIRLinalgToStandard.a
                ${MLIR_LIB_DIR}/libMLIRLinalgTransforms.a
                ${MLIR_LIB_DIR}/libMLIRLinalgUtils.a
                ${MLIR_LIB_DIR}/libMLIRLoopAnalysis.a
                ${MLIR_LIB_DIR}/libMLIRLoopLikeInterface.a
                ${MLIR_LIB_DIR}/libMLIRMath.a
                ${MLIR_LIB_DIR}/libMLIRMathTransforms.a
                ${MLIR_LIB_DIR}/libMLIRMemRef.a
                ${MLIR_LIB_DIR}/libMLIRNVVMIR.a
                ${MLIR_LIB_DIR}/libMLIRNVVMToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIROpenACC.a
                ${MLIR_LIB_DIR}/libMLIROpenMP.a
                ${MLIR_LIB_DIR}/libMLIROpenMPToLLVM.a
                ${MLIR_LIB_DIR}/libMLIROpenMPToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIROptLib.a
                ${MLIR_LIB_DIR}/libMLIRPDL.a
                ${MLIR_LIB_DIR}/libMLIRPDLInterp.a
                ${MLIR_LIB_DIR}/libMLIRPDLToPDLInterp.a
                ${MLIR_LIB_DIR}/libMLIRParser.a
                ${MLIR_LIB_DIR}/libMLIRPass.a
                ${MLIR_LIB_DIR}/libMLIRPresburger.a
                ${MLIR_LIB_DIR}/libMLIRPublicAPI.dylib
                ${MLIR_LIB_DIR}/libMLIRQuant.a
                ${MLIR_LIB_DIR}/libMLIRROCDLIR.a
                ${MLIR_LIB_DIR}/libMLIRROCDLToLLVMIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRReduce.a
                ${MLIR_LIB_DIR}/libMLIRRewrite.a
                ${MLIR_LIB_DIR}/libMLIRSCF.a
                ${MLIR_LIB_DIR}/libMLIRSCFToGPU.a
                ${MLIR_LIB_DIR}/libMLIRSCFToOpenMP.a
                ${MLIR_LIB_DIR}/libMLIRSCFToSPIRV.a
                ${MLIR_LIB_DIR}/libMLIRSCFToStandard.a
                ${MLIR_LIB_DIR}/libMLIRSCFTransforms.a
                ${MLIR_LIB_DIR}/libMLIRSDBM.a
                ${MLIR_LIB_DIR}/libMLIRSPIRV.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVBinaryUtils.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVConversion.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVDeserialization.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVModuleCombiner.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVSerialization.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVTestPasses.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVTransforms.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVTranslateRegistration.a
                ${MLIR_LIB_DIR}/libMLIRSPIRVUtils.a
                ${MLIR_LIB_DIR}/libMLIRShape.a
                ${MLIR_LIB_DIR}/libMLIRShapeOpsTransforms.a
                ${MLIR_LIB_DIR}/libMLIRShapeTestPasses.a
                ${MLIR_LIB_DIR}/libMLIRShapeToStandard.a
                ${MLIR_LIB_DIR}/libMLIRSideEffectInterfaces.a
                ${MLIR_LIB_DIR}/libMLIRStandard.a
                ${MLIR_LIB_DIR}/libMLIRStandardOpsTransforms.a
                ${MLIR_LIB_DIR}/libMLIRStandardToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRStandardToSPIRV.a
                ${MLIR_LIB_DIR}/libMLIRSupport.a
                ${MLIR_LIB_DIR}/libMLIRSupportIndentedOstream.a
                ${MLIR_LIB_DIR}/libMLIRTableGen.a
                ${MLIR_LIB_DIR}/libMLIRTargetLLVMIRExport.a
                ${MLIR_LIB_DIR}/libMLIRTargetLLVMIRImport.a
                ${MLIR_LIB_DIR}/libMLIRTensor.a
                ${MLIR_LIB_DIR}/libMLIRTensorTransforms.a
                ${MLIR_LIB_DIR}/libMLIRTestAnalysis.a
                ${MLIR_LIB_DIR}/libMLIRTestDialect.a
                ${MLIR_LIB_DIR}/libMLIRTestIR.a
                ${MLIR_LIB_DIR}/libMLIRTestPass.a
                ${MLIR_LIB_DIR}/libMLIRTestReducer.a
                ${MLIR_LIB_DIR}/libMLIRTestRewrite.a
                ${MLIR_LIB_DIR}/libMLIRTestTransforms.a
                ${MLIR_LIB_DIR}/libMLIRToLLVMIRTranslationRegistration.a
                ${MLIR_LIB_DIR}/libMLIRTosa.a
                ${MLIR_LIB_DIR}/libMLIRTosaTestPasses.a
                ${MLIR_LIB_DIR}/libMLIRTosaToLinalg.a
                ${MLIR_LIB_DIR}/libMLIRTosaToSCF.a
                ${MLIR_LIB_DIR}/libMLIRTosaToStandard.a
                ${MLIR_LIB_DIR}/libMLIRTosaTransforms.a
                ${MLIR_LIB_DIR}/libMLIRTransformUtils.a
                ${MLIR_LIB_DIR}/libMLIRTransforms.a
                ${MLIR_LIB_DIR}/libMLIRTranslation.a
                ${MLIR_LIB_DIR}/libMLIRVector.a
                ${MLIR_LIB_DIR}/libMLIRVectorInterfaces.a
                ${MLIR_LIB_DIR}/libMLIRVectorToLLVM.a
                ${MLIR_LIB_DIR}/libMLIRVectorToROCDL.a
                ${MLIR_LIB_DIR}/libMLIRVectorToSCF.a
                ${MLIR_LIB_DIR}/libMLIRVectorToSPIRV.a
                ${MLIR_LIB_DIR}/libMLIRViewLikeInterface.a
                ${MLIR_LIB_DIR}/libgtest.a
                ${MLIR_LIB_DIR}/libgtest_main.a
                ${MLIR_LIB_DIR}/libmlir_async_runtime.dylib
                ${MLIR_LIB_DIR}/libmlir_c_runner_utils.dylib
                ${MLIR_LIB_DIR}/libmlir_runner_utils.dylib
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
                COMMAND ${MLIR_TABLEGEN} ${ARG_SRCS} -I${MLIR_INCLUDE_DIR} -o=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS}
                ARGS ${ARG_FLAG}
        )
        add_custom_target(Ops_${ARG_TARGET} ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_OUTS})
    endfunction()
endif ()
