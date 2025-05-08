#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Transform/Affine/Passes.h"
#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp"
#include "mlir/Transforms/Passes.h"

void polyToLLVMPipelineBuilder(mlir::OpPassManager& manager)
{
    // === Poly 方言降低到 Standard 方言 ===
    // 1. 将 Poly 方言操作转换为 Standard 方言 (arith, tensor, scf)
    manager.addPass(mlir::tutorial::poly::createPolyToStandard());
    // 2. 运行通用规范化，清理 IR
    manager.addPass(mlir::createCanonicalizerPass());

    // === Standard 方言降低到 Linalg 方言 ===
    // 3. 将逐元素操作转换为 linalg.generic
    manager.addPass(mlir::createConvertElementwiseToLinalgPass());
    // 4. 将其他 tensor 操作转换为 Linalg
    manager.addPass(mlir::createConvertTensorToLinalgPass());

    // === Bufferization: Tensor -> MemRef ===
    // 5. 执行一次性 Bufferization，将 tensor 转换为 memref
    // ref: https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
    mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;  // 函数边界也进行转换
    manager.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
    // 6. 添加 Buffer 释放相关的 Pass，管理 memref 生命周期
    mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
    mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                         deallocationOptions);

    // === Linalg 降低到显式循环 (SCF) ===
    // 7. 将 Linalg 操作转换为 SCF 循环 (scf.for)
    manager.addPass(mlir::createConvertLinalgToLoopsPass());

    // === MemRef 相关处理 ===
    // 8. 展开 memref.subview 操作，为降低到 LLVM 做准备
    manager.addPass(mlir::memref::createExpandStridedMetadataPass());

    // === 控制流和算术运算降低到 LLVM ===
    // 9. (可选，通常 ControlFlowToLLVM 已包含 SCF) 将 SCF 转换为 CF (Control
    // Flow) manager.addPass(mlir::createConvertSCFToCFPass());
    // 10. 将 SCF/CF 控制流转换为 LLVM 的基本块和分支
    manager.addPass(mlir::createConvertControlFlowToLLVMPass());
    // 11. 将 Arith 方言操作转换为 LLVM 指令
    manager.addPass(mlir::createArithToLLVMConversionPass());
    // 12. 将 Func 方言操作 (函数定义、调用、返回) 转换为 LLVM
    manager.addPass(mlir::createConvertFuncToLLVMPass());
    // 13. 完成 MemRef 到 LLVM 指针和内存操作的最终转换
    manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    // 14. 清理转换过程中残留的 unrealized_conversion_cast
    manager.addPass(mlir::createReconcileUnrealizedCastsPass());

    // === 清理和优化 ===
    // 15. 再次运行通用规范化
    manager.addPass(mlir::createCanonicalizerPass());
    // 16. 稀疏条件常量传播
    manager.addPass(mlir::createSCCPPass());
    // 17. 公共子表达式消除
    manager.addPass(mlir::createCSEPass());
    // 18. 符号级死代码消除 (移除未使用的函数等)
    manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char** argv)
{
    mlir::DialectRegistry registry;
    registry.insert<mlir::tutorial::poly::PolyDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    mlir::tutorial::registerAffinePasses();
    mlir::tutorial::poly::registerPolyToStandardPasses();

    mlir::PassPipelineRegistration<>(
        "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
        polyToLLVMPipelineBuilder);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}