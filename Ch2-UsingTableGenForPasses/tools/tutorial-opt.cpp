#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "mlir-tutorial/Transform/Affine/Passes.h"
#include "mlir/IR/DialectRegistry.h"

int main(int argc, char** argv)
{
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    // Register our pass
    mlir::tutorial::registerAffinePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}