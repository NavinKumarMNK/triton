// RUN: triton-opt %s -split-input-file -convert-triton-gpu-to-llvm 2>&1  | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 33792 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_0d1d2d(%arg0: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<128x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f8E5M2, 1> -> tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x1xi32, #blocked>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked> -> tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x128xi32, #blocked>
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf8E5M2, #blocked>
    %11 = triton_gpu.local_alloc %10 {allocation.offset = 0 : i32} : (tensor<128x128xf8E5M2, #blocked>) -> !tt.memdesc<128x128xf8E5M2, #shared>
    %12 = tt.splat %arg1 : !tt.ptr<f8E5M2, 1> -> tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>
    %13 = tt.addptr %12, %2 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x1xi32, #blocked>
    %14 = tt.broadcast %13 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked> -> tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked>
    %15 = tt.addptr %14, %8 : tensor<128x128x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x128xi32, #blocked>
    %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf8E5M2, #blocked>
    %17 = triton_gpu.local_alloc %16 {allocation.offset = 16384 : i32} : (tensor<128x128xf8E5M2, #blocked>) -> !tt.memdesc<128x128xf8E5M2, #shared1>
    %18 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<128x1x!tt.ptr<f32, 1>, #blocked>
    %19 = tt.addptr %18, %2 : tensor<128x1x!tt.ptr<f32, 1>, #blocked>, tensor<128x1xi32, #blocked>
    %20 = tt.broadcast %19 : tensor<128x1x!tt.ptr<f32, 1>, #blocked> -> tensor<128x128x!tt.ptr<f32, 1>, #blocked>
    %21 = tt.addptr %20, %8 : tensor<128x128x!tt.ptr<f32, 1>, #blocked>, tensor<128x128xi32, #blocked>
    %22 = tt.load %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32, #blocked>
    triton_nvidia_gpu.fence_async_shared {bCluster = false}
    %23 = tt.dot %11, %17, %cst {allowTF32 = true, maxNumImpreciseAcc = 64 : i32} : !tt.memdesc<128x128xf8E5M2, #shared> * !tt.memdesc<128x128xf8E5M2, #shared1> -> tensor<128x128xf32, #mma>
    // CHECK-COUNT-256: llvm.fadd
    %24 = triton_gpu.convert_layout %23 {allocation.offset = 0 : i32} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %25 = arith.addf %24, %22 : tensor<128x128xf32, #blocked>
    tt.store %21, %25 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf32, #blocked>
    tt.return
  }
}
