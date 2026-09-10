// =============================================================================
// Bài tập 8.2c — Dot Product chạy được qua mlir-cpu-runner
//
// Tính dot product: result = sum(A[i] * B[i]) cho i = 0..N-1
//
// Đây là bài tập tổng hợp: dùng func, arith, scf, memref
// và chạy thật qua mlir-cpu-runner để verify kết quả.
//
// Pipeline lower + run:
//   mlir-opt 03_dot_product.mlir \
//     --convert-scf-to-cf \
//     --convert-arith-to-llvm \
//     --finalize-memref-to-llvm \
//     --convert-cf-to-llvm \
//     --convert-func-to-llvm \
//     --reconcile-unrealized-casts \
//   | mlir-cpu-runner -e main -entry-point-result=void \
//     --shared-libs=libmlir_runner_utils.so,libmlir_c_runner_utils.so
//
// Expected output: 30.0 (= 1*1 + 2*2 + 3*3 + 4*4)
// =============================================================================

// Helper: print memref (từ runtime library)
func.func private @printMemrefF32(memref<*xf32>)

// --- Dot product kernel ---
// Nhận 2 memref 1D cùng kích thước, trả về scalar
func.func @dot_product(%A: memref<4xf32>, %B: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32

  // Loop tính sum of products
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %f0) -> (f32) {
    %a_i = memref.load %A[%i] : memref<4xf32>
    %b_i = memref.load %B[%i] : memref<4xf32>
    %prod = arith.mulf %a_i, %b_i : f32
    %new_acc = arith.addf %acc, %prod : f32
    scf.yield %new_acc : f32
  }

  return %result : f32
}

// --- Main: tạo data + gọi dot_product + in kết quả ---
func.func @main() {
  // Allocate vectors
  %A = memref.alloc() : memref<4xf32>
  %B = memref.alloc() : memref<4xf32>

  // Index constants
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  // Float constants
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32

  // Store A = [1, 2, 3, 4]
  memref.store %f1, %A[%c0] : memref<4xf32>
  memref.store %f2, %A[%c1] : memref<4xf32>
  memref.store %f3, %A[%c2] : memref<4xf32>
  memref.store %f4, %A[%c3] : memref<4xf32>

  // Store B = [1, 2, 3, 4]
  // dot(A, B) = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
  memref.store %f1, %B[%c0] : memref<4xf32>
  memref.store %f2, %B[%c1] : memref<4xf32>
  memref.store %f3, %B[%c2] : memref<4xf32>
  memref.store %f4, %B[%c3] : memref<4xf32>

  // Gọi dot product
  %result = func.call @dot_product(%A, %B) : (memref<4xf32>, memref<4xf32>) -> f32

  // Lưu kết quả vào memref để print
  %result_mem = memref.alloc() : memref<f32>
  memref.store %result, %result_mem[] : memref<f32>
  %result_unranked = memref.cast %result_mem : memref<f32> to memref<*xf32>
  func.call @printMemrefF32(%result_unranked) : (memref<*xf32>) -> ()

  // Cleanup
  memref.dealloc %A : memref<4xf32>
  memref.dealloc %B : memref<4xf32>
  memref.dealloc %result_mem : memref<f32>

  return
}
