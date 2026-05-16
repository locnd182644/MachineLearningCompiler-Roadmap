// matmul_tiled_3level.cu — Bài tập 6.1
// STUB: tiled matmul 3 cấp, CUTLASS-style. Cài đặt các phần TODO.
//
//   Block tile  (128x128) -> chia work cho thread block
//   Warp tile   (32x32)   -> chia trong block
//   Thread tile (8x8)     -> mỗi thread compute 1 sub-block
//
// Mục tiêu: ~70-80% cuBLAS. Profile bằng Nsight Compute.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(e)); exit(1); } } while (0)

#define BLOCK_TILE  128
#define WARP_TILE   32
#define THREAD_TILE 8

// TODO: kernel matmul 3 cấp tiling.
//   1. __shared__ float A_block[BLOCK_TILE][BLOCK_TILE] (+ B_block).
//   2. Vòng lặp trên K theo block tile:
//        - Cooperative load block tile A,B từ global -> shared.
//        - __syncthreads().
//        - Mỗi warp xử lý warp tile 32x32.
//        - Mỗi thread load sub-block 8x8 vào register, compute outer product,
//          tích lũy vào register accumulator.
//        - __syncthreads().
//   3. Ghi accumulator về global memory.
//   Gợi ý: cân nhắc double buffering shared memory để giấu latency load.
__global__ void matmul_tiled_3level(float* C, const float* A,
                                    const float* B, int N) {
    // TODO
}

int main() {
    const int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        hA[i] = (float)rand() / RAND_MAX;
        hB[i] = (float)rand() / RAND_MAX;
    }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // TODO: cấu hình grid/block, launch kernel, bench bằng cudaEvent,
    //       in GFLOPS, so với cuBLAS.

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    printf("Cài đặt kernel + launch trong matmul_tiled_3level.cu\n");
    return 0;
}
