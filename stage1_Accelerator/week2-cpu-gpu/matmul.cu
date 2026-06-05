// matmul.cu — Bài tập 2.2
// 3 phiên bản matmul CUDA. Build qua CMake (CUDA toolkit cần sẵn).
//
// Chạy: ./matmul [version]   version = naive | tiled | cublas (mặc định: tất cả)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); \
    if (e != cudaSuccess) { printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(e)); exit(1); } } while (0)

// V1: Naive — mỗi thread tính 1 phần tử C[i,j]
__global__ void matmul_naive(float* C, const float* A, const float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// V2: Shared memory tiling
#define TILE 16
__global__ void matmul_tiled(float* C, const float* A, const float* B, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}

static double bench_kernel(void (*launch)(float*, const float*, const float*, int),
                           float* dC, const float* dA, const float* dB, int N) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    launch(dC, dA, dB, N);  // warmup
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    const int iters = 50;
    for (int i = 0; i < iters; i++) launch(dC, dA, dB, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    return ms / iters;
}

static void launch_naive(float* C, const float* A, const float* B, int N) {
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    matmul_naive<<<grid, block>>>(C, A, B, N);
}
static void launch_tiled(float* C, const float* A, const float* B, int N) {
    dim3 block(TILE, TILE), grid(N / TILE, N / TILE);
    matmul_tiled<<<grid, block>>>(C, A, B, N);
}

static cublasHandle_t cublas_handle;
static void launch_cublas(float* C, const float* A, const float* B, int N) {
    float alpha = 1.0f;
    float beta = 0.0f;
    // cuBLAS is column-major: C = A * B row-major layout is computed as B * A in column-major
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N);
}

int main(int argc, char** argv) {
    const int N = 2048;
    const char* which = argc > 1 ? argv[1] : "all";

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

    double gflop = 2.0 * N * N * N / 1e9;

    if (!strcmp(which, "naive") || !strcmp(which, "all")) {
        double ms = bench_kernel(launch_naive, dC, dA, dB, N);
        printf("V1 Naive : %.2f ms, %.0f GFLOPS\n", ms, gflop / (ms / 1e3));
    }
    if (!strcmp(which, "tiled") || !strcmp(which, "all")) {
        double ms = bench_kernel(launch_tiled, dC, dA, dB, N);
        printf("V2 Tiled : %.2f ms, %.0f GFLOPS\n", ms, gflop / (ms / 1e3));
    }
    if (!strcmp(which, "cublas") || !strcmp(which, "all")) {
        cublasCreate(&cublas_handle);
        double ms = bench_kernel(launch_cublas, dC, dA, dB, N);
        printf("V3 cuBLAS: %.2f ms, %.0f GFLOPS\n", ms, gflop / (ms / 1e3));
        cublasDestroy(cublas_handle);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
}
