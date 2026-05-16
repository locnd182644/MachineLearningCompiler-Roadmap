// avx_matmul.cpp — Bài tập 2.1
// Matmul thủ công: naive vs AVX2. Build: g++ -O3 -mavx2 -mfma ...
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <vector>

void matmul_naive(float* C, const float* A, const float* B, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void matmul_avx(float* C, const float* A, const float* B, int N) {
    // Giả định N chia hết cho 8
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < N; k++) {
                __m256 a = _mm256_broadcast_ss(&A[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            _mm256_storeu_ps(&C[i * N + j], acc);
        }
    }
}

int main() {
    const int N = 512;
    std::vector<float> A(N * N), B(N * N), C(N * N);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    auto bench = [&](auto fn, const char* name) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gflops = 2.0 * N * N * N / sec / 1e9;
        std::cout << name << ": " << sec * 1000 << " ms, " << gflops << " GFLOPS\n";
    };

    bench([&] { matmul_naive(C.data(), A.data(), B.data(), N); }, "Naive");
    bench([&] { matmul_avx(C.data(), A.data(), B.data(), N); }, "AVX");
}
