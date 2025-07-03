#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    // 矩阵维度 (4090 适合大尺寸)
    int M = 2048, N = 2048, K = 2048;
    size_t bytes_A = M * K * sizeof(__half);
    size_t bytes_B = K * N * sizeof(__half);
    size_t bytes_C = M * N * sizeof(__half);

    // 分配设备内存
    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    // 初始化 cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 设置累加器为 FP16
    cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;  // FP16 累加
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;           // 自动选择算法

    // 标量参数 (FP16)
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    // 预热运行
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, CUDA_R_16F, M,
        d_B, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_16F, M,
        computeType, algo
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // 性能测试
    const int trials = 200;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_16F, M,
            d_B, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, M,
            computeType, algo
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // 计算性能
    float time_ms = std::chrono::duration<float>(end - start).count() * 1000 / trials;
    double flops = 2.0 * M * N * K * 1e-12; // TFLOPs
    double tflops = flops / (time_ms / 1000);

    std::cout << "Time: " << time_ms << " ms" << std::endl;
    std::cout << "Perf: " << tflops << " TFLOPS" << std::endl;

    // 清理
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
