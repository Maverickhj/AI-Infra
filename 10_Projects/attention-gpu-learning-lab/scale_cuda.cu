// Attention lab: apply S <- S / sqrt(d) to a dense float32 score matrix.
// Both kernels compute exactly the same output; only the thread-to-element
// mapping changes. No performance ratio is assumed.
//
// Build for your GPU's compute capability (e.g. sm_89 for RTX 4070 Ti SUPER):
//   nvcc -O3 -lineinfo -arch=sm_89 scale_cuda.cu -o scale_cuda
// Run: ./scale_cuda [N=4096] [iterations=50]
// Inspect machine instructions: cuobjdump --dump-sass ./scale_cuda
// Optional PTX: nvcc -O3 -arch=compute_89 -ptx scale_cuda.cu -o scale_cuda.ptx
// NVIDIA reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/
// Coalescing: https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
//
// Preconditions: NVIDIA CUDA GPU, working driver, CUDA Toolkit (nvcc).
// Timing excludes allocation, host/device copies, and correctness checks.
// Reported GB/s uses requested logical bytes; it is NOT measured DRAM traffic.
// Caches and transaction granularity can make physical traffic differ.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                               \
    if (error != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,       \
                   #call, cudaGetErrorString(error));                        \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// blockDim == (32, 8): each warp has a fixed threadIdx.y and x=0..31.
// Coalesced=true: adjacent lanes access adjacent columns of one row.
// Coalesced=false: adjacent lanes access different rows of one column.
// Each mapping visits every output coordinate exactly once, including tails.
template <bool Coalesced>
__global__ void attention_scale(const float* score, float* output, int n,
                                float scale) {
  const int row = Coalesced ? blockIdx.y * blockDim.y + threadIdx.y
                            : blockIdx.y * blockDim.x + threadIdx.x;
  const int col = Coalesced ? blockIdx.x * blockDim.x + threadIdx.x
                            : blockIdx.x * blockDim.y + threadIdx.y;
  if (row < n && col < n) {
    const size_t index = static_cast<size_t>(row) * n + col;
    output[index] = score[index] * scale;
  }
}

template <bool Coalesced>
float run_case(const float* device_score, float* device_output, int n,
               float scale, int iterations) {
  const dim3 block(32, 8);
  const dim3 grid(Coalesced ? (n + 31) / 32 : (n + 7) / 8,
                  Coalesced ? (n + 7) / 8 : (n + 31) / 32);
  for (int i = 0; i < 5; ++i) {
    attention_scale<Coalesced><<<grid, block>>>(device_score, device_output, n,
                                               scale);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    attention_scale<Coalesced><<<grid, block>>>(device_score, device_output, n,
                                               scale);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return milliseconds / iterations;
}

int parse_positive(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (text[0] == '\0' || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "%s must be a positive integer.\n", name);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

int main(int argc, char** argv) {
  if (argc > 3) {
    std::fprintf(stderr, "Usage: %s [N=4096] [iterations=50]\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int n = argc > 1 ? parse_positive(argv[1], "N") : 4096;
  const int iterations = argc > 2 ? parse_positive(argv[2], "iterations") : 50;
  // Keep this learning experiment bounded and its int launch arithmetic safe.
  if (n > 32768) {
    std::fprintf(stderr, "This small experiment supports N <= 32768.\n");
    return EXIT_FAILURE;
  }
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("GPU: %s; compute capability: %d.%d; SMs: %d\n",
              properties.name, properties.major, properties.minor,
              properties.multiProcessorCount);
  const size_t count = static_cast<size_t>(n) * n;
  const size_t bytes = count * sizeof(float);
  constexpr int head_dimension = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  std::printf("float32 score: [%d,%d]; scale=1/sqrt(%d); block=(32,8)\n",
              n, n, head_dimension);
  std::printf("Each matrix: %.2f MiB; warmup=5; iterations=%d\n",
              static_cast<double>(bytes) / (1024 * 1024), iterations);
  std::vector<float> input(count), output(count);
  for (size_t i = 0; i < count; ++i) {
    input[i] = (static_cast<int>(i % 251) - 125) * 0.001f;
  }
  float *device_score = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_score, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_score, input.data(), bytes,
                         cudaMemcpyHostToDevice));

  for (int mode = 0; mode < 2; ++mode) {
    const float ms = mode == 0
                         ? run_case<true>(device_score, device_output, n, scale,
                                          iterations)
                         : run_case<false>(device_score, device_output, n, scale,
                                           iterations);
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes,
                           cudaMemcpyDeviceToHost));
    float max_abs_error = 0;
    for (size_t i = 0; i < count; ++i) {
      const float expected = input[i] * scale;
      if (!std::isfinite(output[i])) {
        std::fprintf(stderr, "Nonfinite output at index %zu\n", i);
        return EXIT_FAILURE;
      }
      max_abs_error = std::max(max_abs_error, std::fabs(output[i] - expected));
    }
    const double requested_gbps = (2.0 * bytes) / (ms * 1.0e6);
    std::printf("%-22s %9.4f ms; requested traffic bandwidth %9.2f GB/s; "
                "max_abs_error %.3g\n",
                mode == 0 ? "lane -> adjacent col" : "lane -> strided row",
                ms, requested_gbps, max_abs_error);
    if (max_abs_error > 1.0e-6f) {
      std::fprintf(stderr, "Correctness check failed.\n");
      return EXIT_FAILURE;
    }
  }
  CUDA_CHECK(cudaFree(device_score));
  CUDA_CHECK(cudaFree(device_output));
  std::puts("Inspect MemoryWorkloadAnalysis in Nsight Compute to distinguish "
            "requested bytes, cache-sector traffic, and DRAM bytes.");
  return EXIT_SUCCESS;
}
