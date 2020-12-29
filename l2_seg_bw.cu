// code from "Dissecting the NVIDIA Volta GPU Architecture via
// Microbenchmarking"
#include <fstream>
#include <iostream>
#include <stdint.h>
#define THREADS_NUM 256
#define WARP_SIZE 32

__global__ void l2_bw(uint32_t *startClk, uint32_t *stopClk, float *dsink,
                      uint32_t *posArray, uint32_t *l2_size_input) {
  uint32_t l2_size = l2_size_input[0];
  uint32_t stride = THREADS_NUM;
  uint32_t iter_num = l2_size / stride;
  for (int iter = 0; iter < iter_num; ++iter) {
    uint32_t iter_offset = iter * stride;
    // thread index
    uint32_t tid = threadIdx.x;
    // a register to avoid compiler optimization
    float sink = 0;
    // populate l1 cache to warm up
    for (uint32_t i = tid; i < l2_size; i += THREADS_NUM) {
      float *ptr = (float *)posArray + i;
      asm volatile("{\t\n"
                   ".reg .f32 data;\n\t"
                   "ld.global.cg.f32 data, [%1];\n\t"
                   "add.f32 %0, data, %0;\n\t"
                   "}"
                   : "+f"(sink)
                   : "l"(ptr)
                   : "memory");
    }
    // synchronize all threads
    asm volatile("bar.sync 0;");
    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

    for (uint32_t i = 0; i < iter_num; i += THREADS_NUM) {
      float *ptr = (float *)posArray + iter_offset + i;

      uint32_t offset = tid;
      asm volatile("{\t\n"
                   ".reg .f32 data;\n\t"
                   "ld.global.cg.f32 data, [%1];\n\t"
                   "add.f32 %0, data, %0;\n\t"
                   "}"
                   : "+f"(sink)
                   : "l"(ptr + offset)
                   : "memory");
    }
    // synchronize all threads
    asm volatile("bar.sync 0;");
    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
    // write time and data back to memory
    startClk[iter_offset + tid] = start;
    stopClk[iter_offset + tid] = stop;
    dsink[tid] = sink;
  }
}

void TestL2BandWidth(uint32_t test_cache_size, std::ofstream &ofs);
void GetSharedSize() {
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, l2_bw);
  std::cout << "function preferredShmemCarveout " << attr.preferredShmemCarveout
            << "%" << std::endl;
}

int main() {
  std::ofstream ofs("l2_seg_bw.csv");
  ofs << "l2_offset,bw(GB/s)\n";
  TestL2BandWidth(8 * 1024 * 4, ofs);
}

void TestL2BandWidth(uint32_t test_cache_size, std::ofstream &ofs) {
  uint32_t l2_element_count = test_cache_size / sizeof(float);
  uint32_t seg_num = l2_element_count / THREADS_NUM;

  uint32_t *startClk_host = new uint32_t[seg_num * THREADS_NUM];
  uint32_t *endClk_host = new uint32_t[seg_num * THREADS_NUM];

  uint32_t *posArray_dev;

  cudaMalloc(&posArray_dev, test_cache_size);
  uint32_t *startClk_dev;
  uint32_t *endClk_dev;
  float *dsink_dev;
  uint32_t *l2_size_dev;
  cudaMalloc(&startClk_dev, seg_num * THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&endClk_dev, seg_num * THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&dsink_dev, THREADS_NUM * sizeof(float));
  cudaMalloc(&l2_size_dev, sizeof(uint32_t));

  cudaMemcpy(l2_size_dev, &l2_element_count, sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  l2_bw<<<1, THREADS_NUM>>>(startClk_dev, endClk_dev, dsink_dev, posArray_dev,
                            l2_size_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(startClk_host, startClk_dev,
             seg_num * THREADS_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(endClk_host, endClk_dev, seg_num * THREADS_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  for (int iter = 0; iter < seg_num; ++iter) {
    double cycle_count = 0;

    for (int i = 0; i < THREADS_NUM; ++i) {
      uint32_t single_thread_cycle = endClk_host[iter * THREADS_NUM + i] -
                                     startClk_host[iter * THREADS_NUM + i];
      cycle_count += single_thread_cycle;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double clock_rate = prop.clockRate * 1000;

    double load_time = 1;
    double total_transfer_byte = THREADS_NUM * sizeof(uint32_t);
    double t = cycle_count / THREADS_NUM / clock_rate;
    double gb = 1024 * 1024 * 1024;

    double bw_gb_s = total_transfer_byte / t / gb;

    std::cout << "offset " << iter * THREADS_NUM * sizeof(uint32_t) << " L2 BandWidth: " << bw_gb_s << "GByte/s" << std::endl;
    ofs << iter * THREADS_NUM * sizeof(uint32_t) << "," << bw_gb_s << "\n";
  }

  cudaFree(posArray_dev);
  cudaFree(startClk_dev);
  cudaFree(endClk_dev);
  cudaFree(dsink_dev);
  cudaFree(l2_size_dev);

  delete[] startClk_host;
  delete[] endClk_host;
}
