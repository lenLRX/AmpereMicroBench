// code from "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
#include <iostream>
#include <fstream>
#include <stdint.h>
#define THREADS_NUM 1024
#define WARP_SIZE 32
// const int L1_SIZE_BYTE = 192 * 1024;
//const int L1_SIZE_BYTE = 256 * 1024;
//const int L1_SIZE = L1_SIZE_BYTE / sizeof(double);

__global__ void l1_bw(uint32_t *startClk, uint32_t *stopClk, double *dsink,
                      uint32_t *posArray, uint32_t *l1_size_input) {
  uint32_t l1_size = l1_size_input[0];
  // thread index
  uint32_t tid = threadIdx.x;
  // a register to avoid compiler optimization
  double sink = 0;
  // populate l1 cache to warm up
  for (uint32_t i = tid; i < l1_size; i += THREADS_NUM) {
    double *ptr = (double *)posArray + i;
    asm volatile("{\t\n"
                 ".reg .f64 data;\n\t"
                 "ld.global.ca.f64 data, [%1];\n\t"
                 "add.f64 %0, data, %0;\n\t"
                 "}"
                 : "+d"(sink)
                 : "l"(ptr)
                 : "memory");
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  for (uint32_t i = 0; i < l1_size; i += THREADS_NUM) {
    double *ptr = (double *)posArray + i;
    // every warp loads all data in l1 cache
    for (uint32_t j = 0; j < THREADS_NUM; j += WARP_SIZE) {
      uint32_t offset = (tid + j) % THREADS_NUM;
      asm volatile("{\t\n"
                   ".reg .f64 data;\n\t"
                   "ld.global.ca.f64 data, [%1];\n\t"
                   "add.f64 %0, data, %0;\n\t"
                   "}"
                   : "+d"(sink)
                   : "l"(ptr + offset)
                   : "memory");
    }
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
  // write time and data back to memory
  startClk[tid] = start;
  stopClk[tid] = stop;
  dsink[tid] = sink;
}

void TestL1BandWidth(uint32_t test_cache_size, std::ofstream& ofs);
void GetSharedSize() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, l1_bw);
    std::cout << "function preferredShmemCarveout "
      << attr.preferredShmemCarveout << "%" << std::endl;
}

int main() {
    std::ofstream ofs("l1_bw.csv");
    ofs << "l1_size,bw(GB/s)\n";
    for (int i = 128;i < 256;++i) {
        ofs << i << ",";
        TestL1BandWidth(i * 1024, ofs);
    }
}

void TestL1BandWidth(uint32_t test_cache_size, std::ofstream& ofs) {
  uint32_t l1_double_count = test_cache_size / sizeof(double);

  uint32_t *startClk_host = new uint32_t[THREADS_NUM];
  uint32_t *endClk_host = new uint32_t[THREADS_NUM];

  uint32_t *posArray_dev;

  cudaMalloc(&posArray_dev, test_cache_size);
  uint32_t *startClk_dev;
  uint32_t *endClk_dev;
  double *dsink_dev;
  uint32_t *l1_size_dev;
  cudaMalloc(&startClk_dev, THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&endClk_dev, THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&dsink_dev, THREADS_NUM * sizeof(double));
  cudaMalloc(&l1_size_dev, sizeof(uint32_t));

  cudaMemcpy(l1_size_dev, &l1_double_count, sizeof(uint32_t), cudaMemcpyHostToDevice);

  l1_bw<<<1, THREADS_NUM>>>(startClk_dev, endClk_dev, dsink_dev, posArray_dev, l1_size_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(startClk_host, startClk_dev, THREADS_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(endClk_host, endClk_dev, THREADS_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  double cycle_count = 0;

  for (int i = 0; i < THREADS_NUM; ++i) {
    uint32_t single_thread_cycle = endClk_host[i] - startClk_host[i];
    cycle_count += single_thread_cycle;
    //std::cout << "thread " << i << " time:" << endClk_host[i] - startClk_host[i]
    //          << std::endl;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  double clock_rate = prop.clockRate * 1000;
  //std::cout << "Device freq:" << prop.clockRate << "khz" << std::endl;
  double load_time = THREADS_NUM / WARP_SIZE;

  double total_transfer_byte = test_cache_size * load_time;
  double t = cycle_count / THREADS_NUM / clock_rate;
  double gb = 1024*1024*1024;

  double bw_gb_s = total_transfer_byte/t/gb;

  std::cout << "TestSize: " << test_cache_size << std::endl;
  std::cout << "L1 BandWidth: " << bw_gb_s << "GByte/s" << std::endl;
  std::cout << total_transfer_byte / (cycle_count / THREADS_NUM) << "Bytes/cycle" << std::endl;
  ofs << bw_gb_s << "\n";

  cudaFree(posArray_dev);
  cudaFree(startClk_dev);
  cudaFree(endClk_dev);
  cudaFree(dsink_dev);
  cudaFree(l1_size_dev);

  delete[] startClk_host;
  delete[] endClk_host;
}
