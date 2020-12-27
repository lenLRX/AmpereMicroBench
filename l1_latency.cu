#include <fstream>
#include <iostream>
#include <stdint.h>

#define THREADS_NUM 32
#define WARP_SIZE 32

uint32_t *MakeChaseBuffer(uint32_t size) {
  uint32_t element_count = size / sizeof(uint32_t);
  uint32_t *result = new uint32_t[element_count + WARP_SIZE * 2];
  int round = element_count / WARP_SIZE;
  for (uint32_t i = 0; i < round; ++i) {
    for (uint32_t j = 0; j < WARP_SIZE; ++j) {
      result[i * WARP_SIZE + j] = WARP_SIZE;
    }
  }
  return result;
}

__global__ void l1_chase(uint32_t *startClk, uint32_t *stopClk, uint32_t *dsink,
                         uint32_t *posArray, uint32_t *l1_size_input) {
  uint32_t l1_size = l1_size_input[0];
  uint32_t iter_num = l1_size / WARP_SIZE;
  // thread index
  uint32_t tid = threadIdx.x;
  // a register to avoid compiler optimization
  uint32_t sink = 0;
  // populate l1 cache to warm up
  for (uint32_t i = tid; i < l1_size; i += THREADS_NUM) {
    uint32_t *ptr = (uint32_t *)posArray + i;
    uint32_t idx;
    asm volatile("{\t\n"
                 "ld.global.ca.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(idx)
                 : "l"(ptr)
                 : "memory");
    sink += idx;
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  posArray = posArray + tid;

  for (uint32_t i = 0; i < iter_num; ++i) {
    uint32_t idx;
    asm volatile("{\t\n"
                 "ld.global.ca.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(idx)
                 : "l"(posArray)
                 : "memory");
    posArray += idx;
    sink += idx;
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

void Testl1Latency(uint32_t test_cache_size, std::ofstream &ofs) {
  uint32_t l1_element_count = test_cache_size / sizeof(uint32_t);

  uint32_t *startClk_host = new uint32_t[THREADS_NUM];
  uint32_t *endClk_host = new uint32_t[THREADS_NUM];
  uint32_t *chaseBuffer_host = MakeChaseBuffer(test_cache_size);

  uint32_t *chaseBuffer_dev;

  cudaMalloc(&chaseBuffer_dev, (l1_element_count + WARP_SIZE) * sizeof(uint32_t));
  uint32_t *startClk_dev;
  uint32_t *endClk_dev;
  uint32_t *dsink_dev;
  uint32_t *l1_size_dev;
  cudaMalloc(&startClk_dev, THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&endClk_dev, THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&dsink_dev, THREADS_NUM * sizeof(uint32_t));
  cudaMalloc(&l1_size_dev, sizeof(uint32_t));

  cudaMemcpy(l1_size_dev, &l1_element_count, sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(chaseBuffer_dev, chaseBuffer_host,
             (l1_element_count + WARP_SIZE) * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  l1_chase<<<1, THREADS_NUM>>>(startClk_dev, endClk_dev, dsink_dev,
                               chaseBuffer_dev, l1_size_dev);

  cudaDeviceSynchronize();
  cudaMemcpy(startClk_host, startClk_dev, THREADS_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(endClk_host, endClk_dev, THREADS_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  double cycle_count = 0;

  for (int i = 0; i < THREADS_NUM; ++i) {
    uint32_t single_thread_cycle = endClk_host[i] - startClk_host[i];
    cycle_count += single_thread_cycle;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  double clock_rate = prop.clockRate * 1000;
  // std::cout << "Device freq:" << prop.clockRate << "khz" << std::endl;

  double total_access_time = l1_element_count / WARP_SIZE;
  double t = cycle_count / THREADS_NUM / clock_rate;
  double gb = 1024 * 1024 * 1024;

  double bw_gb_s = test_cache_size / t / gb;
  double latency = cycle_count / THREADS_NUM / total_access_time;

  std::cout << "TestSize: " << test_cache_size << std::endl;
  std::cout << "Latency:" << latency << "cycle" << std::endl;
  ofs << latency << "\n";

  cudaFree(chaseBuffer_dev);
  cudaFree(startClk_dev);
  cudaFree(endClk_dev);
  cudaFree(dsink_dev);
  cudaFree(l1_size_dev);

  delete[] startClk_host;
  delete[] endClk_host;
  delete[] chaseBuffer_host;
}

int main() {
  std::ofstream ofs("l1_latency.csv");
  ofs << "l1_size(KB),latency(cycle)\n";
  for (int i = 128; i < 256; ++i) {
    ofs << i << ",";
    Testl1Latency(i * 1024, ofs);
  }
}