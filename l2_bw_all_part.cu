// code from "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
#include <iostream>
#include <fstream>
#include <stdint.h>

//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define THREADS_NUM 1024
#define WARP_SIZE 32
#define BLOCK_NUM 108

#define STRIDE (BLOCK_NUM * THREADS_NUM)
#define ALL_THREAD_NUM STRIDE

__global__ void l2_bw(uint32_t *startClk, uint32_t *stopClk, double *dsink,
                      uint32_t *posArray, uint32_t *l2_size_input) {
  uint32_t l2_size = l2_size_input[0];
  uint32_t block_size = l2_size / BLOCK_NUM;
  // thread index
  uint32_t tid = threadIdx.x;
  uint32_t block_id = blockIdx.x;
  // a register to avoid compiler optimization
  double sink = 0;
  // populate l2 cache to warm up
  uint32_t start_offset = block_id * block_size;
  uint32_t end_offset = (block_id + 1) * block_size;
  for (uint32_t i = start_offset + tid; i < l2_size; i += THREADS_NUM) {
    double *ptr = (double *)posArray + i;
    asm volatile("{\t\n"
                 ".reg .f64 data;\n\t"
                 "ld.global.cg.f64 data, [%1];\n\t"
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

  for (uint32_t i = start_offset; i < end_offset; i += THREADS_NUM) {
    double *ptr = (double *)posArray + i;
      uint32_t offset = tid;
      asm volatile("{\t\n"
                   ".reg .f64 data;\n\t"
                   "ld.global.cg.f64 data, [%1];\n\t"
                   "add.f64 %0, data, %0;\n\t"
                   "}"
                   : "+d"(sink)
                   : "l"(ptr + offset)
                   : "memory");
    
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");
  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
  // write time and data back to memory
  startClk[block_id * THREADS_NUM + tid] = start;
  stopClk[block_id * THREADS_NUM + tid] = stop;
  dsink[block_id * THREADS_NUM + tid] = sink;
}

void TestL2BandWidth(uint32_t test_cache_size, std::ofstream& ofs);
void GetSharedSize() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, l2_bw);
    std::cout << "function preferredShmemCarveout "
      << attr.preferredShmemCarveout << "%" << std::endl;
}

int main() {
    std::ofstream ofs("l2_bw_system_part.csv");
    ofs << "l2_size(kb),bw(GB/s)\n";
    for (int i = 1 * 1024;i < 60*1024; i += 1) {
        ofs << i << ",";
        TestL2BandWidth(i * 1024, ofs);
    }
}

void TestL2BandWidth(uint32_t test_cache_size, std::ofstream& ofs) {
  uint32_t l2_double_count = test_cache_size / sizeof(double);

  uint32_t *startClk_host = new uint32_t[ALL_THREAD_NUM];
  uint32_t *endClk_host = new uint32_t[ALL_THREAD_NUM];

  uint32_t *posArray_dev;

  cudaMalloc(&posArray_dev, test_cache_size);
  uint32_t *startClk_dev;
  uint32_t *endClk_dev;
  double *dsink_dev;
  uint32_t *l2_size_dev;
  cudaMalloc(&startClk_dev, ALL_THREAD_NUM * sizeof(uint32_t));
  cudaMalloc(&endClk_dev, ALL_THREAD_NUM * sizeof(uint32_t));
  cudaMalloc(&dsink_dev, ALL_THREAD_NUM * sizeof(double));
  cudaMalloc(&l2_size_dev, sizeof(uint32_t));

  cudaMemcpy(l2_size_dev, &l2_double_count, sizeof(uint32_t), cudaMemcpyHostToDevice);

  l2_bw<<<BLOCK_NUM, THREADS_NUM>>>(startClk_dev, endClk_dev, dsink_dev, posArray_dev, l2_size_dev);

  gpuErrchk(cudaDeviceSynchronize());
  cudaMemcpy(startClk_host, startClk_dev, ALL_THREAD_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(endClk_host, endClk_dev, ALL_THREAD_NUM * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  double cycle_count = 0;

  for (int i = 0; i < ALL_THREAD_NUM; ++i) {
    uint32_t single_thread_cycle = endClk_host[i] - startClk_host[i];
    cycle_count += single_thread_cycle;
    //std::cout << "thread " << i << " time:" << endClk_host[i] - startClk_host[i]
    //          << std::endl;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  double clock_rate = prop.clockRate * 1000;
  //std::cout << "Device freq:" << prop.clockRate << "khz" << std::endl;

  //double load_time = THREADS_NUM / WARP_SIZE;
  double load_time = 1;
  double total_transfer_byte = test_cache_size * load_time;
  double t = cycle_count / ALL_THREAD_NUM / clock_rate;
  double gb = 1024*1024*1024;

  double bw_gb_s = total_transfer_byte/t/gb;

  std::cout << "TestSize: " << test_cache_size << std::endl;
  std::cout << "L2 BandWidth: " << bw_gb_s << "GByte/s" << std::endl;
  ofs << bw_gb_s << "\n";

  cudaFree(posArray_dev);
  cudaFree(startClk_dev);
  cudaFree(endClk_dev);
  cudaFree(dsink_dev);
  cudaFree(l2_size_dev);

  delete[] startClk_host;
  delete[] endClk_host;
}
