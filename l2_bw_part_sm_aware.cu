#include <fstream>
#include <vector>
#include <iostream>
#include <stdint.h>

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define THREADS_NUM 1024
#define BLOCK_NUM 108
#define WARP_SIZE 32

uint32_t *MakeChaseBuffer(uint32_t size, uint32_t stride) {
  uint32_t element_count = size / sizeof(uint32_t);
  uint32_t *result = new uint32_t[element_count + WARP_SIZE * 2];
  int round = element_count / WARP_SIZE;
  for (uint32_t i = 0; i < round; ++i) {
    for (uint32_t j = 0; j < WARP_SIZE; ++j) {
      result[i * WARP_SIZE + j] = stride;
    }
  }
  return result;
}

__device__ __inline__ uint32_t get_smid() {
  uint32_t smid;    
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));    
  return smid;
}


__global__ void l2_chase(uint32_t *duration, uint32_t *dsink,
                         uint32_t *posArray, uint32_t chase_buffer_size,
                         uint32_t data_point_count,
                         uint32_t focus_smid) {
  uint32_t smid = get_smid();
  if (smid != focus_smid) {
    return;
  }
  // thread index
  uint32_t tid = threadIdx.x;
  // a register to avoid compiler optimization
  uint32_t sink = 0;
  // populate l2 cache to warm up
  // should populate by sm0
  for (uint32_t i = 0; i < chase_buffer_size; i+=1) {
    uint32_t *ptr = (uint32_t *)posArray + i;
    uint32_t idx;
    asm volatile("{\t\n"
                 "ld.global.cg.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(idx)
                 : "l"(ptr)
                 : "memory");
    sink += idx;
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  __shared__ uint32_t s_tvalue[4096];
  __shared__ uint32_t s_index[4096];

  posArray = posArray + tid;

  for (uint32_t i = 0; i < data_point_count; i+=1) {
    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
    uint32_t idx;
    asm volatile("{\t\n"
                 "ld.global.cg.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(idx)
                 : "l"(posArray)
                 : "memory");
    posArray += idx;
    s_index[i] = idx;
    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
    s_tvalue[i] = stop - start - 35; // 35 is overhead of clock
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // write time and data back to memory
  for (uint32_t i = 0; i < data_point_count; ++i) {
    duration[i] = s_tvalue[i];
    dsink[i] = s_index[i] + sink;
  }
}


__global__ void l2_bw(double* read_data, double *dsink,
                      int* l2_part_array, uint32_t l2_size_input,
                      int repeat) {
  uint32_t tid = threadIdx.x;
  uint32_t block_id = blockIdx.x;
  double sink = 0;

  uint32_t smid = get_smid();

  // 8KB each partition
  int start_offset = l2_part_array[smid] * 8192 / sizeof(double);

  //if (tid == 0)
  //printf("block id: %d, smid: %d, offset: %d\n", block_id, smid, start_offset);

  for (int r = 0; r < repeat; ++r)
  for (uint32_t i = start_offset + tid; i < l2_size_input; i += THREADS_NUM * 2) {
      asm volatile("{\t\n"
                   ".reg .f64 data;\n\t"
                   "ld.global.cg.f64 data, [%1];\n\t"
                   "add.f64 %0, data, %0;\n\t"
                   "}"
                   : "+d"(sink)
                   : "l"(read_data + i)
                   : "memory");
    
  }

  read_data[start_offset] = sink;
}

void TestL2BandWidth() {
  const int full_sm_count = 128;
  int* mask = new int[full_sm_count];
  //uint32_t l2_element_count = 8192 * 2 * 16 * 16;
  uint32_t l2_element_count = 8192;

  uint32_t stride = 128; // stride in elements

  uint32_t data_point_count = 64;

  
  uint32_t *chaseBuffer_host = MakeChaseBuffer(l2_element_count * sizeof(uint32_t), stride);

  uint32_t *chaseBuffer_dev;

  cudaMalloc(&chaseBuffer_dev,
             (l2_element_count) * sizeof(uint32_t));
  uint32_t *duration_dev;
  uint32_t *dsink_dev;
  cudaMalloc(&duration_dev, data_point_count * sizeof(uint32_t));
  cudaMalloc(&dsink_dev, l2_element_count * sizeof(uint32_t));

  cudaMemcpy(chaseBuffer_dev, chaseBuffer_host,
             (l2_element_count) * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  
  

  uint32_t **duration_hosts = new uint32_t*[full_sm_count];

  std::ofstream ofs("l2_latency_wrt_sm.csv");
  ofs << "address";
  for (int i = 0; i < full_sm_count; ++i) {
    ofs << ",sm_" << i;

    duration_hosts[i] = new uint32_t[data_point_count];
  }
  ofs << "\n";

  for (int i = 0;i < full_sm_count; ++i) {
    cudaMemset(duration_dev, 0, data_point_count*sizeof(uint32_t));
    l2_chase<<<BLOCK_NUM, 1>>>(duration_dev, dsink_dev,
                               chaseBuffer_dev, l2_element_count, data_point_count, i);

    gpuErrchk(cudaDeviceSynchronize());
    cudaMemcpy(duration_hosts[i], duration_dev, data_point_count * sizeof(uint32_t),
              cudaMemcpyDeviceToHost);
  }

  for (int i = 0;i < data_point_count; ++i) {
    ofs << std::hex << ((uint64_t)(chaseBuffer_dev + i * stride)) << std::dec;
    for (int j = 0;j < full_sm_count; ++j) {
      ofs << "," << duration_hosts[j][i];
    }
    ofs << "\n";
  }

  for (int j = 0;j < full_sm_count; ++j) {
    if (duration_hosts[j][0] > 250) {
      mask[j] = 1;
    }
    else {
      mask[j] = 0;
    }
  }

  int* l2_mask_dev;

  cudaMalloc(&l2_mask_dev, full_sm_count * sizeof(int));

  cudaMemcpy(l2_mask_dev, mask,
             full_sm_count * sizeof(int),
             cudaMemcpyHostToDevice);
  double *double_dsink_dev;
  cudaMalloc(&double_dsink_dev, BLOCK_NUM * THREADS_NUM * sizeof(double));
  
  int repeat = 10000;

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  l2_bw<<<BLOCK_NUM, 16>>>((double*)chaseBuffer_dev, double_dsink_dev,
                          l2_mask_dev, l2_element_count/2, repeat);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  gpuErrchk(cudaDeviceSynchronize());

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  double read_data_size = l2_element_count * sizeof(uint32_t) * repeat * BLOCK_NUM / 2;

  double bandwidth = read_data_size / (milliseconds/1000) / 1024/1024/1024;
  std::cout << "L2 bandwidth: " << bandwidth << "GB/s, kernel time:" << milliseconds << 
      "ms read " << read_data_size << "bytes\n";


  cudaFree(chaseBuffer_dev);
  cudaFree(duration_dev);
  cudaFree(dsink_dev);

  for (int i = 0;i < full_sm_count; ++i) {
    delete[] duration_hosts[i];
  }

  delete[] duration_hosts;
  delete[] chaseBuffer_host;
}


int main() {
  TestL2BandWidth();
}