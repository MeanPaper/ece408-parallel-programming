// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define BLOCK_SIZE_TWO 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float * S, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float TempScan[2*BLOCK_SIZE];
  int bid = blockIdx.x;
  int bdim = blockDim.x;
  int tid = threadIdx.x;
  int index = 2 * bid * bdim + tid;
  
  if(index < len) TempScan[tid] = input[index];
  else TempScan[tid] = 0.0;
  
  if(index + bdim < len) TempScan[tid+bdim] = input[index + bdim];
  else TempScan[tid+bdim] = 0.0;
  

  // Bent-Kung algo from the lecture
  int stride = 1;
  while(stride < 2*bdim){
    __syncthreads();
    int idx = (tid + 1) * stride * 2 - 1;
    if(idx < 2*bdim && (idx - stride) >= 0){
      TempScan[idx] += TempScan[idx-stride];
    }
    stride = stride *2;
  }

  stride = bdim / 2;
  while(stride > 0){
    __syncthreads();
    int idx = (tid + 1) * stride * 2 - 1;
    if ((idx + stride) < 2*bdim){
      TempScan[idx + stride] += TempScan[idx];
    }
    stride = stride / 2;
  }
  

  // in place change
  __syncthreads();
  if (index < len) output[index] = TempScan[tid];
  if (index + bdim < len) output[index + bdim] = TempScan[tid + bdim];

  // use output to do extra work
  if (S != NULL){ 
    __syncthreads();
    if(tid == (bdim-1)){
      S[bid] = TempScan[2*bdim-1];
    }
  }
}


__global__ void parallelAdd(float * input, float * S, int len){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < len && blockIdx.x > 0){
    input[index] += S[blockIdx.x-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  

  int s_length = ceil(numElements / (BLOCK_SIZE*2.0));
  float * deviceS;  // use by device, store the max prefix sum from each block
  float * hostS = (float*) malloc(s_length * sizeof(float));
  cudaMalloc((void **)&deviceS, s_length * sizeof(float));

  dim3 dimGrid(ceil(numElements / (BLOCK_SIZE*2.0)),1,1); // use much less blocks
  dim3 dimBlock(BLOCK_SIZE, 1, 1); 

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // part 1, reduction scan to compute partial prefix sum, and output max prefix sum of each block
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceS, numElements);
  cudaDeviceSynchronize();

  // part 2, compute the prefix sum elements in deviceS  
  dim3 dimGrid2(ceil(s_length / (BLOCK_SIZE_TWO*2.0)),1,1);
  dim3 dimBlock2(BLOCK_SIZE_TWO,1,1);
  scan<<<dimGrid2, dimBlock2>>>(deviceS, deviceS, NULL, s_length);
  cudaDeviceSynchronize();
  
  // part 2 CPU implementation
  // cudaMemcpy(hostS, deviceS, s_length*sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 1; i < s_length; ++i){
  //   hostS[i] += hostS[i-1];  
  // }
  // cudaMemcpy(deviceS, hostS, s_length*sizeof(float), cudaMemcpyHostToDevice);
  // free(hostS);
  
  // part 3, add the sum to each elements
  dim3 dimGrid3(ceil(numElements / (BLOCK_SIZE*2.0)),1,1); 
  dim3 dimBlock3(BLOCK_SIZE * 2, 1, 1);                   
  parallelAdd<<<dimGrid3, dimBlock3>>>(deviceOutput, deviceS, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceS);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
