// MP 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here

  // find out which block the idx belongs to, 
  // as well as the specific thread within the block
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < len){
    out[idx] = in1[idx] + in2[idx];
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");

  int loadSize = sizeof(float) * inputLength;
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput1, loadSize);
  cudaMalloc((void **) &deviceInput2, loadSize);
  cudaMalloc((void **) &deviceOutput, loadSize); 
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, loadSize, cudaMemcpyHostToDevice); // copy from host input 1 to the gpu
  cudaMemcpy(deviceInput2, hostInput2, loadSize, cudaMemcpyHostToDevice); // copy from host input 2 to the gpu

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 gridSize(inputLength / 256, 1, 1); // trying to accomplish this wtih 256 threads / block, a naive implementation
  if((inputLength % 256)){
    gridSize.x++;
  }
  dim3 blockSize(256, 1, 1); // trying to have 256 threads within a single block, right now it is using 1d representation

  wbTime_start(Compute, "Performing CUDA computation");

  //@@ Launch the GPU Kernel here
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, loadSize);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, loadSize, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
