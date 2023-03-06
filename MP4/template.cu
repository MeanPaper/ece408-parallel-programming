#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_WIDTH 3  // the kernel mask
#define RADIUS 1      // radius of the mask
#define TILE_WIDTH 3  // tile for the 3D cube
#define CACHE_WIDTH TILE_WIDTH + KERNEL_WIDTH - 1  // cube width of the cache, which is 5 here

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float convTile[CACHE_WIDTH][CACHE_WIDTH][CACHE_WIDTH]; // the cache for the tiling
  int tz = threadIdx.z, ty = threadIdx.y, tx = threadIdx.x; // thread z, y, x

  int z_out = blockIdx.z * TILE_WIDTH + tz;   // index of z, within the expanded thread block
  int y_out = blockIdx.y * TILE_WIDTH + ty;   // index of y, within the expanded thread block
  int x_out = blockIdx.x * TILE_WIDTH + tx;   // index of x, within the expanded thread block

  // change to input coordinates
  int z_in = z_out - RADIUS;
  int y_in = y_out - RADIUS;
  int x_in = x_out - RADIUS;

  float partial = 0.0f;

  if(0 <= z_in && z_in < z_size &&
     0 <= y_in && y_in < y_size &&
     0 <= x_in && x_in < x_size){
    convTile[tz][ty][tx] = input[z_in*(x_size*y_size) + y_in*x_size + x_in]; 
  }
  else{
    convTile[tz][ty][tx] = 0.0f;
  }  
  
  __syncthreads();
  
  if(tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH){
    for(int i = 0; i < TILE_WIDTH; ++i){ // for z index
      for(int j = 0; j < TILE_WIDTH; ++j){ // for y index
        for(int k = 0; k < TILE_WIDTH; ++k){ // for x index
          partial += deviceKernel[i][j][k] * convTile[i+tz][j+ty][k+tx];
        }
      }
    }
    if(z_out < z_size && y_out < y_size && x_out < x_size){
      output[z_out*x_size*y_size + y_out*x_size + x_out] = partial;
    } 
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength-3)*sizeof(float)); // device input mem
  cudaMalloc((void**) &deviceOutput, (inputLength-3)*sizeof(float)); // device output mem

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice); // copy host input to the device input
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float)); // copy to the constant memory

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size)/TILE_WIDTH),ceil((1.0*y_size)/TILE_WIDTH), ceil((1.0*z_size)/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH -1, TILE_WIDTH + KERNEL_WIDTH -1);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
