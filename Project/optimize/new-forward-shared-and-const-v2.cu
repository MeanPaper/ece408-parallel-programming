#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH  16
#define MASK_SIZE   7 
__constant__ float kernel_mask[1*7*7*4*16];


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = ceil((Width_out*1.0)/TILE_WIDTH);
    const int blocksize = TILE_WIDTH + K - 1;

    extern __shared__ float tileMem[]; // shared memory

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) kernel_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shareMem(i2, i1, i0) tileMem[(i2)*(blocksize*blocksize) + (i1) * blocksize + (i0)] // use for 3d shared memory
    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int c;  // image_index, map_inedx, specific_height, specific_width, channel
    int h_topleft= (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_topleft = (blockIdx.z % W_grid) * TILE_WIDTH;
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = h_topleft + ty; // the output height index
    int w = w_topleft + tx; // the output width index
    float acc = 0.0;
    
    // with shared tiles, may be able to reduce sync_threads
    for(c = 0; c < Channel; ++c){
        for(int i = ty; i < blocksize; i+= TILE_WIDTH){
            for(int j = tx; j < blocksize; j += TILE_WIDTH){
                if(h_topleft + i < Height && w_topleft + j < Width){
                    shareMem(c,i,j) = in_4d(n, c, h_topleft + i, w_topleft + j);
                }
                else{
                    shareMem(c,i,j) = 0.0f;
                }
            }
        }
    }

    __syncthreads();

    if (h < Height_out && w < Width_out){   
        for(c = 0; c < Channel; ++c){  
            for(int p = 0; p < K; ++p){     // for loop, the mask K x K
                for(int q = 0; q < K; ++q){
                    acc += shareMem(c ,ty+p , tx+q) * mask_4d(m,c,p,q);
                    // acc += in_4d(n, c, h+p, w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(n,m,h,w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef shareMem
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    int H_out = Height - K + 1; // compute the output height
    int W_out = Width - K + 1;  // compute the input width
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * H_out * W_out * sizeof(float));   // using the output height and width
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));   // using the input height and width       
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice); // copy host input to device input    
    cudaMemcpyToSymbol(kernel_mask, host_mask, 1 * 7 * 7 * 4 * 16*sizeof(float)); // copy to the constant memory

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    
    // defined the grids required for the output
    int H_grid = ceil((H_out*1.0)/TILE_WIDTH);  // height grids 
    int W_grid = ceil((W_out*1.0)/TILE_WIDTH);  // width grids
    int Z = H_grid * W_grid;                    // h x w requirements for the image
    int blocksize = TILE_WIDTH + K - 1;

    // dim3 DimBlocks(CACHE_SIZE,CACHE_SIZE, 1); // block dimension
    dim3 DimBlocks(TILE_WIDTH,TILE_WIDTH, 1); // the un-optimized one
    dim3 DimGrids(Batch, Map_out, Z);         // grid dimension

    conv_forward_kernel<<<DimGrids, DimBlocks, Channel * blocksize * blocksize *sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, H_out * W_out * Batch * Map_out * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}