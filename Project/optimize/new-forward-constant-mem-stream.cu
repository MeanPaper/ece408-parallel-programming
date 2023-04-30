#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define STREAM_NUM  4
__constant__ float kernel_mask[1 * 7 * 7 * 4 * 16];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, 
    const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = ceil((Width_out*1.0)/TILE_WIDTH);
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) kernel_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    
    int n, m, h, w, c;  // image_index, map_inedx, specific_height, specific_width, channel
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0;
    if(h < Height_out && w < Width_out){
        for(c = 0; c < Channel; ++c){
            for(int p = 0; p < K; ++p){     // for loop, the mask K x K
                for(int q = 0; q < K; ++q){
                    acc += in_4d(n, c, h+p, w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(n,m,h,w) = acc;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
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


    // defined the grids required for the output
    int H_grid = ceil((H_out*1.0)/TILE_WIDTH);  // height grids 
    int W_grid = ceil((W_out*1.0)/TILE_WIDTH);  // width grids
    int Z = H_grid * W_grid;                    // h x w requirements for the image
    const int segSize = 25; // define seg size
    
    cudaStream_t streams[STREAM_NUM];
    for(int i = 0; i < STREAM_NUM; ++i){    // create streams
        cudaStreamCreate(&streams[i]);
    } 
    
    int input_CHW = Channel * Height * Width;
    int output_MHW = Map_out * H_out * W_out;
    int in_copy_size = segSize * input_CHW;
    int out_copy_size = segSize * output_MHW;
    
    dim3 DimBlocks(TILE_WIDTH,TILE_WIDTH, 1); // the un-optimized one
    dim3 DimGrids(segSize, Map_out, Z);              // grid dimension

    for (int i = 0; i < Batch; i += (STREAM_NUM * segSize)){

        // input offsets
        int offset0 = (i + 0 * segSize) * input_CHW;
        int offset1 = (i + 1 * segSize) * input_CHW;
        int offset2 = (i + 2 * segSize) * input_CHW;
        int offset3 = (i + 3 * segSize) * input_CHW;

        // output offset
        int out_offset0 = (i + 0 * segSize) * output_MHW;
        int out_offset1 = (i + 1 * segSize) * output_MHW;
        int out_offset2 = (i + 2 * segSize) * output_MHW;
        int out_offset3 = (i + 3 * segSize) * output_MHW;

        // async cpy input, host to device
        cudaMemcpyAsync(*device_input_ptr + offset0, host_input + offset0, in_copy_size * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        cudaMemcpyAsync(*device_input_ptr + offset1, host_input + offset1, in_copy_size * sizeof(float), cudaMemcpyHostToDevice, streams[1]);
        cudaMemcpyAsync(*device_input_ptr + offset2, host_input + offset2, in_copy_size * sizeof(float), cudaMemcpyHostToDevice, streams[2]);
        cudaMemcpyAsync(*device_input_ptr + offset3, host_input + offset3, in_copy_size * sizeof(float), cudaMemcpyHostToDevice, streams[3]);

        // stream kernel calls
        conv_forward_kernel<<<DimGrids, DimBlocks, 0, streams[0]>>>(*device_output_ptr + out_offset0, *device_input_ptr + offset0, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<DimGrids, DimBlocks, 0, streams[1]>>>(*device_output_ptr + out_offset1, *device_input_ptr + offset1, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<DimGrids, DimBlocks, 0, streams[2]>>>(*device_output_ptr + out_offset2, *device_input_ptr + offset2, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<DimGrids, DimBlocks, 0, streams[3]>>>(*device_output_ptr + out_offset3, *device_input_ptr + offset3, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);

        // async cpy output, device to host
        cudaMemcpyAsync((void*)(host_output + out_offset0), *device_output_ptr + out_offset0, out_copy_size * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        cudaMemcpyAsync((void*)(host_output + out_offset1), *device_output_ptr + out_offset1, out_copy_size * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
        cudaMemcpyAsync((void*)(host_output + out_offset2), *device_output_ptr + out_offset2, out_copy_size * sizeof(float), cudaMemcpyDeviceToHost, streams[2]);
        cudaMemcpyAsync((void*)(host_output + out_offset3), *device_output_ptr + out_offset3, out_copy_size * sizeof(float), cudaMemcpyDeviceToHost, streams[3]);
    }

    cudaFree(*device_input_ptr);
    cudaFree(*device_output_ptr);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // int H_out = Height - K + 1;
    // int W_out = Width - K + 1;
    
    // // defined the grids required for the output
    // int H_grid = ceil((H_out*1.0)/TILE_WIDTH);  // height grids 
    // int W_grid = ceil((W_out*1.0)/TILE_WIDTH);  // width grids
    // int Z = H_grid * W_grid;                    // h x w requirements for the image


    return;
    // cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    // int H_out = Height - K + 1;
    // int W_out = Width - K + 1;    
    // Free device memory
    return;
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
