// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 64
//@@ insert code here



__device__ float clamp(float x, float start, float end){
  return min(max(x, start), end);
}

// the value here is referring to cdf[ucharImage[ii]]
__device__ float correct_color(float value, float cdfMin){
  return clamp(255.0*(value - cdfMin) / (1.0-cdfMin), 0.0, 255.0);
}

// this one will not print anything which is strange
// __global__ void rgbToGrayScale(unsigned char * RGBInput, unsigned char * GrayScaleOutput, int width, int height, int channels){
//   int Col = threadIdx.x + blockIdx.x * blockDim.x;
//   int Row = threadIdx.y + blockIdx.y * blockDim.y;
//   if(blockIdx.x == 0 && threadIdx.x == 0){
//     printf("gray scale ");
//   }
//   int grayOffset = Row * width + Col;
//   if(grayOffset < width * height) {
 
//     int rgbOffset = grayOffset*channels;
//     // unsigned char r = RGBInput[rgbOffset];
//     // unsigned char g = RGBInput[rgbOffset + 1];
//     // unsigned char b = RGBInput[rgbOffset + 2];
//     GrayScaleOutput[grayOffset] = (unsigned char) (0.21f*RGBInput[rgbOffset] + 0.71f*RGBInput[rgbOffset+1] + 0.07f*RGBInput[rgbOffset+2]);

//   }
// }
// __global__ void convertImage(float * input, unsigned char * output, int width, int height, int channels){
//   // the image is going to be stored in a 1d format
//   int index = blockIdx.x * blockDim.x + threadIdx.x;  
//   // conversion
//   if(index < width * height * channels){
//     output[index] = (unsigned char) (255* input[index]);
//   }
// }


__global__ void image_to_gray_and_uchar(float * input, unsigned char * output, unsigned char * gray, int width, int height, int channels){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < width * height){
    unsigned char r = (unsigned char) (input[3 * index] * 255.0);
    unsigned char g = (unsigned char) (input[3 * index + 1] * 255.0);
    unsigned char b = (unsigned char) (input[3 * index + 2] * 255.0);
    
    output[3 * index] = r;
    output[3 * index + 1] = g;
    output[3 * index + 2] = b;
    gray[index] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}




__global__ void createHistogram(unsigned char * grayImage, unsigned int * histogram, int width, int height){
  __shared__ unsigned int private_histo[HISTOGRAM_LENGTH]; // 256 element 1d array
  int t = threadIdx.x; // thread x 
  // if(blockIdx.x == 0 && threadIdx.x == 0){
  //   printf("histo ");
  // }
  if(t < HISTOGRAM_LENGTH){ // initialize shared mem values to 0
    private_histo[t] = 0;
  }
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;  // grab elements from the gray scale image
  int stride = blockDim.x * gridDim.x;            // grab all the threads
  
  while(i < width*height){                        // goes through all the elements in grayImage
    atomicAdd( &(private_histo[grayImage[i]]), 1);  
    i += stride;
  }

  __syncthreads();
  if(t < HISTOGRAM_LENGTH){
    atomicAdd( &(histogram[t]), private_histo[t]);
  }

}

__global__ void createOutputImage(unsigned char * ucharImage, float * cdf, float * output, int width, int height, int channels){
  // if(blockIdx.x == 0 && threadIdx.x == 0){
  //   // printf("out ");
  // }
  int index = blockIdx.x * blockDim.x + threadIdx.x;  
  if(index < width * height * channels){
    // ucharImage[index] = (unsigned char) min(max(255.0*(cdf[ucharImage[index]] - cdf[0]) / (1.0-cdf[0]), 0.0f), 255.0f);
    ucharImage[index] = (unsigned char) correct_color(cdf[ucharImage[index]], cdf[0]);
    output[index] = (float)(ucharImage[index] / 255.0f);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  // use to convert image from float to unsigned char
  float * deviceInputImageData; // the input image from the host
  float * deviceOutputImageData; // the output image after conversion
  unsigned char * ucharImage;
  unsigned char * grayScaleImageData;
  unsigned int * deviceHistogram;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int imageRawSize = imageWidth*imageHeight*imageChannels;
  cudaMalloc((void**)&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));          // device input
  cudaMalloc((void**)&ucharImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char)); // device output unsigned result
  cudaMalloc((void**)&grayScaleImageData, imageWidth*imageHeight*sizeof(unsigned char)); // gray scale image
  // histogram device array initialization
  cudaMalloc((void**)&deviceHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH*sizeof(unsigned int));
  
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);  // copy to device

  // convert image to unsigned char and gray scale
  dim3 convertBlock(BLOCK_SIZE, 1, 1); // block Dim
  dim3 convertGrid(ceil(imageWidth*imageHeight*(1.0)/BLOCK_SIZE),1,1); // grid Dim
  image_to_gray_and_uchar<<<convertGrid, convertBlock>>>(deviceInputImageData, ucharImage, grayScaleImageData, imageWidth, imageHeight, imageChannels);

  // dim3 convertBlock2(BLOCK_SIZE, 1, 1); // block Dim
  // dim3 convertGrid2(ceil(imageRawSize*(1.0)/BLOCK_SIZE),1,1); // grid Dim
  // convertImage<<<convertGrid2, convertBlock2>>>(deviceInputImageData, ucharImage, imageWidth, imageHeight, imageChannels);

  // convert image to gray scale
  // the block dim and grid dim are restricted to gray scale images
  // dim3 grayBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  // dim3 grayGrid(ceil(((float)imageWidth) / BLOCK_SIZE), ceil(((float)imageHeight)/BLOCK_SIZE), 1); // x for the width the cols, y for the height the rows
  // rgbToGrayScale<<<grayGrid, grayBlock>>>(ucharImage, grayScaleImageData, imageWidth, imageHeight, imageChannels);

  // unsigned char * tempData = (unsigned char *) malloc( 100 * sizeof(unsigned char));
  // cudaMemcpy(tempData, grayScaleImageData, 100*sizeof(unsigned char), cudaMemcpyDeviceToHost);  // copy to device
  // for(int i = 1; i < 30; ++i){
  //   // cdf[i] = cdf[i-1] + (float)(hostHistogram[i] / imageDim);
  //   printf("%d ", tempData[i]);
  // }


  // compute histogram from the grayScaleImage
  dim3 histoBlock(HISTOGRAM_LENGTH, 1, 1);
  dim3 histoGrid(ceil(imageWidth*imageHeight*1.0 / HISTOGRAM_LENGTH),1,1);
  createHistogram<<<histoGrid, histoBlock>>>(grayScaleImageData, deviceHistogram, imageWidth, imageHeight);

  // we are going to use cpu to do the work, since the scan size is relatively small
  unsigned int * hostHistogram = (unsigned int *)malloc(HISTOGRAM_LENGTH*sizeof(unsigned int));
  float * cdf = (float*)malloc(HISTOGRAM_LENGTH*sizeof(float));
  cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  // finding the cdf and free the hostHistogram after things are done
  float imageDim = imageWidth * imageHeight * 1.0;
  cdf[0] = (float)(hostHistogram[0] / imageDim);
  for(int i = 1; i < HISTOGRAM_LENGTH; ++i){
    cdf[i] = cdf[i-1] + (float)(hostHistogram[i] / imageDim);
  }

  float * deviceCDF;
  cudaMalloc((void**)&deviceCDF, HISTOGRAM_LENGTH*sizeof(float));
  cudaMalloc((void**)&deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMemcpy(deviceCDF, cdf, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 outBlock(BLOCK_SIZE, 1,1);
  dim3 outGrid(ceil(imageRawSize*(1.0)/BLOCK_SIZE),1,1);
  createOutputImage<<<outGrid, outBlock>>>(ucharImage, deviceCDF, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
  
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageRawSize*sizeof(float), cudaMemcpyDeviceToHost);


  wbSolution(args, outputImage);
  // wbExport("test_img", outputImage);
  //@@ insert code here
  free(hostHistogram);
  free(cdf);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(ucharImage);
  cudaFree(grayScaleImageData);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);

  return 0;
}
