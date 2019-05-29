
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
 
#include "config.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

__global__ void calc_hist(unsigned char *input_image, 
                       unsigned int *output_hist){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
    int val = input_image[location];
    atomicAdd(&(output_hist[val]), 1);

}
                
__global__ void update_px(unsigned char *input_image, 
                       unsigned int *input_lut,
                       unsigned char *output_image){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
    //output_image = input_lut[input_image[location]];
}

__global__ void warmup(unsigned char *input, 
                       unsigned char *output){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
    output[location] = 0;

}

void calc_hist(unsigned char *data,
               int size,
               unsigned int *hist) {
    for(int ii = 0; ii < size; ++ii) {
        hist[data[ii]] += 1;
    }
}

void calc_lut(unsigned int* lut,
              unsigned int* hist) {
    lut[0] = 0;
    for (int ii = 1; ii < 256; ++ii) {
        lut[ii] = (lut[ii-1] + hist[ii]);
    }
    for (int ii = 1; ii < 256; ++ii) {
        lut[ii] = ((1.0f * lut[ii]) / lut[255]) * 255;
    }
}

__global__ apply_lut(unsigned char* data,
               int size, 
               unsigned int* lut) {
    for(int ii = 0; ii < size; ++ii) {
        data[ii] = lut[data[ii]];
    }
}

// questions:
//why seperate buffers for input and output
//multiple kernals vs helper functions?
void gpu_function(unsigned char *data,  
                  unsigned int height, 
                  unsigned int width){
    
    unsigned char *image_in;
    unsigned char *image_out;
    unsigned int  *hist_to_lut;

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	int size = XSize*YSize;

    ///////
    // sources
    //      https://hackernoon.com/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23
    //////

    ////////////////////////
    // calculate histogram of image -> write a kernal for this:
    /////////////////////////

    // init hist
    unsigned int hist[256] = {0};

	// Allocate arrays in GPU memory
    // image in 
	checkCuda(cudaMalloc((void**)&image_in, size*sizeof(unsigned char)));
    // image out
	checkCuda(cudaMalloc((void**)&image_out, size*sizeof(unsigned char)));
    // histogram(out) / lut(in)
	checkCuda(cudaMalloc((void**)&hist_to_lut, 256*sizeof(unsigned int)));
	
    // zero out histogram
    checkCuda(cudaMemset(image_out, 0, size*sizeof(unsigned char)));
	
    // copy image to GPU
    checkCuda(cudaMemcpy(image, 
        data, 
        size*sizeof(unsigned char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// Kernel Call
	#ifdef CUDA_TIMING
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        // calculate histogram on gpu
        //calc_hist<<<dimGrid, dimBlock>>>(input_image, output_hist_lut);
        calc_hist(data, size, hist);
        

        // retrieve hist from gpu
        //checkCuda(cudaMemcpy(hist, 
        //        output_hist, 
        //        256*sizeof(unsigned int), 
        //        cudaMemcpyDeviceToHost));

        ///////////////
        // LUT is a prefix sum! -> bad target for parrallel (unless implement scan)
        // create LUT:
        //      LUT[0] = histogram[0]
        //      LUT[i] = LUT[i-1] + histogram[i]
        //      normalize LUT to 0-255
        //          - sub all by LUT[0], 
        //////////////
        unsigned int lut[256];
        calc_lut(lut, hist);
        
        // write lut to gpu
        //checkCuda(cudaMemcpy(output_hist_lut, 
        //        lut, 
         //       256*sizeof(unsigned int), 
          //      cudaMemcpyHostToDevice));

        ////////////
        // use LUT to compute new value -> write a kernal for this:
        // output[location] = lut[input[location]]
        ///////////
        //apply_lut(data, size, lut);
        apply_lut<<<dimGrid, dimBlock>>>(image_in, image_out, hist_to_lut);
            
        // From here on, no need to change anything
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#ifdef CUDA_TIMING
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(data,
                image_out, 
                size*sizeof(unsigned char), 
                cudaMemcpyDeviceToHost));
        
    // Free resources and end the program
	checkCuda(cudaFree(image_in));
	checkCuda(cudaFree(image_out));
	checkCuda(cudaFree(hist_to_lut));

}

void gpu_warmup(unsigned char *data, 
                unsigned int height, 
                unsigned int width){
    
    unsigned char *input_gpu;
    unsigned char *output_gpu;
     
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
            
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
    // Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    warmup<<<dimGrid, dimBlock>>>(input_gpu, 
                                  output_gpu);
                                         
    checkCuda(cudaDeviceSynchronize());
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));

}

