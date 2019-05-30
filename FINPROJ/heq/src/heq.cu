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

__global__ void warmup(unsigned char *input, 
                       unsigned char *output){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
    output[location] = 0;

}

// only 4% ish of total runtime: prio (low)
// implement local hists?
// data packing?
__global__ void calc_hist(unsigned char *image,
               int size,
               unsigned int *hist) {
    
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int location = y*(gridDim.x*TILE_SIZE)+x;
   
    if(location >= size) return;

    atomicAdd(&hist[image[location]], 1);
}

// only 2% ish of total runtime: prio (very low)
__global__ void calc_lut(unsigned int* hist_to_lut) {
    // maybe make shared?
    unsigned int lut[256];
    lut[0] = 0;
    for (int ii = 1; ii < 256; ++ii) {
        lut[ii] = (lut[ii-1] + hist_to_lut[ii]);
    }
    for (int ii = 1; ii < 256; ++ii) {
        lut[ii] = ((1.0f * lut[ii]) / lut[255]) * 255;
    }
    for (int ii = 0; ii < 256; ++ii) {
        hist_to_lut[ii] = lut[ii];
    }
}

// more pixels per thread (data packing)?
__global__ void apply_lut(unsigned char* image,
               int size, 
               unsigned int* lut) {

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int location = y*(gridDim.x*TILE_SIZE)+x;

    if(location >= size) return;
    
    image[location] = lut[image[location]];
}

// play around with block size?
// occupancy already at 99%, block size work would be
// mostly for memory coalescing?
void gpu_function(unsigned char *data,  
                  unsigned int height, 
                  unsigned int width){
    
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	int size = XSize*YSize;

    ///////
    // sources:
    //      https://hackernoon.com/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23
    //      NUCAR GPU Class Slides
    //////


    unsigned char *image;
    unsigned int  *hist_to_lut;

    // alloc image  
	checkCuda(cudaMalloc((void**)&image, size*sizeof(unsigned char)));
    // alloc hist
	checkCuda(cudaMalloc((void**)&hist_to_lut, 256*sizeof(unsigned int)));
    // zero out histogram
    checkCuda(cudaMemset(hist_to_lut, 0, 256*sizeof(unsigned int)));
    // copy image
    checkCuda(cudaMemcpy(image, 
                data, 
                size*sizeof(unsigned char), 
                cudaMemcpyHostToDevice));
    
    // throughput is higher with pinned mem, but the images aren't big enought to overcome
    // the overhead of copying over to pinned mem

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
        
        // hist
        calc_hist<<<dimGrid, dimBlock>>>(image, size, hist_to_lut);

        // lut
        calc_lut<<<1,1>>>(hist_to_lut);
        
        // apply
         apply_lut<<<dimGrid, dimBlock>>>(image, size, hist_to_lut);
            
        // From here on, no need to change anything
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#ifdef CUDA_TIMING
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(data,
             image, 
             size*sizeof(unsigned char), 
             cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(image));
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

