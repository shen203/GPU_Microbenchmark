//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the latency of L1 cache 32f read

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define BLOCKS_NUM 1
#define THREADS_NUM 32
#define TOTAL_THREADS (BLOCKS_NUM*THREADS_NUM)
#define WARP_SIZE 32
#define ITERS 32768
#define L2_SIZE 1572864
#define ARRAY_SIZE (L2_SIZE*4)
// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

//Measure latency of 32768 reads. 
__global__ void mem_lat(uint32_t *startClk, uint32_t *stopClk, float *posArray, float *dsink){	
	// thread index
	uint32_t tid = threadIdx.x;
		// a register to avoid compiler optimization
		float sink = 0;
		
		// synchronize all threads
		asm volatile ("bar.sync 0;");
		// start timing
		uint32_t start = 0;
		asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
		for(uint32_t i=tid; i<ARRAY_SIZE; i+=TOTAL_THREADS) {
			float *ptr = posArray +	i;
			// load data from l1 cache and accumulate
			asm volatile ("{\t\n"
				".reg .f32 data;\n\t"
				"ld.global.cv.f32 data, [%1];\n\t"
				"add.f32 %0, data, %0;\n\t"
				"}" : "+f"(sink) : "l"(ptr) : "memory"
			);
			// synchronize all threads
			asm volatile("bar.sync 0;");
		}
		// stop timing
		uint32_t stop = 0;
		asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
		// write time and data back to memory
		startClk[tid] = start;
		stopClk[tid] = stop;
		dsink[tid] = sink;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	float *posArray = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *dsink = (float*) malloc(TOTAL_THREADS*sizeof(float));
	
	uint32_t *startClk_g;
        uint32_t *stopClk_g;
        float *posArray_g;
        float *dsink_g;
	
	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(float)) );
	
	mem_lat<<<BLOCKS_NUM,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );
	printf("L1 Latency for %d threads = %u \n", TOTAL_THREADS, (stopClk[0]-startClk[0]));

	return 0;
} 