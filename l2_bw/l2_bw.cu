//This code is a modification of L2 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L2 cache
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bandwidth.cu

//This code have been tested on Volta V100 architecture

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//Array size must not exceed L2 size 
#define BLOCKS_NUM 80
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM * THREADS_NUM)
#define REPEAT_TIMES 64 
#define ARRAY_SIZE (TOTAL_THREADS + REPEAT_TIMES)
#define WARP_SIZE 32 
#define L2_SIZE 98304 //number of doubles can store

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/
__global__ void l2_bw (uint32_t*startClk, uint32_t*stopClk, double*dsink, double*posArray){
	// block and thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;

	// a register to avoid compiler optimization
	double sink = 0;
	
	// warm up l2 cache
	for(uint32_t i = tid; i<ARRAY_SIZE; i+=THREADS_NUM){
		double* ptr = posArray+i;
		// every warp loads all data in l2 cache
		asm volatile("{\t\n"
			".reg .f64 data;\n\t"
			"ld.global.cg.f64 data, [%1];\n\t"
			"add.f64 %0, data, %0;\n\t"
			"}" : "+d"(sink) : "l"(ptr) : "memory"
		);
	}
	
	asm volatile("bar.sync 0;");	

	// start timing
	uint32_t start = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// benchmark starts
	// load data from l2 cache and accumulate,
	for(uint32_t i = 0; i<REPEAT_TIMES; i++){
		for (uint32_t j = tid; j<TOTAL_THREADS; j+=THREADS_NUM){
			double* ptr = posArray+i+j;
			asm volatile("{\t\n"
				".reg .f64 data;\n\t"
				"ld.global.cg.f64 data, [%1];\n\t"
				"add.f64 %0, data, %0;\n\t"
				"}" : "+d"(sink) : "l"(ptr) : "memory"
			);
		}
	}
	asm volatile("bar.sync 0;");
	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// store the result
	startClk[bid*THREADS_NUM+tid] = start;
	stopClk[bid*THREADS_NUM+tid] = stop;
	dsink[bid*THREADS_NUM+tid] = sink;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
        uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));

	double *posArray = (double*) malloc(ARRAY_SIZE*sizeof(double));
	double *dsink = (double*) malloc(TOTAL_THREADS*sizeof(double));

	double *posArray_g;
	double *dsink_g;
	uint32_t *startClk_g;
        uint32_t *stopClk_g;

        for (int i=0; i<ARRAY_SIZE; i++)
                posArray[i] = (double)i;

        gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(double)) );
        gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(double)) );
        gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
        gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );

        gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(double), cudaMemcpyHostToDevice) );


        l2_bw<<<BLOCKS_NUM,THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g, posArray_g);
	gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(double), cudaMemcpyDeviceToHost) );

        for(int i=0; i<32; i++){
		printf("startClk(%u) = %u, ", i, startClk[i]);
		printf("stopClk(%u) = %u, ", i, stopClk[i]);
                printf("Clk(%u) = %u \n", i, stopClk[i]-startClk[i]);
        }

        return 0;
}