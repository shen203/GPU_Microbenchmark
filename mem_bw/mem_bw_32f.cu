//This code is a modification of L2 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L2 cache
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

//This code have been tested on Volta V100 architecture

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//Array size must not exceed L2 size 
#define BLOCKS_NUM 640
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM*THREADS_NUM)
#define ARRAY_SIZE 805306368
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
__global__ void l2_bw (double*dsink, double*posArray, double*outArray){
	// block and thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;

	uint32_t id = bid * blockDim.x + tid; 

	// a register to avoid compiler optimization
	double sink0 = 0;
	double sink1 = 0;
//	double sink2 = 0;
//	double sink3 = 0;
//	double sink4 = 0;
//	double sink5 = 0;
//	double sink6 = 0;
//	double sink7 = 0;
	asm volatile("bar.sync 0;");

	// start timing
//	uint32_t start = 0;
//	asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// benchmark starts
	// load data from l2 cache and accumulate,
	for(uint32_t i = id*2; i<ARRAY_SIZE; i+=TOTAL_THREADS*2){
		double* ptr0 = posArray+i;
		double* ptr1 = outArray+i;
//		double* ptr2 = ptr0 + ARRAY_SIZE/2;
//		double* ptr3 = ptr1 + ARRAY_SIZE/2;
		asm volatile("{\t\n"
			".reg .f64 data<2>;\n\t"
			"ld.global.v2.f64 {data0,data1}, [%2];\n\t"
			"add.f64 %0, data0, %0;\n\t"
			"add.f64 %1, data1, %1;\n\t"
			"st.global.v2.f64 [%3], {data0,data1};\n\t"
			"}" : "+d"(sink0),"+d"(sink1) : "l"(ptr0), "l"(ptr1)
		);
	}
	asm volatile("bar.sync 0;");

	// stop timing
//	uint32_t stop = 0;
//	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// store the result
//	startClk[bid*THREADS_NUM+tid] = start;
//	stopClk[bid*THREADS_NUM+tid] = stop;
	dsink[bid*THREADS_NUM+tid] = sink0+sink1;
}

int main(){
//	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
//	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	double *outArray = (double*) malloc(ARRAY_SIZE*sizeof(double));

	double *posArray = (double*) malloc(ARRAY_SIZE*sizeof(double));
	double *dsink = (double*) malloc(TOTAL_THREADS*sizeof(double));

	double *outArray_g;
	double *posArray_g;
	double *dsink_g;
//	uint32_t *startClk_g;
//	uint32_t *stopClk_g;

        for (uint32_t i=0; i<ARRAY_SIZE; i++)
                posArray[i] = (double)i;

        gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(double)) );
        gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(double)) );
//	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
//	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&outArray_g, ARRAY_SIZE*sizeof(double)) );

        gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(double), cudaMemcpyHostToDevice) );


        l2_bw<<<BLOCKS_NUM,THREADS_NUM>>>(dsink_g, posArray_g, outArray_g);
	gpuErrchk( cudaPeekAtLastError() );
	
//	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
//	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(double), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(outArray, outArray_g, ARRAY_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );

/*
        for(int i=0; i<32; i++){
		//printf("startClk(%u) = %u, ", i, startClk[i]);
		//printf("stopClk(%u) = %u, ", i, stopClk[i]);
                printf("Clk(%u) = %u \n", i, stopClk[i]-startClk[i]);
        }
*/
//	double bw;
//	bw = ((double)(TOTAL_THREADS*4))/((double)(stopClk[0]-startClk[0]));
//	printf("bandwidth = %f (byte/cycle)\n", bw);
//	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

        return 0;
}
