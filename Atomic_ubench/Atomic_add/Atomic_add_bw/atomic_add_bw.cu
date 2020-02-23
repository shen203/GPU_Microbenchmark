#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <iostream>
#include <algorithm>

#define THREADS_PER_BLOCK 1024
#define BLOCKS_NUM 160
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 1

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1, T *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	//register T s1 = data1[gid];
	//register T s2 = data2[gid];
	//register T result = 0;
	// synchronize all threads
	int32_t res0, res1, res2, res3, res4, res5, res6, res7;
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
		res0 = atomicAdd(&data1[gid+0*TOTAL_THREADS], 10);
		res1 = atomicAdd(&data1[gid+1*TOTAL_THREADS], 10);
		res2 = atomicAdd(&data1[gid+2*TOTAL_THREADS], 10);
		res3 = atomicAdd(&data1[gid+3*TOTAL_THREADS], 10);

		res4 = atomicAdd(&data1[gid+4*TOTAL_THREADS], 10);
		res5 = atomicAdd(&data1[gid+5*TOTAL_THREADS], 10);
		res6 = atomicAdd(&data1[gid+6*TOTAL_THREADS], 10);
		res7 = atomicAdd(&data1[gid+7*TOTAL_THREADS], 10);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = res0 + res1 + res2 + res3 + res4 + res5 + res6 + res7;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	int32_t *data1 = (int32_t*) malloc(TOTAL_THREADS*8*sizeof(int32_t));
	//int32_t *data2 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *res = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	int32_t *data1_g;
	//int32_t *data2_g;
	int32_t *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS*8; i++) {
		data1[i] = (int32_t)i;
		//data2[i] = (int32_t)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*8*sizeof(int32_t)) );
	//gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(int32_t)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*8*sizeof(int32_t), cudaMemcpyHostToDevice) );
	//gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );

	max_flops<int32_t><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyDeviceToHost) );

	float bw;
	uint32_t total_time = *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS-1])-*std::min_element(&startClk[0],&startClk[TOTAL_THREADS-1]);
	bw = ((float)(REPEAT_TIMES*TOTAL_THREADS*4*8)/(float)(total_time));
	printf("int32 bendwidth = %f (byte/clk)\n", bw);
	printf("Total Clk number = %u \n", total_time);

	return 0;
} 

