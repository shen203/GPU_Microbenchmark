#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <iostream>
#include <algorithm>

#define BLOCKS_NUM 160
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM * THREADS_NUM)
#define REPEAT_TIMES 2048 
#define WARP_SIZE 32 
#define ARRAY_SIZE (TOTAL_THREADS + REPEAT_TIMES*WARP_SIZE)

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


template <class T>
__global__ void max_flops(uint64_t *startClk, uint64_t *stopClk, T *data1, T *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	//register T s1 = data1[gid];
	//register T s2 = data2[gid];
	//register T result = 0;
	// synchronize all threads
	//int32_t res0, res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13, res14, res15;
	int32_t sum;
	asm volatile ("bar.sync 0;");

	// start timing
	uint64_t start = clock64();

	for(uint32_t i = 0; i<REPEAT_TIMES; i++){
		sum = sum + atomicAdd(&data1[(i*WARP_SIZE)+gid], 10);
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint64_t stop = clock64();

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = sum;
}

int main(){
	uint64_t *startClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));

	//int32_t *data2 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *res = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *data1 = (int32_t*) malloc(ARRAY_SIZE*sizeof(int32_t));

	uint64_t *startClk_g;
	uint64_t *stopClk_g;
	int32_t *data1_g;
	//int32_t *data2_g;
	int32_t *res_g;

	for (uint32_t i=0; i<ARRAY_SIZE; i++) {
		data1[i] = (int32_t)i;
		//data2[i] = (int32_t)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&data1_g, ARRAY_SIZE*sizeof(int32_t)) );
	//gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(int32_t)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, ARRAY_SIZE*sizeof(int32_t), cudaMemcpyHostToDevice) );
	//gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );

	max_flops<int32_t><<<BLOCKS_NUM,THREADS_NUM>>>(startClk_g, stopClk_g, data1_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyDeviceToHost) );

	float bw;
	uint64_t total_time = *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS])-*std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
	bw = (((float)REPEAT_TIMES*(float)TOTAL_THREADS*4*8)/(float)(total_time));
	printf("int32 bendwidth = %f (byte/clk)\n", bw);
	printf("Total Clk number = %ld \n", total_time);

	return 0;
} 

