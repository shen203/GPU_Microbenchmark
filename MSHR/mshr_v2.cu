//This code is a modification of MSHR micro benchmark from 
//"A Detailed GPU Cache Model Based on Reuse Distance Theory": https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6835955&tag=1

//This benchmark measures the MSHR size 

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <fstream>

//#define NUM_LOADS 6
//#define NUM_WARPS 1
 
#define THREADS_PER_BLOCK (NUM_WARPS*32)
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define ARRAY_SIZE (THREADS_PER_BLOCK*32+THREADS_PER_BLOCK*NUM_WARPS*32)
#define REPEATS 10
// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void mshr(uint32_t *startClk, uint32_t *stopClk, float *dsink, float *posArray){
	
	// thread index
	uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
	
	//
	float sink = 0;
	float tmp;
	uint32_t start = 0;
	uint32_t stop = 0;
	asm volatile("bar.sync 0;");
	if (uid % 32 == 0){
		asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
		
		for (uint32_t i=0; i<NUM_LOADS; i++){
			tmp = posArray[32*(uid + i*NUM_WARPS*32)];
			sink = sink+tmp;
		}
		asm volatile("bar.sync 0;");	
		asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
		
	}	
	
	// write time and data back to memory
	startClk[uid] = start;
	stopClk[uid] = stop;
	dsink[uid] = sink;
}

uint32_t findmin(uint32_t arr[], int n){
	uint32_t tmp = arr[0];
	for (int i=0; i<n; i++){
		if(tmp>arr[i]){
			tmp = arr[i];
		}
	}
	return tmp;
}

uint32_t benchmark_host(loads, warps){
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
	
	gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	
	mshr<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g, posArray_g);
        gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );
	
	cudaFree(posArray_g);
	cudaFree(startClk_g);
	cudaFree(stopClk_g);
	cudaFree(dsink_g);
	
	free(posArray);
	free(dsink);
	printf("latency for %d loads and %d warps = %u \n", NUM_LOADS, NUM_WARPS, stopClk[0]-startClk[0]);

	return (stopClk[0]-startClk[0]);
}

int main(){
	uint32_t tmp[REPEATS];
	uint32_t clk;
	for (int i=0; i<REPEATS; ++i){
		tmp[i] = benchmark_host(j);
	}

	clk = findmin(tmp,REPEATS);
	std::ofstream myfile;
       	myfile.open ("output.csv", std::ios::app);
       	myfile << NUM_LOADS << "," << NUM_WARPS << "," << clk << "\n";
       	myfile.close();
}
 
