//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the latency of L1 cache

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <fstream>

//Define constants
#define L1_SIZE_BYTE (128*1024) //L1 size in bytes
#define L2_SIZE_BYTE (6144*1024) //L2 size in bytes
#define WARP_SIZE 32

//Varibales
#define L1_SIZE (L1_SIZE_BYTE/4) //L1 size in 32 bit words
#define L2_SIZE (L2_SIZE_BYTE/4) //L2 size in 32 bit words

#define SHARED_MEM_SIZE_BYTE (48*1024) //size in bytes, max 96KB for v100
#define SHARED_MEM_SIZE (SHARED_MEM_SIZE_BYTE/4)

#define ARRAY_SIZE (4*L2_SIZE)
#define BLOCK_SIZE 32   //Launch only one thread to calcaulte the latency using a pointer-chasing array technique
#define GRID_SIZE 1   
#define TOTAL_THREADS (BLOCK_SIZE*GRID_SIZE)

#define STRIDE_SIZE 1
#define WARMUP_ITER L2_SIZE/STRIDE_SIZE
#define TEST_ITER SHARED_MEM_SIZE


// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}
 
__global__ void l1_associativity(uint32_t *results, uint32_t *dsink, uint32_t stride, uint32_t *array){

	// thread index
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid*blockDim.x+tid;
    uint32_t n_threads = blockDim.x * gridDim.x;
	
	uint32_t start;
	uint32_t stop;
	__shared__ uint32_t time[SHARED_MEM_SIZE];
    //uint32_t *array = new uint32_t(ARRAY_SIZE);
    //__shared__ uint32_t start[SHARED_MEM_SIZE]; //static shared memory
	//__shared__ uint32_t stop[SHARED_MEM_SIZE]; //static shared memory
	// one thread to initialize the pointer-chasing array
	for (uint32_t i=uid; i<(ARRAY_SIZE); i+=n_threads)
		array[i] = (i+stride)%ARRAY_SIZE;
	__syncthreads();
	if(uid == 0){
        //initalize pointer chaser
		uint32_t p_chaser = 0;
		for(uint32_t i=0; i<WARMUP_ITER; ++i) {
			// chase pointer
			p_chaser = array[p_chaser];
		}

		for(uint32_t i=0; i<TEST_ITER; ++i) {	
			__syncthreads();

			// start timing
			start = clock();
			__syncthreads();

			// chase pointer
			p_chaser = array[p_chaser];
			__syncthreads();

			dsink[i] = p_chaser;
			__syncthreads();

			// stop timing
			stop = clock();
			time[i] = stop - start;
		}
		__syncthreads();
		// write time and data back to memory
		for (uint32_t i=0; i<SHARED_MEM_SIZE; i++){
			results[i] = time[i];
		}
	}

}

int main(){
    uint32_t *results = (uint32_t*) malloc(SHARED_MEM_SIZE*sizeof(uint32_t));
	uint32_t *dsink = (uint32_t*) malloc(TEST_ITER*sizeof(uint32_t));
	uint32_t *array = (uint32_t*) malloc(ARRAY_SIZE*sizeof(uint32_t));
	
	uint32_t *results_g;
	uint32_t *dsink_g;
	uint32_t *array_g;
		
	gpuErrchk( cudaMalloc(&results_g, SHARED_MEM_SIZE*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&dsink_g, TEST_ITER*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&array_g, ARRAY_SIZE*sizeof(uint32_t)) );
	
	l1_associativity<<<GRID_SIZE, BLOCK_SIZE>>>(results_g, dsink_g,STRIDE_SIZE, array_g);
    gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(results, results_g, SHARED_MEM_SIZE*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TEST_ITER*sizeof(uint32_t), cudaMemcpyDeviceToHost) );

	// write results to file
	std::ofstream myfile;
	myfile.open ("output_l1_associativity.csv", std::ios::app);
	for (uint32_t i=0; i<SHARED_MEM_SIZE; i++)
		myfile << results[i] << "\n";
	myfile.close();

	return 0;
} 
