#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <mma.h>

#define THREADS_PER_BLOCK 1
#define THREADS_PER_SM 1
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 4096

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace nvcuda;

template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *a, T *b) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;

	//register T result = 0;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);
	wmma::load_matrix_sync(b_frag, b, 16);

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
		wmma::load_matrix_sync(a_frag, a, 16);
		wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
		wmma::store_matrix_sync(a, c_frag, 16, wmma::mem_row_major);
	}

	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	//res[gid] = result;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	half *data1 = (half*) malloc(TOTAL_THREADS*sizeof(half));
	half *data2 = (half*) malloc(TOTAL_THREADS*sizeof(half));
	//half *res = (half*) malloc(TOTAL_THREADS*sizeof(half));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	half *data1_g;
	half *data2_g;
	//half *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (half)i;
		data2[i] = (half)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(half)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(half)) );
	//gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(half)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(half), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(half), cudaMemcpyHostToDevice) );

	max_flops<half><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(data1, data1_g, TOTAL_THREADS*sizeof(half), cudaMemcpyDeviceToHost) );

	float latency;
	latency = ((float)(stopClk[0]-startClk[0]))/((float)(REPEAT_TIMES*4));
	printf("int32 latency = %f (clk)\n", latency);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
} 

