
//This benchmark measures the kernel overhead as linear function a + Xb where X is the number of launched TBs


#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_NUM 1024     // one thread to initialize the pointer-chasing array
#define WARP_SIZE 32
#define ARRAY_SIZE 4096

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void kernel_lat_1TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_2TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_4TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_8TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_16TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_32TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_64TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_128TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_256TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}


__global__ void kernel_lat_512TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_1024TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

__global__ void kernel_lat_2048TB(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink){

	
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(THREADS_NUM*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(THREADS_NUM*sizeof(uint32_t));
	uint64_t *dsink = (uint64_t*) malloc(THREADS_NUM*sizeof(uint64_t));
	
	uint32_t *startClk_g;
        uint32_t *stopClk_g;
        uint64_t *posArray_g;
        uint64_t *dsink_g;
	
	gpuErrchk( cudaMalloc(&startClk_g, THREADS_NUM*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, THREADS_NUM*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&dsink_g, THREADS_NUM*sizeof(uint64_t)) );
	
	kernel_lat_1TB<<<1,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );



	kernel_lat_2TB<<<2,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );


/*
	kernel_lat_4TB<<<4,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	kernel_lat_8TB<<<8,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	kernel_lat_16TB<<<16,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	kernel_lat_32TB<<<32,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );


	kernel_lat_64TB<<<64,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	kernel_lat_128TB<<<128,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );


	kernel_lat_256TB<<<256,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	kernel_lat_512TB<<<1024,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

	kernel_lat_1024TB<<<1024,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );

*/

	kernel_lat_2048TB<<<2048,THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	gpuErrchk( cudaPeekAtLastError() );


	gpuErrchk( cudaMemcpy(startClk, startClk_g, THREADS_NUM*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, THREADS_NUM*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, THREADS_NUM*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	printf("Kernel Launch Latency : Check CUDA profiler events\n");

	return 0;
} 
