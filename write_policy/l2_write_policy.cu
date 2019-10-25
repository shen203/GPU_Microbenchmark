#include <iostream>
#include <cstdio>
using namespace std;
#include <cuda_runtime.h>
#define TIMES 24

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////HELP FUNCTIONS/////////////////////////////////////////////////
void RandomInit(float* data, int n)
{
    for (int i=0; i<n; i++)
	{
        data[i] = rand() / (float)RAND_MAX;
	}
}

void RandomInit(unsigned* data, int n)
{
    for (int i=0; i<n; i++)
	{
        data[i] = rand() % n;
	}
}
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////_VECTOR_ADDITION_///////////////////////////////////////////////////////
// Device code


__global__ void irreguler(const float* A, float* C, float* F)	
	
{	
int i = blockDim.x * blockIdx.x + threadIdx.x;	
	if(i == 0)
	{
	C[i] = A[i];
	__syncthreads();
	//high latncy
	C[i] = C[i] + A[i];
	__syncthreads();
	
	F[i] = C[i+1];
	}
	
}

__global__ void mb2(float* A, float* C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i == 0) {
	C[i] = A[i]; //write to C[i] is a miss (cache line is missing)
	__syncthreads();
	C[i+1] = A[i]; //write to C[i+1] is a hit (cache line is found)
	__syncthreads();
	C[i] = C[i] + A[i]; //read of C[i] is a miss (entire sector is missing, fetch it from memory)
	__syncthreads();
	A[i] = C[i] + C[i+1]; //read C[i] and C[i+1] are hits (entire sector exists)
	}
}	


__global__ void l1(float* A, float* C)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i == 0) {
		C[i] = A[i] + A[i+32];

                __syncthreads();

        C[i] = A[i] + A[i+32] + C[i];
	}
}	


// Host code
void VectorAddition(int N, int threadsPerBlock)
{
	cout<<"Vector Addition for input size "<<N<<" :\n";
	// Variables
	float* h_A;
	float* h_C;

	
	float* d_A;
	float* d_C;
	
	float total_time=0;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_C = (float*)malloc(size);

    
    // Initialize input vectors
    RandomInit(h_A, N);

	
    // Allocate vectors in device memory
    checkCudaErrors( cudaMalloc((void**)&d_A, size) );
    checkCudaErrors( cudaMalloc((void**)&d_C, size) );

	
    // Copy vectors from host memory to device memory
    checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
	
	checkCudaErrors(cudaThreadSynchronize());
    // Invoke kernel
	cout<<"Invoke Kernel\n";	
	//int threads = 128;
    int blocksPerGrid = ((N+ threadsPerBlock-1) / threadsPerBlock);
  
  
	for (int i = 0; i < 1; i++) {
    l1<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C);
    getLastCudaError("kernel launch failure");
	checkCudaErrors(cudaThreadSynchronize());
	}

	float dSeconds = total_time/((float)TIMES * 1000);
	float dNumOps = N;
	float gflops = 1.0e-9 * dNumOps/dSeconds;
	cout<<"Time = "<<dSeconds*1.0e3<< "msec"<<endl<<"gflops = "<<gflops<<endl;

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
    
    // Verify result
    int i;
    for (i = 0; i < N; ++i) {
        float sum = h_A[i];
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }

        // Free device memory
    if (d_A)
        cudaFree(d_A);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_C)
        free(h_C);
        
    cudaDeviceReset();

	if(i == N)
		cout<<"SUCCSESS"<<endl;
	else 
		cout<<"FAILED"<<endl;   
}
//////////////////////////////////////////////////////
int main(int argc,char *argv[])
{ 
  if(argc < 3)
     printf("Unsuffcient number of arguments!\n");
else
	{
		VectorAddition(atoi(argv[1]), atoi(argv[2]));
	}
}
