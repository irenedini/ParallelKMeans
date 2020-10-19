#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <ctime>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);


#define N 80000
#define K 200
#define EPS 0.000001

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("Error at %s:%d\n",__FILE__,__LINE__);return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)
	
using namespace std;


int init(float *vectorsDev){
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 234ULL));
	CURAND_CALL(curandGenerateUniform(gen, vectorsDev, 2*N));
	CURAND_CALL(curandDestroyGenerator(gen));
	return 0;
}


__global__ void initClustersArray(int* clustersDev){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<K){
		clustersDev[i]=i;
	}
}

__global__ void calculateDistances(float* vectorsDev, float* meansDev, float* distDev){

	__shared__ float meansShared[2*1024];
	int tx = blockDim.x*blockIdx.x+threadIdx.x;
	int cluster = threadIdx.x;
	if(cluster<K){
		meansShared[2*cluster] = meansDev[2*cluster];
		meansShared[2*cluster+1] = meansDev[2*cluster+1];
	}
	__syncthreads();
	if(tx<N){
		float v1 = vectorsDev[2*tx];
		float v2 = vectorsDev[2*tx+1];
		for(int c = 0; c<K; c++){
			distDev[K*tx+c] = (v1-meansShared[2*c])*(v1-meansShared[2*c]) + (v2-meansShared[2*c+1])*(v2-meansShared[2*c+1]);
		}
	}
}


__global__ void assignmentStep(float* distDev, int* clustersDev){

	int index = blockIdx.x*blockDim.x+threadIdx.x;
	float minDist = 3;
	float dist = 3;
	int cluster = 0;

	if(index<N){
		for(int i = index*K; i<(index+1)*K; i++){
			dist = distDev[i];
			if(dist<minDist){
				minDist = dist;
				cluster = i-(index*K);
			}
		}
		clustersDev[index] = cluster;
	}
}

__global__ void sumStep(int* clustersDev, float* sumComponentsDev, int *numComponentsDev, float* vectorsDev){
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	if(tx<N){
		int c = clustersDev[tx];
		atomicAdd(&(sumComponentsDev[2*c+ty]), vectorsDev[2*tx+ty]);
		atomicAdd(&(numComponentsDev[c]), 1);
	}

}

__global__ void divideStep(float* sumComponentsDev, int* numComponentsDev, float* meansDev, int* changedDev){
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	if(tx<K){
		float oldMean = meansDev[2*tx+ty];
		float newMean = sumComponentsDev[2*tx+ty]/(numComponentsDev[tx]/2);
		meansDev[2*tx+ty] = newMean;
		if(abs(oldMean-newMean)>EPS){
			atomicAdd(&(changedDev[0]), 1);
		}
	}
}


int main(int argc, char* argv[]){
	std::clock_t start;
	double duration;

	int iter = 0;
	float *vectorsDev, *meansDev, *distDev, *sumComponentsDev, *zeroArray;
	int *clustersDev, *changedDev, *numComponentsDev;
	int changedHost = 1;

	zeroArray = (float *)calloc(2*K, sizeof(float));


	CUDA_CALL(cudaMalloc((float **)&vectorsDev, 2*N*sizeof(float)));
	CUDA_CALL(cudaMalloc((int **)&clustersDev, N*sizeof(int)));
	CUDA_CALL(cudaMalloc((float **)&meansDev, 2*K*sizeof(float)));
	CUDA_CALL(cudaMalloc((float **)&distDev, N*K*sizeof(float)));
	CUDA_CALL(cudaMalloc((int **)&changedDev, sizeof(int)));
	CUDA_CALL(cudaMalloc((float **)&sumComponentsDev, 2*K*sizeof(float)));
	CUDA_CALL(cudaMalloc((int **)&numComponentsDev, K*sizeof(int)));


	init(vectorsDev);

	dim3 block1(K,1,1);
	dim3 grid1(ceil(K/block1.x),1,1);

	dim3 block3(1024,1,1);
	dim3 grid3(ceil(N/block3.x)+1,1,1);

	dim3 block4(1024,1,1);
	dim3 grid4(ceil(N/block4.x)+1,1,1);

	dim3 block5(512,2,1);
	dim3 grid5(ceil(N/block5.x)+1,1,1);

	dim3 block6(512,2,1);
	dim3 grid6(ceil(K/block6.x)+1,1,1);

	initClustersArray<<<grid1, block1>>>(clustersDev);

	CUDA_CHECK_RETURN(cudaMemcpy(meansDev, vectorsDev, 2*K*sizeof(float), cudaMemcpyDeviceToDevice));

	start = std::clock();

	while(changedHost>0){

		iter++;

		//reset values
		CUDA_CHECK_RETURN(cudaMemset(changedDev, 0, sizeof(int)));
		CUDA_CHECK_RETURN(cudaMemcpy(sumComponentsDev, zeroArray, 2*K*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemset(numComponentsDev, 0, K*sizeof(int)));

		//compute
		calculateDistances<<<grid4,block4>>>(vectorsDev, meansDev, distDev);
		cudaDeviceSynchronize();
		assignmentStep<<<grid3,block3>>>(distDev, clustersDev);
		cudaDeviceSynchronize();
		sumStep<<<grid5,block5>>>(clustersDev, sumComponentsDev, numComponentsDev, vectorsDev);
		cudaDeviceSynchronize();
		divideStep<<<grid6, block6>>>(sumComponentsDev, numComponentsDev, meansDev, changedDev);
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMemcpy(&changedHost, changedDev, sizeof(int), cudaMemcpyDeviceToHost));
	}

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"durata: "<< duration <<'\n';

	printf("numero iterazioni = %d\n", iter);

	free(vectorsHost);
	free(clustersHost);
	free(zeroArray);

	CUDA_CHECK_RETURN(cudaFree(vectorsDev));
	CUDA_CHECK_RETURN(cudaFree(clustersDev));
	CUDA_CHECK_RETURN(cudaFree(meansDev));
	CUDA_CHECK_RETURN(cudaFree(distDev));
	CUDA_CHECK_RETURN(cudaFree(changedDev));

	return EXIT_SUCCESS;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

