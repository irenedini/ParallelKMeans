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
void sequentialKMeans(int N, int K, int EPS, float*devData);

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

int main(int argc, char* argv[]){
	std::clock_t start;
	double duration;

	int iter = 0;
	float *devData, *hostData, *mean;
	bool stopCriterion = false;

	short *clusters;
	float oldMeanX, oldMeanY;

	hostData = (float*) malloc(2*N*sizeof(float));
	mean = (float*) malloc(2*K*sizeof(float));
	clusters = (short*) malloc(N*sizeof(short));


	CUDA_CALL(cudaMalloc((float **)&devData, 2*N*sizeof(float)));
	init(devData);
	CUDA_CHECK_RETURN(cudaMemcpy(hostData, devData, 2*N*sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	memcpy(mean, hostData, 2*K*sizeof(float));

	start = std::clock();
	while (!stopCriterion) {
		stopCriterion = true;
		/*
		 * distanze e minimo
		 */
		for (int v = 0; v < N; v++) {
			float minDistance = 3.0;
			short minIndex = -1;
			float distance = 0;
			for (int c = 0; c < K; c++) {
				distance = pow((hostData[2*v] - mean[2*c]),2)+pow((hostData[2*v+1] - mean[2*c+1]),2);

				if (distance < minDistance) {
					minIndex = c;
					minDistance = distance;
				}

			}
			clusters[v] = minIndex;
		}
		/*
		 * nuova media
		 */
		for (int i = 0; i < K; i++) {
			int numComponents = 0;
			float *arraySum = (float*)calloc(2, sizeof(float));
			for (int j = 0; j < N; j++) {
				if (clusters[j] == i) {
					numComponents++;
					arraySum[0] += hostData[2*j];
					arraySum[1] += hostData[2*j+1];

				}
			}
			oldMeanX = mean[2*i];
			mean[2*i] = arraySum[0] / numComponents;
			oldMeanY = mean[2*i+1];
			mean[2*i+1] = arraySum[1] / numComponents;
			if(abs(mean[2*i] - oldMeanX) > EPS || abs(mean[2*i+1] - oldMeanY) > EPS){
				stopCriterion = false;
			}
		}

		iter++;
	}
	printf("Numero di iterazioni: %d\r\n", iter);
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"durata: "<< duration <<'\n';

	free(hostData);
	free(mean);
	free(clusters);

	CUDA_CHECK_RETURN(cudaFree(devData));

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

