#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include <vector>
#include<sys/time.h>
using namespace std;

__global__ void quamsim(const float *U_matrix,const float *input_matrix, float *output_matrix,int num_of_elements, int q_bit)
{
	int threadid = blockDim.x*blockIdx.x + threadIdx.x;
	int q_id = threadid^(1<<q_bit);
	//printf("Threadid:%d, %0.3f\n",threadid,input_matrix[threadid]);
	if (threadid < num_of_elements)
	{
	if ((threadid& 1<< q_bit) ==0){
		output_matrix[threadid] = (U_matrix[0]*input_matrix[threadid]) + (U_matrix[1]*input_matrix[q_id]);
		output_matrix[q_id] = (U_matrix[2]*input_matrix[threadid]) + (U_matrix[3]*input_matrix[q_id]);
	
	}
	}
}


#if TYPE==1
int main(int argc , char** argv)
{
	float input_value;
	FILE *input_file;
	input_file = fopen(argv[1],"r");
	vector<float> input_vector;

	while(fscanf(input_file,"%f",&input_value)==1)
	{
	input_vector.push_back(input_value);
	}
	
	int num_of_elements;

	num_of_elements = input_vector.size();
	num_of_elements = num_of_elements -5;


	size_t length;
	length = num_of_elements*sizeof(float);

	float *h_input_matrix = (float *)malloc(length);
	float *h_output_matrix = (float *)malloc(length);
	float h_U_matrix[4];
//	printf("hello");

	int q_bit = input_vector[input_vector.size()-1];
	int k;
	for (k =0; k<input_vector.size()-1;k++)
	{
	if(k<4) h_U_matrix[k] = input_vector[k];
	else h_input_matrix[k-4] = input_vector[k];
	}

	cudaError_t err = cudaSuccess;

	float *d_input_matrix = NULL;
	float *d_output_matrix= NULL;
	float *d_U_matrix = NULL;

	err = cudaMalloc((void **)&d_input_matrix,length);
	
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device input vector  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMalloc((void **)&d_output_matrix,length);

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device output vector  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMalloc((void **)&d_U_matrix, 4*sizeof(float));

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device U vector  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_input_matrix,h_input_matrix, length, cudaMemcpyHostToDevice);
if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_U_matrix, h_U_matrix , 4*sizeof(float), cudaMemcpyHostToDevice);
if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector U from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


	int threadsPerBlock = 256;
	int blocksPerGrid =(num_of_elements + threadsPerBlock -1)/ threadsPerBlock;
	struct timeval start,end;

	gettimeofday(&start, NULL);

	quamsim<<<blocksPerGrid, threadsPerBlock>>>(d_U_matrix,d_input_matrix,d_output_matrix, num_of_elements,q_bit);
	err = cudaGetLastError();
	gettimeofday(&end, NULL);
if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int time_in_us = 1e6*(end.tv_sec -start.tv_sec) + (end.tv_usec -start.tv_usec);

	err= cudaMemcpy(h_output_matrix, d_output_matrix, length, cudaMemcpyDeviceToHost);

	for (int l =0 ; l< num_of_elements;l++)
		printf("%0.3f\n",h_output_matrix[l]);

	free(h_input_matrix);
	free(h_output_matrix);

	err= cudaFree(d_input_matrix);
	err= cudaFree(d_output_matrix);

	err = cudaDeviceReset();

	return 1;

}
#endif

#if TYPE==2
int main(int argc, char** argv)
{
	float input_value;
	FILE *input_file;
	input_file = fopen(argv[1],"r");
	vector<float> input_vector;

	while(fscanf(input_file,"%f",&input_value)==1)
	{
	input_vector.push_back(input_value);
	}
	
	int num_of_elements;

	num_of_elements = input_vector.size();
	num_of_elements = num_of_elements -5;


	size_t length;
	length = num_of_elements*sizeof(float);

//	float *h_input_matrix = (float *)malloc(length);
//	float *h_output_matrix = (float *)malloc(length);
//	float h_U_matrix[4];
//	printf("hello");
	
	float *h_input_matrix,*h_output_matrix,*h_U_matrix;

	cudaMallocManaged(&h_U_matrix,4*sizeof(float));
	cudaMallocManaged(&h_input_matrix,length);
	cudaMallocManaged(&h_output_matrix,length);


	int q_bit = input_vector[input_vector.size()-1];
	int k;
	for (k =0; k<input_vector.size()-1;k++)
	{
	if(k<4) h_U_matrix[k] = input_vector[k];
	else h_input_matrix[k-4] = input_vector[k];
	}

	cudaError_t err = cudaSuccess;

	

	int threadsPerBlock = 256;
	int blocksPerGrid =(num_of_elements + threadsPerBlock -1)/ threadsPerBlock;
	struct timeval start,end;

	gettimeofday(&start, NULL);

	quamsim<<<blocksPerGrid, threadsPerBlock>>>(h_U_matrix,h_input_matrix,h_output_matrix, num_of_elements,q_bit);
	err = cudaGetLastError();
	gettimeofday(&end, NULL);
if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int time_in_us = 1e6*(end.tv_sec -start.tv_sec) + (end.tv_usec -start.tv_usec);
	cudaDeviceSynchronize();

	for (int l =0 ; l< num_of_elements;l++)
		printf("%0.3f\n",h_output_matrix[l]);


	err= cudaFree(h_input_matrix);
	 if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err= cudaFree(h_output_matrix);
	 if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaFree(h_U_matrix);
 if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector U (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaDeviceReset();
	//printf("heelllloo");
	return 1;



}
#endif
