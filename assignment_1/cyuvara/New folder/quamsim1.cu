#include <stdio.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void
quamsim(const float *U, const float *A, float *B, int number_elem,int q)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int quad_i = i^(1<<q);

    if (i < number_elem)
    {if((i& 1<<q )==0){
          B[i] =  (U[0]*A[i]) + (U[1]*A[quad_i]);
          B[quad_i] =  (U[2]*A[i]) + (U[3]*A[quad_i]);   }
    }}

void CallKernelFunction(float *d_U,float *h_U, float *d_A,float *d_B,int threadsPerBlock, int blocksPerGrid,int number_elem,int quadbit,int itert);
int main(int argc, char** argv)
{
    cudaError_t err = cudaSuccess;
    vector<float> A_number;
    float num;
    int count=0;
    float h1_U[4],h2_U[4],h3_U[4],h4_U[4],h5_U[4],h6_U[4];
    size_t size ;
    int number_elem,q_bit[6]; 
    FILE *myfile;
    myfile = fopen(argv[1],"r"); 
    while(fscanf(myfile, "%f", &num) == 1) {
      A_number.push_back(num);
      count++;    }   
    number_elem = A_number.size()-30;
    size = number_elem*sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL)
    {   fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);    }
    for(int i=0;i<A_number.size()-6;i++)
      if(i<4) h1_U[i]=A_number[i];
      else if(i<8)  h2_U[i-4]  = A_number[i];
      else if(i<12) h3_U[i-8]  = A_number[i];
      else if(i<16) h4_U[i-12] = A_number[i];
      else if(i<20) h5_U[i-16] = A_number[i];
      else if(i<24) h6_U[i-20] = A_number[i];
      else          h_A[i-24]  = A_number[i];

    q_bit[0] = A_number[A_number.size()-6];
    q_bit[1] = A_number[A_number.size()-5];
    q_bit[2] = A_number[A_number.size()-4];
    q_bit[3] = A_number[A_number.size()-3];
    q_bit[4] = A_number[A_number.size()-2];
    q_bit[5] = A_number[A_number.size()-1];
   
    
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_U = NULL;
    err = cudaMalloc((void **)&d_U, 4*sizeof(float));
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_B1 = NULL;
    err = cudaMalloc((void **)&d_B1, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_B2 = NULL;
    err = cudaMalloc((void **)&d_B2, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_B3 = NULL;
    err = cudaMalloc((void **)&d_B3, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_B4 = NULL;
    err = cudaMalloc((void **)&d_B4, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_B5 = NULL;
    err = cudaMalloc((void **)&d_B5, size);
    if (err != cudaSuccess)
    {       exit(EXIT_FAILURE);    }
    float *d_B6 = NULL;
    err = cudaMalloc((void **)&d_B6, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    int threadsPerBlock = 32;
    int blocksPerGrid =(number_elem + threadsPerBlock - 1) / threadsPerBlock;
    int itert =1;
    CallKernelFunction(d_U,h1_U,d_A, d_B1,threadsPerBlock,blocksPerGrid,number_elem,q_bit[0],itert++);
    CallKernelFunction(d_U,h2_U,d_B1,d_B2,threadsPerBlock,blocksPerGrid,number_elem,q_bit[1],itert++);
    CallKernelFunction(d_U,h3_U,d_B2,d_B3,threadsPerBlock,blocksPerGrid,number_elem,q_bit[2],itert++);
    CallKernelFunction(d_U,h4_U,d_B3,d_B4,threadsPerBlock,blocksPerGrid,number_elem,q_bit[3],itert++);
    CallKernelFunction(d_U,h5_U,d_B4,d_B5,threadsPerBlock,blocksPerGrid,number_elem,q_bit[4],itert++);
    CallKernelFunction(d_U,h6_U,d_B5,d_B6,threadsPerBlock,blocksPerGrid,number_elem,q_bit[5],itert++);
    
    
    err = cudaMemcpy(h_B, d_B6, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {       exit(EXIT_FAILURE);    }
    for (int i = 0; i < number_elem; ++i)
    printf("%.3f\n",h_B[i]);
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B1);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B2);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B3);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B4);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B5);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B6);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }         
    
    free(h_A);
    free(h_B);
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    return 0;
}

void CallKernelFunction(float *d_U,float *h_U, float *d_A,float *d_B,int threadsPerBlock, int blocksPerGrid,int number_elem,int quadbit,int itert){
    cudaError_t err;
    err = cudaMemcpy(d_U, h_U, 4*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    quamsim<<<blocksPerGrid, threadsPerBlock>>>(d_U,d_A, d_B, number_elem,quadbit);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    } }

