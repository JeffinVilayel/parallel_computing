
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;
__global__ void
quamsim(const float *U, const float *A, float *B, const int *Q, int number_elem)
{    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int base_memory_idx ;
    __shared__ float A_shared[64];
    int memory_idx;
    int Q_bit_vector=0;
    int Q_bit_mask ;
    if(threadIdx.x==0){
        base_memory_idx = blockIdx.x;
        for(int it=0;it<6;it++)
           base_memory_idx = ((base_memory_idx >> Q[it])<<(Q[it]+1)) | ((1<< Q[it])-1)& base_memory_idx;     }
 if(threadIdx.x <16){
    __syncthreads();
    Q_bit_mask = threadIdx.x;
    for(int it=0;it<4;it++){
      if((Q_bit_mask & 1)== 1)
       Q_bit_vector =  Q_bit_vector | 1<<Q[it];
      else
       Q_bit_vector =  Q_bit_vector & ~(1<<Q[it]);  
      Q_bit_mask = Q_bit_mask >>1;    }
    memory_idx = base_memory_idx | Q_bit_vector; 
    A_shared[threadIdx.x & ~(1<<5) & ~(1<<4)] = A[memory_idx & ~(1<<Q[5]) & ~(1<<Q[4]) ]; 
    A_shared[threadIdx.x & ~(1<<5) |  (1<<4)] = A[memory_idx & ~(1<<Q[5]) |  (1<<Q[4]) ];
    A_shared[threadIdx.x |  (1<<5) & ~(1<<4)] = A[memory_idx |  (1<<Q[5]) & ~(1<<Q[4]) ]; 
    A_shared[threadIdx.x |  (1<<5) |  (1<<4)] = A[memory_idx |  (1<<Q[5]) |  (1<<Q[4]) ];
    __syncthreads();
    if (i < number_elem)
    {        for(int it=0;it<6;it++){          
           int index1 = (((2*threadIdx.x) >> it)<<(it+1)) | ((1<<it)-1)& (2*threadIdx.x);
           int index2 = (((2*threadIdx.x+1) >> it)<<(it+1)) | ((1<<it)-1)& (2*threadIdx.x+1);   
           float temp1 =A_shared[index1] ;
           float temp2 =A_shared[index2] ;
           A_shared[index1]= (U[it*4]*A_shared[index1]) + (U[it*4+1]*A_shared[index1 ^ (1<<it)]);
           A_shared[index1 ^ (1<<it)] =  (U[it*4+2]*temp1) + (U[it*4+3]*A_shared[index1 ^ (1<<it)]);
           A_shared[index2]= (U[it*4]*A_shared[index2]) + (U[it*4+1]*A_shared[index2 ^ (1<<it)]);
           A_shared[index2 ^ (1<<it)] =  (U[it*4+2]*temp2) + (U[it*4+3]*A_shared[index2 ^ (1<<it)]);
         __syncthreads();        }
        B[memory_idx & ~(1<<Q[5]) & ~(1<<Q[4]) ]  =A_shared[threadIdx.x & ~(1<<5) & ~(1<<4)] ; 
        B[memory_idx & ~(1<<Q[5]) |  (1<<Q[4]) ]  =A_shared[threadIdx.x & ~(1<<5) |  (1<<4)]  ;
        B[memory_idx |  (1<<Q[5]) & ~(1<<Q[4]) ]  =A_shared[threadIdx.x |  (1<<5) & ~(1<<4)]  ; 
        B[memory_idx |  (1<<Q[5]) |  (1<<Q[4]) ]  =A_shared[threadIdx.x |  (1<<5) |  (1<<4)]  ;           } }}
void CallKernelFunction(float *d_U,float *d_A,float *d_B,int *d_Q,int threadsPerBlock, int blocksPerGrid,int number_elem);
int main(int argc, char** argv)
{    cudaError_t err = cudaSuccess;
    vector<float> A_number;
    float number;
    int count=0;
    float h_U[24];
    size_t size ;
    int number_elem,q_bit[6]; 
    FILE *myfile;
    myfile = fopen(argv[1],"r"); 
    while(fscanf(myfile, "%f", &number) == 1) { 
      A_number.push_back(number);
      count++;    }    
    number_elem = A_number.size()-30;
    size = number_elem*sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL)
    {        exit(EXIT_FAILURE);    }
    for(int i=0;i<A_number.size()-6;i++)
      if(i<24) h_U[i]=A_number[i];
      else h_A[i-24]=A_number[i];

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
    err = cudaMalloc((void **)&d_U, 24*sizeof(float));
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    int *d_Q = NULL;
    err = cudaMalloc((void **)&d_Q, 6*sizeof(int));
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaMemcpy(d_Q,q_bit, 6*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaMemcpy(d_U, h_U, 24*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    int threadsPerBlock = 32;
    int blocksPerGrid =(number_elem>>1 + threadsPerBlock - 1) / threadsPerBlock;
    CallKernelFunction(d_U,d_A,d_B,d_Q,threadsPerBlock,blocksPerGrid,number_elem);    
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    for (int i = 0; i < number_elem; ++i)
     printf("%0.3f\n",h_B[i]);
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_U);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    err = cudaFree(d_Q);
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    free(h_A);
    free(h_B);
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    }
    return 0;}

void CallKernelFunction(float *d_U,float *d_A,float *d_B,int *d_Q,int threadsPerBlock, int blocksPerGrid,int number_elem){
    cudaError_t err;    
    quamsim<<<blocksPerGrid, threadsPerBlock>>>(d_U,d_A,d_B,d_Q, number_elem>>1);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {        exit(EXIT_FAILURE);    } }

