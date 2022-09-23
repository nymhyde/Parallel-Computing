// Transposing a Matrix using CUDA

#include <stdio.h>
#include "gputimer.h"

// matrix size : N x N
const int N = 3;


// verify :: check the two matrices if they match or not
int compare_matrices(float *gpu, float *ref, int N)
{
    int result = 1;
    for (int j=0; j < N; j++)
        for(int i=0; i < N; i++)
            if (ref[i+j*N] != gpu[i+j*N])
                {result = 0;}

    return result;
}


// fill a Matrix with sequential numbers in the range 0 -> N-1
void fill_matrix(float *mat, int N)
{
    for(int j=0; j < N*N; j++)
        mat[j] = (float) j;

    printf("Original Matrix :: \n");
    /* Display the matrix */
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f\t", mat[j+i*N]);
        }
        printf("\n");
    }
    printf("\n");
}


// transpose matrix :: using CPU
void transpose_CPU(float in[], float out[])
{
    for(int j=0; j < N; j++)
        for (int i=0; i < N; i++)
            out[j+i*N] = in[i+j*N];     // implements flip out(j,i) = in(i,j)

    printf("Transposed Matrix :: \n");
    /* Display the matrix */
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f\t", out[j+i*N]);
        }
        printf("\n");
    }
    printf("\n");
}


// launched on a single thread
__global__ void transpose_serial(float in[], float out[])
{
    for (int j=0; j < N; j++)
        for(int i=0; i < N; i++)
            out[j+i*N] = in[i+j*N];
}

// launched on one thread per Row
__global__ void transpose_parallel_per_row(float in[], float out[])
{
    int i = threadIdx.x + blockIdx.y * blockDim.x;

    for (int j=0; j < N; j++)
        out[j+i*N] = in[i+j*N];
}
    



int main(int argc, char **argv)
{
    int numbytes = N * N * sizeof(float);   // total bytes to store matrix values
    float *in = (float *) malloc(numbytes);
    float *out = (float *) malloc(numbytes);
    float *gold = (float *) malloc(numbytes);

    fill_matrix(in, N);
    transpose_CPU(in, gold);


    // initialize variable pointers
    float *d_in, *d_out;
    // allocate memory
    cudaMalloc(&d_in, numbytes);
    cudaMalloc(&d_out, numbytes);
    // copy input matrix memory to GPU
    cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

    // GPU Timer
    GpuTimer timer;

    // -----------------
    // GPU : Serialized
    // -----------------

    timer.Start();
    // Run GPU Kernel
    transpose_serial<<<1,1>>>(d_in, d_out);
    timer.Stop();
    // clean out
    for (int i=0; i < N*N; i++){out[i]=0.0;}
    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    // time it
    printf("Transpose Method :: Serial = %g ms \n", timer.Elapsed());
    printf("Verifying ... %s\n", compare_matrices(out, gold, N) ? "Success" : "Failed");

    // -----------------
    // Parallel Per Row
    // -----------------

    // clean d_out
    cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice);

    timer.Start();
    // Run GPU Kernel
    transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
    timer.Stop();
    // clean out
    for (int i=0; i < N*N; i++){out[i]=0.0;}
    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    // time it
    printf("Transpose Method :: Parallel Per Row = %g ms \n", timer.Elapsed());
    printf("Verifying ... %s\n", compare_matrices(out, gold, N) ? "Success" : "Failed");


    
    


    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    // free memory
    cudaFree(d_in);
    cudaFree(d_out);

    
}
