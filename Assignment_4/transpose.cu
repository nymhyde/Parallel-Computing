// Transposing a Matrix using CUDA

#include <stdio.h>
#include "gputimer.h"


// scale factor
const int scale = 1;
// matrix size : N x N
const int N = 1024*scale*scale;
// constansts
const int TILE_DIM = 32*scale;
const int BLOCK_ROWS = TILE_DIM/4;


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
}


// transpose matrix :: using CPU
void transpose_CPU(float in[], float out[])
{
    for(int j=0; j < N; j++)
        for (int i=0; i < N; i++)
            out[j+i*N] = in[i+j*N];     // implements flip out(j,i) = in(i,j)
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
    // int i = threadIdx.x + blockIdx.y * blockDim.x;
    int i = threadIdx.x;

    for (int j=0; j < N; j++)
        out[j+i*N] = in[i+j*N];
}
   

// launched in tile fashion non-shared memory
    // doesn't use shared memory
    // Global memory reads are coalesced but writes are not
__global__ void transpose_tiled(float in[], float out[])
{
    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (int j=0; j < TILE_DIM; j+= BLOCK_ROWS)
        out[x*N + (y+j)] = in[(y+j)*N + x];
}


// launched in tile fashion shared memory
    // Uses shared memory to achieve coalesign in both reads and writes
__global__ void transpose_tiled_shared(float in[], float out[])
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = threadIdx.x + blockIdx.x * TILE_DIM;
    int y = threadIdx.y + blockIdx.y * TILE_DIM;

    for (int j=0; j < TILE_DIM; j+= BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*N + x];

    // like the name suggests - sync threads 
    __syncthreads();

    x = threadIdx.x + blockIdx.y * TILE_DIM;
    y = threadIdx.y + blockIdx.x * TILE_DIM;

    for (int j=0; j < TILE_DIM; j+= BLOCK_ROWS)
        out[(y+j)*N + x] = tile[threadIdx.x][threadIdx.y + j];
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
    // time it
    printf("Transpose Method :: Serial = %g ms \n", timer.Elapsed());
    // clean out
    for (int i=0; i < N*N; i++){out[i]=0.0;}
    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
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
    // time it
    printf("Transpose Method :: Parallel Per Row = %g ms \n", timer.Elapsed());
    // clean out
    for (int i=0; i < N*N; i++){out[i]=0.0;}
    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    printf("Verifying ... %s\n", compare_matrices(out, gold, N) ? "Success" : "Failed");

    // -----------------------
    // Tiled 32x32 non-shared
    // -----------------------

    dim3 dimGrid(N/TILE_DIM, N/TILE_DIM, 1);
    dim3 dimBlock(N/TILE_DIM, BLOCK_ROWS, 1);

    // clean d_out
    cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice);

    timer.Start();
    //Run GPU Kernel
    transpose_tiled<<<dimGrid,dimBlock>>>(d_in, d_out);
    timer.Stop();
    // time it
    printf("Transpose Method :: Tiled Non-Shared Mem = %g ms \n", timer.Elapsed());
    // clean out
    for (int i=0; i < N*N; i++){out[i]=0.0;}
    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    printf("Verifying ... %s\n", compare_matrices(out, gold, N) ? "Success" : "Failed");
    

    // -------------------
    // Tiled 32x32 shared
    // -------------------

    // dim3 dimGrid(32, 32, 1);
    // dim3 dimBlock(32, 8, 1);

    // clean d_out
    cudaMemcpy(d_out, d_in, numbytes, cudaMemcpyDeviceToDevice);

    timer.Start();
    //Run GPU Kernel
    transpose_tiled_shared<<<dimGrid,dimBlock>>>(d_in, d_out);
    timer.Stop();
    // time it
    printf("Transpose Method :: Tiled Shared Mem = %g ms \n", timer.Elapsed());
    // clean out
    for (int i=0; i < N*N; i++){out[i]=0.0;}
    // copy output matrix memory back to Host
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    printf("Verifying ... %s\n", compare_matrices(out, gold, N) ? "Success" : "Failed");

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);

}
