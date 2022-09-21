#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

// geeneral strcuture to define complex numbers
struct cuComplex
{
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2(void) 
    { 
        return r * r + i * i; 
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};


// checking if the point belong to julia set
__device__ int julia(int x, int y)
{
    const float scale = 0.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++)
    {
        a  = a*a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}


// kernel that iterates through all the points we want to render
__global__ void kernel(unsigned char *ptr)
{
    // map from thredIdx / BlockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    
    int offset = x + y * gridDim.x;

    // now we calcualte the value at that position

    int juliavalue = julia(x,y);

    ptr[offset*4 + 0] = 255 * juliavalue;
    ptr[offset*4 + 1] = 50;
    ptr[offset*4 + 2] = 50;
    ptr[offset*4 + 3] = 150;

}


// main block
int main(void)
{
    CPUBitmap bitmap(DIM, DIM);

    // gpu implementation 
    unsigned char *dev_bitmap;

    cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() );

    dim3 grid(DIM, DIM);
    
    kernel<<<grid,1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    // clean up
    cudaFree(dev_bitmap);
}

