#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

// geeneral strcuture to define complex numbers
struct cuComplex
{
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b) {}
    float magnitude2(void) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};


// checking if the point belong to julia set
int julia(int x, int y)
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
void kernel(unsigned char ptr[])
{
    for(int y=0; y<DIM; y++)
    {
        for(int x=0; x<DIM; x++)
        {
            int offset = x + y * DIM;
            int juliavalue = julia(x,y);

            ptr[offset*4 + 0] = 255 * juliavalue;
            ptr[offset*4 + 1] = 50;
            ptr[offset*4 + 2] = 50;
            ptr[offset*4 + 3] = 150;

        }
    }
}



int main(void)
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();

    kernel(ptr);

    bitmap.display_and_exit();
}

