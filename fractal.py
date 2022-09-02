#!/usr/bin/env python3

# << imports >> #
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from PIL import Image
import time



img_size = int(input('Enter Image Size : '))

# << Pure Python Run >> #

def mandel(x,y, max_iters):
    c = complex(x, y)
    z = 0.0j

    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_y

        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y,x] = color


image = np.zeros((img_size, img_size), dtype=np.uint8)

start = time.time()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
py_run = end - start
print(f'Pure Python Run :: Total Elapsed Time to run the program : {py_run:.2f} seconds')


# << Just-In-Time Compiler >> #

@jit(nopython=True)
def mandel(x,y, max_iters):
    c = complex(x, y)
    z = 0.0j

    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_y

        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y,x] = color

image = np.zeros((img_size, img_size), dtype=np.uint8)

# < First JIT run : where it compiles the decorated function into machine code >
start = time.time()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
just_compile = end - start
print(f'First JIT run :: Total Elapsed Time to run the program : {just_compile:.2f} seconds')

# < Pre-compile run : here the decorated functions are already converted to machine code >
start = time.time()
create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
pre_compile = end - start
print(f'Pre-Compiled JIT run :: Total Elapsed Time to run the program : {pre_compile:.2f} seconds')


print(f'\n\nFirst JIT is faster than Pure Python run by a factor of : {py_run/just_compile}')
print(f'Pre-Compiled JIT is faster than Pure Python run by a factor of : {py_run/pre_compile}')
print(f'Pre-Compiled JIT is faster than first JIT run by a factor of : {just_compile/pre_compile}')


# << Plotting Fractal Image >> #
plt.imshow(image)
plt.show()
