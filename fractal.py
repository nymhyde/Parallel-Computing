#!/usr/bin/env python3

# << imports >> #
import multiprocessing as mp
import numpy as np
import numba as nb
from numba import jit
import matplotlib.pyplot as plt
from PIL import Image
import time

print(f'Total Number of CPUs : {mp.cpu_count()}')

width = int(input('Enter Width for Image : '))
height = int(input('Enter Height for Image : '))

print(f'Total Number of Pixels :: {width*height}')

print(f'\n***** Sequential Run *****')

# << Sequential :: Pure Python Run >> #

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

def pure_python():
    create_fractal(-2.0, -1.7, -0.1, 0.1, image, 80)


image = np.zeros((height, width), dtype=np.uint8)

start = time.time()
pure_python()
end = time.time()
py_run = end - start
print(f'Pure Python (Sequential) Run :: Time to Execute :: {py_run:.5f} seconds')


# << Sequential Machine Code :: Just-In-Time Compiler >> #
print(f'\n***** JIT Machine Code Run *****')

@jit(nopython=True)
def jit_mandel(x,y, max_iters):
    c = complex(x, y)
    z = 0.0j

    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@jit(nopython=True)
def jit_create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_y

        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = jit_mandel(real, imag, iters)
            image[y,x] = color

image = np.zeros((height, width), dtype=np.uint8)

# < First JIT run : where it compiles the decorated function into machine code >
start = time.time()
jit_create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
just_compile = end - start
print(f'Non-compiled JIT run :: Time to Execute :: {just_compile:.5f} seconds')


# < Pre-compile run : here the decorated functions are already converted to machine code >
start = time.time()
jit_create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
pre_compile = end - start
print(f'Compiled JIT run :: Time to Execute :: {pre_compile:.5f} seconds')


print(f'\nNon-Compiled JIT >> Pure Python :: {py_run/just_compile:.2f}x Faster')
print(f'Pre-Compiled JIT >> Pure Python :: {py_run/pre_compile:.2f}x Faster')
print(f'Pre-Compiled JIT >> Non-Compiled JIT :: {just_compile/pre_compile:.2f}x Faster')

# ------------------------------------------
# << Just-In-Time Compiler + Parallelism >> #
# ------------------------------------------
print(f'\n\n***** JIT Machine Code + Parallelism Run *****')

@jit(nopython=True)
def para_mandel(x,y, max_iters):
    c = complex(x, y)
    z = 0.0j

    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@jit(nopython=True, parallel=True)
def para_create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in nb.prange(width):
        real = min_x + x * pixel_size_y

        for y in nb.prange(height):
            imag = min_y + y * pixel_size_y
            color = para_mandel(real, imag, iters)
            image[y,x] = color

image = np.zeros((height, width), dtype=np.uint8)

# < First JIT run : where it compiles the decorated function into machine code >
start = time.time()
para_create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
para_just_compile = end - start
print(f'Non-Compiled JIT Parallel run :: Time to Execute :: {para_just_compile:.5f} seconds')


# < Pre-compile run : here the decorated functions are already converted to machine code >
start = time.time()
para_create_fractal(-2.0, -1.7, -0.1, 0.1, image, 20)
end = time.time()
para_pre_compile = end - start
print(f'Compiled JIT Parallel run :: Time to Execute :: {para_pre_compile:.5f} seconds')

print(f'\nNon-Compiled JIT + Parallel >> Pure Python :: {py_run/para_just_compile:.2f}x Faster')
print(f'Non-Compiled JIT + Parallel >> Non-Compiled JIT :: {just_compile/para_just_compile:.2f}x Faster')
print(f'Compiled JIT + Parallel >> Pure Python :: {py_run/para_pre_compile:.2f}x Faster')
print(f'Compiled JIT + Parallel >> Compiled JIT :: {pre_compile/para_pre_compile:.2f}x Faster')





# ----------------------------------------------------
# << Just-In-Time Compiler + Parallelism on CUDA >> #
# ----------------------------------------------------
print(f'\n\n***** JIT Machine Code + Parallelism (CUDA) Run *****')

from numba import cuda
from numba import *

@cuda.jit('uint32(f8, f8, uint32)', device=True)
def cuda_mandel(x,y, max_iters):
    c = complex(x, y)
    z = 0.0j

    for i in nb.prange(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters


@cuda.jit('void(f8, f8, f8, f8, uint8[:,:], uint32)')
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y


    for x in range(width):
        real = min_x + x * pixel_size_y

        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = cuda_mandel(real, imag, iters)
            image[y,x] = color

gimage = np.zeros((height, width), dtype=np.uint8)
blockdim = (32, 8)
gridim = (32, 16)


# < First JIT run : where it compiles the decorated function into machine code >
start = time.time()
dimage = cuda.to_device(gimage)
mandel_kernel[gridim, blockdim](-2.0, -1.7, -0.1, 0.1, dimage, 20)
end = time.time()
cuda_just_compile = end - start
print(f'CUDA run :: Time to Execute :: {cuda_just_compile:.5f} seconds')


print(f'\nCUDA run >> Pure Python :: {py_run/cuda_just_compile:.2f}x Faster')
print(f'CUDA run >> JIT :: {pre_compile/cuda_just_compile:.2f}x Faster')
print(f'CUDA run >> JIT + Parallel :: {para_pre_compile/cuda_just_compile:.2f}x Faster \n')

'''
# << Plot the Image >>
plt.imshow(image)
plt.show()
'''
