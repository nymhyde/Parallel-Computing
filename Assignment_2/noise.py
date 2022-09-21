#!/usr/bin/env python3

# << imports >>

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from numba import jit
import numba as nb
import time
import argparse


# << kernel_size variable as parsed argument >>
ap = argparse.ArgumentParser()
ap.add_argument('-k', '--kernel_size', type=int, default=5,
        help='size of the kernel to be applied n x n')
args = vars(ap.parse_args())

k_size = args['kernel_size']

print(f'Using a Kernel of size {k_size} x {k_size}')

# Let's define the gaussian kernel

def getKernel(k_size):
    sigma = ((k_size-1)*0.5 - 1) + 0.8
    center = k_size // 2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = (1 / (2*np.pi*sigma)) * (np.exp(-(np.square(x) + np.square(y)) / (2*np.square(sigma))))

    return g / np.sum(g)


# Get Gaussian Kernel
gkernel = getKernel(k_size)



# << Sequential Compute :: Box Blur >>

print(f'Executing Sequential Run ... ...')

def blurfilter(in_img, out_img, k_size):

    for channel in range(3):
        for x in range(in_img.shape[1]):
            for y in range(in_img.shape[0]):
                val=0
                for i in range(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in range(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                               val += (int(img[y+j, x+i, channel]))

                out_img[y,x,channel] = val // (k_size**2)

img = np.array(Image.open('noisy1.jpg'))
print(f'Image shape is : {img.shape}')

imgblur = img.copy()

# timing execution
start = time.time()
blurfilter(img, imgblur, k_size)
stop = time.time()
seq_time = stop - start

print(f'Time to run the Sequential Code <Box Filter> :: {seq_time:.2f} seconds')

# --------------------------------------------------------

# << Sequential Compute :: Gaussian Blur >>

def seq_gaussfilter(in_img, out_img, k_size):

    for channel in nb.prange(3):
        for x in nb.prange(in_img.shape[1]):
            for y in nb.prange(in_img.shape[0]):
                val=0
                for i in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                            val += img[y+j, x+i, channel] * gkernel[j,i]

                out_img[y,x,channel] = val


img = np.array(Image.open('noisy1.jpg'))
seq_gauss_imgblur = img.copy()

# timing execution
start = time.time()
seq_gaussfilter(img, seq_gauss_imgblur, k_size)
stop = time.time()
seq_gauss_time = stop - start

print(f'Time to run the Sequential Code <Gaussian Filter> :: {seq_gauss_time:.2f} seconds')


# --------------------------------------------------------

# << JIT Sequential Compute >>

@jit(nopython=True)
def jit_blurfilter(in_img, out_img, k_size):

    for channel in range(3):
        for x in range(in_img.shape[1]):
            for y in range(in_img.shape[0]):
                val=0
                for i in range(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in range(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                               val += (int(img[y+j, x+i, channel]))

                out_img[y,x,channel] = val // (k_size**2)


img = np.array(Image.open('noisy1.jpg'))
jit_imgblur = img.copy()
jit_blurfilter(img, jit_imgblur, k_size)

# timing execution
start = time.time()
jit_blurfilter(img, jit_imgblur, k_size)
stop = time.time()
jit_time = stop - start

print(f'Time to run the Sequential JIT Code <Box Filter> :: {jit_time:.2f} seconds')


# --------------------------------------------------------

# << JIT + Parallel Compute >>
print(f'Executing JIT + Parallel Compute ... ...')

@jit(nopython=True, parallel=True)
def para_blurr(in_img, out_img, k_size):

    for channel in nb.prange(3):
        for x in nb.prange(in_img.shape[1]):
            for y in nb.prange(in_img.shape[0]):
                val=0
                for i in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                               val += (int(img[y+j, x+i, channel]))

                out_img[y,x,channel] = val // (k_size**2)

img = np.array(Image.open('noisy1.jpg'))
para_imgblur = img.copy()
para_blurr(img, para_imgblur, k_size)

# timing execution
start = time.time()
para_blurr(img, para_imgblur, k_size)
stop = time.time()
para_time = stop - start
print(f'Time to run the Parallel Code <Box Filter> :: {para_time:.2f} seconds')



# --------------------------------------------------------

'''
# << Stencil + Parallel Compute >>
# << With Box Filter :: Slower Approach >>
print(f'Executing Stencil Compute ... ...')

from numba import stencil

@stencil(neighborhood=((-3,3), (-3,3), (0,0)))
def stencil_blurr(M):
    for c in range(3):
        output=0
        for i in range(-3,4):
            for j in range(-3,4):
                output += int(M[i, j, 0])

    return output // 49

@jit(nopython=True, parallel=True)
def run_stencil(M):
    out_img = stencil_blurr(M)
    return out_img


st_blur = run_stencil(img)

start = time.time()
st_blur = run_stencil(img)
stop = time.time()
stencil_time = stop - start
print(f'Time to run the Stencil Code :: {stencil_time:.2f} seconds')

st_blur = st_blur.astype(np.uint8)

'''




# --------------------------------------------------------

# << JIT + Parallel Compute + Gaussian Filter >>
print(f'Executing JIT + Parallel Compute + Gaussian Filter ... ...')

# Let's define the gaussian kernel

def getKernel(k_size):
    sigma = ((k_size-1)*0.5 - 1) + 0.8
    center = k_size // 2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = (1 / (2*np.pi*sigma)) * (np.exp(-(np.square(x) + np.square(y)) / (2*np.square(sigma))))

    return g / np.sum(g)

# Get Gaussian Kernel
gkernel = getKernel(k_size)


@jit(nopython=True, parallel=True)
def para_Gaussblurr(in_img, out_img, gkernel, k_size):

    for channel in nb.prange(3):
        for x in nb.prange(in_img.shape[1]):
            for y in nb.prange(in_img.shape[0]):
                val=0
                for i in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                            val += img[y+j, x+i, channel] * gkernel[j,i]

                out_img[y,x,channel] = val



img = np.array(Image.open('noisy1.jpg'))
gauss_imgblur = img.copy()
para_Gaussblurr(img, gauss_imgblur, gkernel, k_size)

# timing execution
start = time.time()
para_Gaussblurr(img, gauss_imgblur, gkernel, k_size)
stop = time.time()
gauss_time = stop - start
print(f'Time to run the Parallel Code <Gaussian Filter> :: {gauss_time:.2f} seconds')



# << Display and Saving >>

fig = plt.figure()

ax = fig.add_subplot(1,5,1)
imgplot = plt.imshow(img)
ax.set_title('Before')

ax = fig.add_subplot(1,5,2)
imgplot = plt.imshow(imgblur)
ax.set_title('After : Python (BOX Blur)')

ax = fig.add_subplot(1,5,3)
imgplot = plt.imshow(jit_imgblur)
ax.set_title('After : JIT + Parallel (BOX Blur)')

ax = fig.add_subplot(1,5,4)
imgplot = plt.imshow(seq_gauss_imgblur)
ax.set_title('After : Python Gaussian Blur')

ax = fig.add_subplot(1,5,5)
imgplot = plt.imshow(gauss_imgblur)
ax.set_title('After : JIT + Parallel Gaussian Blur')

seq_gauss_blur = Image.fromarray(seq_gauss_imgblur)
seq_gauss_blur.save('seq_gauss_blurred.jpg', quality=100)

gauss_blur = Image.fromarray(gauss_imgblur)
gauss_blur.save('numba_gauss_blurred.jpg', quality=100)

jit_blur = Image.fromarray(jit_imgblur)
jit_blur.save('numba_box_blurred.jpg', quality=100)

plt.show()      # to show the images
