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
ap.add_argument('-k', '--kernel_size', type=int, default=7,
        help='size of the kernel to be applied n x n')
args = vars(ap.parse_args())

k_size = args['kernel_size']

print(f'Using a Kernel of size {k_size} x {k_size}')

# << Sequential Compute >>

def blurfilter(in_img, out_img, k_size):
    '''
    For each pixel in in_imag calculate teh mean intensity values
    using square 7x7 stencil
    '''

    for channel in range(3):
        for x in range(in_img.shape[1]):
            for y in range(in_img.shape[0]):
                val=0
                for i in range(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in range(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                               val += (int(img[y+j, x+i, channel]))

                out_img[y,x,channel] = val // 49

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

# << JIT + Parallel Compute >>
print(f'Executing JIT + Parallel Compute ... ...')

@jit(nopython=True, parallel=True)
def para_blurr(in_img, out_img, k_size):
    '''
    For each pixel in in_imag calculate teh mean intensity values
    using square 7x7 stencil
    '''

    for channel in nb.prange(3):
        for x in nb.prange(in_img.shape[1]):
            for y in nb.prange(in_img.shape[0]):
                val=0
                for i in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                    for j in nb.prange(-(k_size-1)//2,k_size//2 + 1 ):
                        if (x+i < img.shape[1]) and (x+i >= 0) and \
                           (y+j < img.shape[0]) and (y+j >= 0):
                               val += (int(img[y+j, x+i, channel]))

                out_img[y,x,channel] = val // 49


para_blurr(img, imgblur, k_size)

# timing execution
start = time.time()
para_blurr(img, imgblur, k_size)
stop = time.time()
para_time = stop - start
print(f'Time to run the Parallel Code <Box Filter> :: {para_time:.2f} seconds')



# --------------------------------------------------------

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


k_size = 15
gkernel = getKernel(k_size)



@jit(nopython=True, parallel=True)
def para_blurr(in_img, out_img, gkernel, k_size):
    '''
    For each pixel in in_imag calculate teh mean intensity values
    using square 7x7 stencil
    '''

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



para_blurr(img, imgblur, gkernel, k_size)

# timing execution
start = time.time()
para_blurr(img, imgblur, gkernel, k_size)
stop = time.time()
para_time = stop - start
print(f'Time to run the Parallel Code <Gaussian Filter> :: {para_time:.2f} seconds')




# << Display and Saving >>

fig = plt.figure()

ax = fig.add_subplot(1,3,1)
imgplot = plt.imshow(img)
ax.set_title('Before')

ax = fig.add_subplot(1,3,2)
imgplot = plt.imshow(imgblur)
ax.set_title('After : JIT + Parallel')

ax = fig.add_subplot(1,3,3)
imgplot = plt.imshow(st_blur)
ax.set_title('After : Stencil')

img2 = Image.fromarray(imgblur)
img2.save('blurred.jpg', quality=100)

st_blur = Image.fromarray(st_blur)
st_blur.save('stencil_blurred.jpg', quality=100)

plt.show()      # to show the images
