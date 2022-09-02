#!/usr/bin/env python3

from numba import jit
import numpy as np
import time

y = int(input('Enter the size of the matrix : '))

start = time.time()

def go_slow(b):
    trace = 0.0
    for i in range(b.shape[0]):
        trace += np.tanh(b[i,i])

    return b + trace

x = np.arange(y**2).reshape(y,y)
go_slow(x)

stop = time.time()
print(f'Without JIT :: Elapsed Time to run the code is : {stop - start : .2f} seconds.')


###########################################33

start = time.time()

@jit(nopython=True)
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i,i])

    return a + trace

x = np.arange(y**2).reshape(y,y)
go_fast(x)

stop = time.time()
print(f'With JIT :: Elapsed Time to run the code is : {stop - start : .2f} seconds.')

