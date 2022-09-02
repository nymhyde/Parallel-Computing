#!/usr/bin/env python3

from numba import jit
import numpy as np
import time

x = int(input('Enter the size of the Matrix :: '))

# << Without JIT compiler >> #
def go_slow(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i,i])
    return a + trace

y = np.arange(x**2).reshape(x,x)
start = time.time()
go_slow(y)
stop = time.time()

print(f'Without JIT :: Total elapsed time to run the code : {stop-start:.2f} seconds')


# << With JIT compiler >> #
@jit(nopython=True)
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i,i])
    return a + trace

y = np.arange(x**2).reshape(x,x)
start = time.time()
go_fast(y)
stop = time.time()

print(f'Pre-Compiled :: Total elapsed time to run the code : {stop-start:.2f} seconds.')

start = time.time()
go_fast(y)
stop = time.time()

print(f'Post-Copiled :: Total elapsed time to run the code : {stop-start:.2f} seconds.')
