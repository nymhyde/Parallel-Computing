#!/usr/bin/env python3

import multiprocessing as mp
import time

print(f'This system has {mp.cpu_count()} number of CPUs')

def print_fun(continent):
    print(f'The name of the continent is : {continent}')


if __name__=='__main__':
    continents = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Australia']

    procs = []

    for continent in continents:
        proc = mp.Process(target=print_fun, args=(continent,))
        procs.append(proc)

        # Start the appended Process
        proc.start()
        print(f'Starting {proc}')


    # Waiting for proc to finish
    for proc in procs:
        proc.join()

