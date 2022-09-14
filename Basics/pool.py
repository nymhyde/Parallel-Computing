#!/usr/bin/env python3

import multiprocessing as mp
from multiprocessing import Pool
import time

start = time.time()

# Print No of CPUs available on the system
print(f'{mp.cpu_count()} of CPUs')


work = (['A', 6], ['B', 4], ['C', 1], ['D', 3], ['E', 10], ['F', 3], ['G', 5], ['X',2],
        ['Y', 8], ['Z',1])

def work_log(work_data):
    print(f'Process {work_data[0]} waiting for {work_data[1]} seconds.')
    time.sleep(int(work_data[1]))
    print(f'Process {work_data[0]} finished.')


def pool_handler():
    p = Pool(10)
    p.map(work_log, work)


pool_handler()


stop = time.time()

print(f'Elapsed Time to Run the program : {stop-start:.2f} seconds')
