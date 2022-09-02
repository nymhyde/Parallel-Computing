#!/usr/bin/env python3

import multiprocessing as mp
from multiprocessing import Pool
import time

work = (["A", 6], ["B", 4], ["C", 1], ["D", 3])

def work_log(work_data):
    print(f'Process {work_data[0]} waiting for {work_data[1]}')
    time.sleep(int(work_data[1]))
    print(f'Process {work_data[0]} is finished.')

def pool_handler():
    p = Pool(4)
    p.map(work_log, work)

if __name__ == "__main__":

    start = time.time()
    pool_handler()
    stop = time.time()

    print(f'Total elapsed time to run the code is : {stop-start:.2f} seconds.')