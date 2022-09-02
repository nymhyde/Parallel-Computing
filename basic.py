#!/usr/bin/env python3

import multiprocessing as mp

# Print No of CPUs available on the system
print(f'This system has {mp.cpu_count()} numbers of CPUs')

def print_fun(continent):
    print(f'Starting Process : {continent}')
    print(f'The name of the continent is : {continent}')

continents = ['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Australia']

procs = []

for i, continent in enumerate(continents):
    # Define the process with its target fun and function argument
    proc = mp.Process(target=print_fun, args=(continent,))

    # Append this initialized process to a list
    procs.append(proc)
    # print(procs)

    # Start the initialized process
    proc.start()


# Tell to Complete the processes and then execute the next code
# Otherwise, future code will be executed before the process are finished.
for proc in procs:
    proc.join()

print(f'\n\nAll processes are finished')

########################################3
