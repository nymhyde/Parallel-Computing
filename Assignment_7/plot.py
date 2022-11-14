from decimal import *
import numpy as np
import matplotlib.pyplot as plt

getcontext().prec = 25

index = np.arange(1,13)
print(index)


# Defining the error (MPI)
error = np.array([0.0008333314113056,
                  0.0000083333333309,
                  0.0000000833333331,
                  0.0000000008333312,
                  0.0000000000083413,
                  0.0000000000001048,
                  0.0000000000000226,
                  0.0000000000001208,
                  0.0000000000000884,
                  0.0000000013865686,
                  0.0000000017674591,
                  0.0000000002392655])

print(error)


# Wall Time (MPI)
time_mpi = np.array([0.000046,
                     0.000042,
                     0.000056,
                     0.000160,
                     0.000143,
                     0.001248,
                     0.010694,
                     0.111374,
                     0.735355,
                     6.882624,
                     67.974126,
                     702.340895])

print(time_mpi)


# Wall Time (single process)

wall_single = np.array([0.000026,
                        0.000006,
                        0.000033,
                        0.000035,
                        0.000839,
                        0.003061,
                        0.056281,
                        0.306743,
                        3.078677,
                        32.805832,
                        307.079831,
                        3121.972127])

print(wall_single)

# Get Accuracy from Error
acc = []
for i in range(len(error)):
   acc.append(Decimal((1-error[i])))

accuracy = np.array(acc)

print(accuracy)

# Plots
#
# 1. Accuracy vs 10^T interval size

plt.figure(figsize=(8,4), tight_layout=True)
plt.plot(accuracy)
plt.show()

