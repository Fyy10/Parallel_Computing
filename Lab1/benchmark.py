import sys
import os

num_proc = sys.argv[1]
target = sys.argv[2]
prime_n = sys.argv[3]

os.system('> tmp')
for i in range(10):
    os.system('mpirun -np {} {} {} >> tmp'.format(num_proc, target, prime_n))

avg_time = 0.0
f = open('tmp')
for line in f:
    data = line.split()
    if len(data) == 3:
        avg_time += float(data[2])
f.close()
os.system('rm tmp')

avg_time /= 10
print('Average execution time:', avg_time)
