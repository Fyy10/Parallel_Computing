import os

target = 'Lab1'
num_proc = [2 ** i for i in range(8)]
prime_n = [100000 * (10 ** i) for i in range(5)]

# clear log
os.system("> exp.log")
for n in prime_n:
    for p in num_proc:
        os.system('mpirun -np {} {} {} >> exp.log'.format(p, target, n))
