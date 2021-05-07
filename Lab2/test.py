import os
import sys
import pandas as pd

target = sys.argv[1]
# order n [8, 9, 10, 11, 12, 13, 14, 15]
order_n = [i for i in range(8, 16)]

# clear log
os.system("> exp.log")
for n in order_n:
    os.system('./{} {} >> exp.log'.format(target, n))

results = []
f = open('exp.log')
for line in f:
    data = line.split()
    if len(data) == 8:
        results.append(float(data[3]))
f.close()

print('Results:', results)
