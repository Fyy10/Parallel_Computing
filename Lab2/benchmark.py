import sys
import os

target = sys.argv[1]
order_n = sys.argv[2]

os.system('> tmp')
for i in range(10):
    os.system('./{} {} >> tmp'.format(target, order_n))

# average billion interactions per second
avg_BIPS = 0.0
f = open('tmp')
for line in f:
    data = line.split()
    if len(data) == 8:
        avg_BIPS += float(data[3])
f.close()
os.system('rm tmp')

avg_BIPS /= 10
print('Average billion interactions per second ({} bodies):'.format(2 << int(order_n)), avg_BIPS)
