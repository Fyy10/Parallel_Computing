# visualization
import matplotlib.pyplot as plt
import numpy as np

order_n = [i for i in range(8, 16)]
order_n = np.array(order_n)

baseline = [1.331, 2.904, 5.943, 11.858, 11.999, 16.539, 23.837, 28.74]

optim = [6.409, 18.236, 37.052, 49.977, 55.08, 56.612, 56.579, 67.179]

baseline = np.array(baseline)
optim = np.array(optim)

# billion interactions per second
fig = plt.figure()
plt.ylabel('Billion Interactions Per Second')
plt.xlabel('N (nBodies = {})'.format(r'$2^{N+1}$'))
plt.plot(order_n, baseline, marker='o', c='c', label='parallel baseline')
plt.plot(order_n + 0.3, optim, marker='o', c='b', label='optimized')
plt.bar(order_n, baseline, color='c', width=0.3, label='parallel baseline')
plt.bar(order_n + 0.3, optim, color='b', width=0.3, label='optimized')
plt.legend()
plt.show()

# speedup
fig = plt.figure()
plt.ylabel('Speedup')
plt.xlabel('N (nBodies = {})'.format(r'$2^{N+1}$'))
plt.bar(order_n, optim / baseline, color='coral', width=0.8, label='Speedup')
plt.legend()
plt.show()
