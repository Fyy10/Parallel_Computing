# visualization
import matplotlib.pyplot as plt
import numpy as np

baseline = [
    [0.000503, 0.005505, 0.061954, 1.173121, 14.617282],
    [0.000313, 0.003197, 0.031962, 0.739045, 8.990941],
    [0.00025, 0.001516, 0.016518, 0.340224, 4.42432],
    [0.000329, 0.001183, 0.013422, 0.252747, 3.473149],
    [0.000383, 0.000852, 0.00719, 0.20962, 3.23075],
    [0.000592, 0.000876, 0.003825, 0.294871, 3.484592],
    [0.001284, 0.00211, 0.004428, 0.072544, 3.206113],
    [0.003033, 0.002252, 0.00726, 0.061647, 3.116729]
]

optim1 = [
    [0.000407, 0.002295, 0.032671, 0.562771, 7.161367],
    [0.000205, 0.001516, 0.016881, 0.328676, 4.295056],
    [0.000229, 0.000844, 0.014388, 0.113358, 2.185208],
    [0.000291, 0.000618, 0.004475, 0.074054, 1.69191],
    [0.000391, 0.000689, 0.004473, 0.06859, 1.589734],
    [0.000499, 0.000776, 0.002376, 0.02795, 1.577404],
    [0.001569, 0.001469, 0.003392, 0.023473, 1.552771],
    [0.003313, 0.028463, 0.006223, 0.034022, 1.386311]
]

optim2 = [
    [0.000273, 0.003135, 0.032848, 0.562548, 7.14584],
    [0.000121, 0.001432, 0.016779, 0.325757, 4.307968],
    [0.000073, 0.000679, 0.008206, 0.114637, 2.122966],
    [0.000086, 0.000541, 0.006688, 0.05524, 1.668108],
    [0.000063, 0.000334, 0.003512, 0.042336, 1.556801],
    [0.000038, 0.000132, 0.001825, 0.019778, 1.634999],
    [0.000157, 0.000385, 0.001728, 0.023221, 1.483369],
    [0.000732, 0.000049, 0.001287, 0.025568, 1.223046]
]

optim3 = [
    [0.000247, 0.002343, 0.035688, 0.309158, 3.035172],
    [0.000122, 0.001429, 0.016245, 0.173176, 1.504099],
    [0.000073, 0.000683, 0.007932, 0.087138, 0.750661],
    [0.000057, 0.000337, 0.007239, 0.052351, 0.436735],
    [0.000087, 0.000282, 0.003713, 0.038306, 0.240831],
    [0.00004, 0.000186, 0.001689, 0.019261, 0.204012],
    [0.001045, 0.001141, 0.001506, 0.008472, 0.155011],
    [0.000089, 0.000822, 0.000922, 0.004648, 0.153483]
]

baseline = np.array(baseline)
optim1 = np.array(optim1)
optim2 = np.array(optim2)
optim3 = np.array(optim3)

colors = ['r', 'g', 'b', 'c', 'm', 'y']
x = [2 ** i for i in range(8)]

# time-process 1e9
fig = plt.figure()
plt.ylabel('Execution Time (s)')
plt.xlabel('Num of Process (p)')
plt.semilogx(x, baseline[:, 4], base=2, marker='o', c=colors[0], label='baseline')
plt.semilogx(x, optim1[:, 4], base=2, marker='o', c=colors[1], label='optim1')
plt.semilogx(x, optim2[:, 4], base=2, marker='o', c=colors[2], label='optim2')
plt.semilogx(x, optim3[:, 4], base=2, marker='o', c=colors[3], label='optim3')
plt.legend()
plt.show()

# time-process with different n
fig = plt.figure()
plt.ylabel('Execution Time (s)')
plt.xlabel('Num of Process (p)')
for i in range(5):
    plt.semilogx(x, optim3[:, i], base=2, marker='o', c=colors[i], label=r'N = $10^{}$'.format(i+5))
plt.legend()
plt.show()

# speedup-process 1e9
fig = plt.figure()
plt.ylabel('Speedup')
plt.xlabel('Num of Process (p)')
plt.semilogx(x, baseline[0, 4] / baseline[:, 4], base=2, marker='o', c=colors[0], label='baseline')
plt.semilogx(x, optim1[0, 4] / optim1[:, 4], base=2, marker='o', c=colors[1], label='optim1')
plt.semilogx(x, optim2[0, 4] / optim2[:, 4], base=2, marker='o', c=colors[2], label='optim2')
plt.semilogx(x, optim3[0, 4] / optim3[:, 4], base=2, marker='o', c=colors[3], label='optim3')
plt.legend()
plt.show()

# speedup-process with different n
fig = plt.figure()
plt.ylabel('Speedup')
plt.xlabel('Num of Process (p)')
for i in range(5):
    plt.semilogx(x, optim3[0, i] / optim3[:, i], base=2, marker='o', c=colors[i], label=r'N = $10^{}$'.format(i+5))
plt.legend()
plt.show()
