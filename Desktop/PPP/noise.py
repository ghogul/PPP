import numpy as np
import matplotlib.pyplot as plt

filename = 'STRESS STRAIN.txt'
stress = np.loadtxt(filename,usecols=(0))
strain = np.loadtxt(filename,usecols=(1))


noise = np.random.normal(scale=5,size=len(stress))

stress = stress + noise

with open('noise_data.txt', 'w') as f:
    f.write('0')
    f.write('\t\t0\n')
    for s,st in zip(stress,strain):
        print(s,st,file = f)


# plt.plot(strain,stress)
# plt.show()