import numpy as np
import matplotlib.pyplot as plt

filename = "data.txt"
data = np.loadtxt(filename, comments = '%')

print(data.shape)

x = np.unique(data[:,0])
y = np.unique(data[:,1])

all_x_spacing = np.diff(x)
all_y_spacing = np.diff(y)

x_spacing = all_x_spacing[0]
y_spacing = all_y_spacing[0]

print(x_spacing)
print(y_spacing)

xx,yy = np.meshgrid(x,y)


for row in data:
    x_values = row[0] - x_spacing/2
    y_values = row[1] - y_spacing/2
    
U = data[:,2].reshape(xx.shape)
V = data[:,3].reshape(xx.shape)

fig, (ax0,ax1) = plt.subplots(nrows = 2, figsize = (15,7))
cf = ax0.contourf(xx,yy,U, cmap = 'RdBu')
plt.colorbar(cf, ax = ax0)
cf = ax1.contourf(xx,yy,V,cmap = 'RdBu')
plt.colorbar(cf, ax = ax1)
plt.show()
    

