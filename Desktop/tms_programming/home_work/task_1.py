import numpy as np
import matplotlib.pyplot as plt


l=2e-6  #length
n_dislocations = 4
initial_position =l*np.array([0.1,0.13,0.16,0.3])


# calculate shear stress 
G=20e9
b=0.1e-9
B=1e-4
poissions_ratio=0.3
delta_t=10
shear_stress1=((G*b)/(2*np.pi*(1-poissions_ratio)))


def shear_stress(initial_position,G,poissions_ratio,b):
        for i,value in enumerate(initial_position):
                x_coord = np.delete(initial_position , i)
                shear_stress = ((G*b)/(2*np.pi*(1-poissions_ratio)))*(1/(value-x_coord))
                total_stress=np.sum([shear_stress])
                print(total_stress)
        return shear_stress

shear_stress(initial_position,26,0.33,0.256)
