import numpy as np
import matplotlib.pyplot as plt


l=2e-6  #length
n_dislocations = 4
initial_position =l*np.array([0.1,0.13,0.16,0.3])


# calculate shear stress 
G=26e9
b=0.256e-9
B=1e-4
poissions_ratio=0.33
delta_t=0
shear_stress1=((G*b)/(2*np.pi*(1-poissions_ratio)))

velocity=b/B

def shear_stress(initial_position,G,poissions_ratio,b):
        internal_stress=([])
        for i,value in enumerate(initial_position):
                x_coord = np.delete(initial_position , i)
                shear_stress = ((G*b)/(2*np.pi*(1-poissions_ratio)))*(1/(value-x_coord))
                total_stress=np.sum([shear_stress])
                internal_stress=np.append(internal_stress,total_stress)
        return internal_stress
        

final=([])
for i in range(1000):
        final_position=initial_position+((1e-11)*(velocity*shear_stress(initial_position,G,poissions_ratio,b)))
        initial_position=final_position
        final=np.append(final,final_position)

final=final.reshape(1000,4)        
print(final)