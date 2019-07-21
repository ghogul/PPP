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
total_time=50e-9
time_step=1000
external_stress=-25e6


def shear_stress(initial_position,G,poissions_ratio,b):
        internal_stress=([])
        for i,value in enumerate(initial_position):
                x_coord = np.delete(initial_position , i)
                shear_stress = ((G*b)/(2*np.pi*(1-poissions_ratio)))*(1/(value-x_coord))
                total_stress=np.sum([shear_stress])
                internal_stress=np.append(internal_stress,total_stress)
        return internal_stress
        
velocity=([])
final=([])
for i in range(time_step):
        velocity1=((b/B))*((shear_stress(initial_position,G,poissions_ratio,b)+external_stress))
        final_position=initial_position+((total_time/time_step)*velocity1)
        zeros=np.zeros(4)
        final_position=np.maximum(final_position,zeros)
        initial_position=final_position
        velocity=np.append(velocity,velocity1)
        final=np.append(final,final_position)
        
final=final.reshape(time_step,4)

y=np.linspace(0,time_step,time_step)
def plot_trajectories(final,y,velocity):
        plt.plot(final,y)
        plt.show()
plot_trajectories(final,y,velocity)

