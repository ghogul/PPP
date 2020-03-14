import numpy as np
import matplotlib.pyplot as plt



def residual_error():
    inter_filename = 'interpolated_simulation_data.txt' 
    inter_stress = np.loadtxt(inter_filename,usecols=(0))
    inter_strain = np.loadtxt(inter_filename,usecols=(1))
    #print(stress,strain)
    exp_filename = 'noise_data.txt'
    exp_stress_1 = np.loadtxt(exp_filename,usecols=(0))
    exp_strain_1 = np.loadtxt(exp_filename,usecols=(1))
    exp_stress = exp_stress_1[:len(inter_stress)]
    exp_strain = exp_strain_1[:len(inter_strain)]


    #residual_error = (np.linalg.norm(np.sum((exp_stress - inter_stress))))/np.sum(exp_stress)
    #residual_error = (np.sum(np.divide((exp_stress-inter_stress),exp_stress,out=np.zeros_like(exp_stress),where=exp_stress!=0)**2))/len(inter_strain)
    #print(residual_error)
    # plt.plot(exp_strain,exp_stress)
    # plt.plot(inter_strain,inter_stress)
    # plt.show()
    residual_error = np.square(np.subtract(exp_stress,inter_stress)).mean()
    return residual_error
    

