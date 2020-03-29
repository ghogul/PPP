import numpy as np
import matplotlib.pyplot as plt



def residual_error():
    # 
    # Reading the files to calculate the mean squared error
    # 
    inter_filename = 'interpolated_simulation_data.txt' 
    inter_stress   = np.loadtxt(inter_filename,usecols=(0))
    inter_strain   = np.loadtxt(inter_filename,usecols=(1))
    
    exp_filename = 'noise_data.txt'
    exp_stress_1 = np.loadtxt(exp_filename,usecols=(0))
    exp_strain_1 = np.loadtxt(exp_filename,usecols=(1))
    exp_stress   = exp_stress_1[:len(inter_stress)]
    exp_strain   = exp_strain_1[:len(inter_strain)]
    
    #  
    # calculating the mean squared error 
    # 
    residual_error = np.square(np.subtract(exp_stress,inter_stress)).mean()
    return residual_error
    

