import numpy as np
import matplotlib.pyplot as plt



def interpolation():
    # 
    # 
    #  Reading the files of expermental and simulation data 
    # 
    # 
    sim_file = "simulation_data.txt"
    exp_file = "noise_data.txt"
    sim_stress = np.loadtxt(sim_file,usecols=(0))
    sim_strain = np.loadtxt(sim_file,usecols=(1))
    exp_stress = np.loadtxt(exp_file,usecols=(0))
    exp_strain = np.loadtxt(exp_file,usecols=(1))
    exp_stress = exp_stress[:len(sim_stress)]
    exp_strain = exp_strain[:len(sim_strain)]

    # 
    #  Warnings if the guess is too large then interpolation is very difficult 
    # 
    
    if sim_strain[-1] < exp_strain[-1]:
        print(".!"*30)
        print("Guess is too large!!!")
        print("Please guess the parameters which is close to the real solution.")
        print(".!"*30)
        exit()
       
    else:
        # 
        # here  interpolating the data between expermental and simulation data and writing in to the file  
        # 

    
        with open('interpolated_simulation_data.txt', 'w') as f:
            f.write('0')
            f.write('\t0\n')
            for i in range(len(sim_strain-1)):
                for j in range(len(sim_strain-1)):
                    if (sim_strain[j] < exp_strain[i]) and sim_strain[j+1] > exp_strain[i]:
                        
                        sigma = sim_stress[j] + (((sim_stress[j+1] - sim_stress[j])/(sim_strain[j+1] - sim_strain[j]))*(exp_strain[i]-sim_strain[j]))
                        f.write(str(sigma))
                        f.write('\t')
                        f.write(str(exp_strain[i]))
                        f.write('\n')
    return None



