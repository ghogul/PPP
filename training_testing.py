import numpy as np
import matplotlib.pyplot as plt



def training_testing():
    # 
    #  Reading the data from the experiment and FEM 
    # 

    sim_data = 'interpolated_simulation_data.txt'
    exp_data = 'noise_data.txt'

    exp_stress = np.loadtxt(exp_data,usecols=(0))
    exp_strain = np.loadtxt(exp_data,usecols=(1))
    sim_stress = np.loadtxt(sim_data,usecols=(0))
    sim_strain = np.loadtxt(sim_data,usecols=(1))

    # 
    # taking out the equal number of data in experiment and simulation
    # 
    exp_stress = exp_stress[:len(sim_stress)]   
    exp_strain = exp_strain[:len(sim_strain)]

    exp_stress_strain = np.array([exp_stress,exp_strain]).T

    shuffle_stress_strain = np.array([exp_stress,exp_strain]).T
    # 
    #  shuffling the data set 
    # 
    np.random.shuffle(shuffle_stress_strain)

    # 
    #  seperating the data set of training and testing
    # 

    training = shuffle_stress_strain[:int(len(sim_stress)*0.6)]
    testing = shuffle_stress_strain[int(len(sim_stress)*0.6):]
    
    # 
    #  Plotting the simulation curve fits with training and testing data
    # 
    

    fig,ax = plt.subplots(2,constrained_layout=True)
    ax[0].plot(sim_strain,sim_stress,color='blue',label='Predicted data')
    ax[0].scatter(training[:,1],training[:,0],color='red',marker='.',label='Training data')
    ax[1].plot(sim_strain,sim_stress,color='blue',label='Predicted data')
    ax[1].scatter(testing[:,1],testing[:,0],color='red',marker='.',label='Testing data')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    ax[0].set_title('Training ')
    ax[0].set_xlabel('Strain')
    ax[0].set_ylabel('Stress')
    ax[1].set_title('Testing')
    ax[1].set_xlabel('Strain')
    ax[1].set_ylabel('Stress')
    fig.savefig('Training_Testing')
    plt.close()
    return None


