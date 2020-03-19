import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy.special import expit
#import tqdm
from ppp_isotropic_nonlinear_hardening import *
from interpolation import *
from residual_error import *
from training_testing import *

start_time = time.time() 
# '''
# ============================================================================================
#                             voce hardening function
# ============================================================================================
# '''
# def voce_fun(x,epsilon):
#     sigma_0 = x[0]
#     sigma_1 = x[1]
#     sigma_2 = x[2]
#     n       = x[3]
#     sigma_yield = sigma_0 + (sigma_1*epsilon) + (sigma_2*(1-np.exp(-n*epsilon)))
#     return sigma_yield
# # voce = voce_fun(x_0,epsilon)
# # print(voce)
# # print(epsilon)
# # plt.plot(epsilon,voce)
# # plt.plot(epsilon,stress)
# # plt.show()
# '''============================================================================================'''

'''
============================================================================================
                            jacobian function
============================================================================================
'''
def jacobian(x,epsilon):
    scaling_parameter = np.array([600,300,300,40])
    sigma_0 = x[0]
    sigma_1 = x[1]
    sigma_2 = x[2]
    n       = x[3]
    no_of_data = len(epsilon)
    no_of_para = len(x)
    a = np.exp(-n*epsilon)
    b = 1 - a
    partial_sigma_0 = np.ones(no_of_data)*scaling_parameter[0]
    partial_sigma_1 = epsilon*scaling_parameter[1]
    partial_sigma_2 = b*scaling_parameter[2]
    partial_n       = sigma_2*epsilon*a*scaling_parameter[3]
    jacobian = np.array([partial_sigma_0,partial_sigma_1,partial_sigma_2,partial_n])
    return jacobian
'''============================================================================================'''

'''
============================================================================================
                                    quadratic function
============================================================================================
'''

def quadratic_model(delta_x,f_0,gradient_0,hessian_0,xs=None):
    xs = np.sqrt(xs/xs.min())
    delta_x = delta_x/xs
    m_1 = f_0 + np.dot(gradient_0,delta_x)+0.5*np.dot(delta_x,np.dot(hessian_0,delta_x))
    return m_1
'''============================================================================================'''



'''
============================================================================================
                                    levenberg marquardt
============================================================================================
'''

def levenberg_marquardt(x,youngs_modulus,exp_stress,exp_strain,sim_stress,sim_strain,levenberg_residual_error,lamda = 0.001, f_tol = 1, eta = 0.25, max_iteration = 100, lamda_iteration = 10):
    no_of_data = len(exp_strain)
    no_of_para = len(x)
    print('='*80)
    print('\t\t\t\tLevenberg algorithm')
    xs = x

    standard_deviation = np.std(exp_stress)
    x_0 = x
    residual_0 = (sim_stress-exp_stress)/standard_deviation
    #print('residual_0',residual_0)
    f_0 = 0.5*(np.linalg.norm(residual_0)**2)
    jacobian_0 = jacobian(x_0,exp_strain)/standard_deviation 
    gradient_0 = np.inner(jacobian_0,residual_0)
    norm_gradient_0 = np.linalg.norm(gradient_0)
    
    i = 1
    j = 1
    delta_f = 1
    loop = 1
    while ((abs(delta_f) > f_tol) and (i <= max_iteration) and (j <= lamda_iteration)):
        
        #print('f_tol',f_tol)
        print('='*80)
        #print('while loop levenberg',i,'iteration')
        square_root_lamda = np.sqrt(lamda)
        square_root_lamda_identity = square_root_lamda*np.eye(no_of_para)

        jacobian_square_root_lamda_identity = np.hstack([jacobian_0,square_root_lamda_identity])

        z = np.zeros(no_of_para) 
        residual_z = np.hstack([residual_0,z])

        (delta_x,residual,rank,sv) = np.linalg.lstsq(jacobian_square_root_lamda_identity.T,-(residual_z), rcond=-1)
        #print('delta_x',delta_x)
        norm_delta_x = np.linalg.norm(delta_x)
        #print('delta_x',delta_x)
        
        #print('youngs_modulus',youngs_modulus)
        x_1 = x_0 + delta_x
        #np.longdouble(x_1)
        #print('x_1',x_1)
        scaling_parameter = np.array([600,300,300,40])
        fem_parameter = np.array([x_1[0]*scaling_parameter[0],x_1[1]*scaling_parameter[1],x_1[2]*scaling_parameter[2],x_1[3]*scaling_parameter[3]])
        sim_stress,sim_strain = nonlinear_fem(youngs_modulus,x_1[0]*scaling_parameter[0],x_1[1]*scaling_parameter[1],x_1[2]*scaling_parameter[2],(x_1[3]*scaling_parameter[3]))

        # print("Youngs modulus", youngs_modulus)
        # print("Yield stress",fem_parameter[0])
        # print("Linear Hardening modulus",fem_parameter[1])
        # print("Non linear hardening modulus", fem_parameter[2])
        # print("Hardening exponent", fem_parameter[3])

        with open('simulation_data.txt', 'w') as f:
            f.write('0')
            f.write('\t\t0\n')
            for s,st in zip(sim_stress,sim_strain):
                print(s,st,file = f)

        interpolation()
        filename = 'interpolated_simulation_data.txt'
        sim_stress_1 = np.loadtxt(filename,usecols=(0))
        sim_strain_1 = np.loadtxt(filename,usecols=(1))
        #exp_stress = exp_stress[:len(sim_stress)]
        sim_stress = np.array([])
        sim_strain = np.array([])
        for i in range(len(sim_stress_1)):
            if sim_strain_1[i] >= exp_strain[0]:
                sim_stress = np.append(sim_stress,sim_stress_1[i])
                sim_strain = np.append(sim_strain,sim_strain_1[i])
        
        levenberg_residual_error[loop] = residual_error()
        
        residual_1 = (sim_stress[:len(exp_stress)] - exp_stress)/standard_deviation
        #print('residual_1',residual_1)
        f_1 = 0.5*(np.linalg.norm(residual_1)**2)
        jacobian_1 = jacobian(x_1,exp_strain)/standard_deviation
        gradient_1 = np.inner(jacobian_1,residual_1)
        norm_gradient = np.linalg.norm(gradient_1)

        gradient_0 = np.inner(jacobian_0,residual_0)
        #print('gradient_0',gradient_0)
        hessian_0 = np.inner(jacobian_0,jacobian_0)
        #print('hessian_0',hessian_0)
        m_0 = f_0
        m_1 = quadratic_model(delta_x,f_0,gradient_0,hessian_0,xs=xs)

        a = (f_0 - f_1)/(m_0 - m_1)
        delta_f = f_1 - f_0
        #print('delta_f',delta_f)

        print('='*80)
        print('\t\t\t\tloop',loop)
        print(".."*20)
        # print('iteration',i,'lamda iteration',j)
        # print('lamda',lamda)
        # print('x_0',x_0)
        # print('f_0',f_0)
        # print('m_0',m_0)
        # print('.'*50)
        # print('x_1',x_1)
        print('Algorithm computed parameters',fem_parameter)
        print('x_0',x_0)
        print('scaling parameter',scaling_parameter)
        print(".."*20)
        print("Youngs modulus", youngs_modulus)
        print("Yield stress",fem_parameter[0])
        print("Linear Hardening modulus",fem_parameter[1])
        print("Non linear hardening modulus", fem_parameter[2])
        print("Hardening exponent", fem_parameter[3])
        # print('f_1',f_1)
        # print('m_1',m_1)
        # print('.'*50)
        print('a',a)
        # print('.'*50)
        print('delta_x',delta_x)
        # print('norm_delta_x',norm_delta_x)
        print('delta_f',delta_f)
        # print('norm_gradient',norm_gradient)
        print('='*80)
        print("\n")

       


        if a <= 0.25:
            lamda = lamda*10
            #print('lamda is increased to = ',lamda)

        else:
            if a > 0.75:
                lamda = lamda/10
                #print('lamda is decreased to = ',lamda)
            # else:
            #     print('keeping same lamda',lamda)

        if a > eta:
            print('step accepted')
            x_0 = x_1
            residual_0 = residual_1
            f_0 = f_1
            jacobian_0 = jacobian_1 
            j = 1
            i += 1

        else:
            print('step rejected')
            delta_f = 1
            j += 1

        if i > max_iteration:
            print('maximum number of iteration is reached')
        if j > lamda_iteration:
            print('maximum number of lamda iteration is reached')

        

        
        loop = loop +1
        
        
    print('x_opt',x_1)
    print('yield stress',x_1[0]*scaling_parameter[0],'hardening mod',x_1[1]*scaling_parameter[1],'delta_y',x_1[2]*scaling_parameter[2],'eta',x_1[3]*scaling_parameter[3])
    print('='*80)
     
    return fem_parameter,youngs_modulus


'''============================================================================================'''


if __name__=="__main__":
    print("="*150)
    print("="*150)
    print("\t\t\t\tParameter identification of an elasto-plastic materials using FEM program")
    print("="*150)
    print("="*150)
    print("Initial guess of the parameters")
    print("."*20)

    print('E,yield,hardening modulus,delta_y,eta')
    youngs_modulus = 200000
    yield_stress = 800
    hardening_modulus = 500
    delta_y = 500
    eta = 20
    print("Youngs modulus", youngs_modulus)
    print("Yield stress", yield_stress)
    print("Linear Hardening modulus",hardening_modulus)
    print("Non linear hardening modulus", delta_y)
    print("Hardening exponent", eta)
    print("\n")



    exp_file = 'noise_data.txt'
    exp_stress = np.loadtxt(exp_file,usecols=(0))
    exp_strain = np.loadtxt(exp_file,usecols=(1))

    sim_stress ,sim_strain = nonlinear_fem(youngs_modulus,yield_stress,hardening_modulus,delta_y,eta)
    
    with open('simulation_data.txt', 'w') as f:
        f.write('0')
        f.write('\t\t0\n')
        for s,st in zip(sim_stress,sim_strain):
            print(s,st,file = f)

    interpolation()
    before_residual_error = residual_error()
    
    print('residual_error',before_residual_error)
    fig,ax = plt.subplots(2,constrained_layout=True)
    ax[0].scatter(exp_strain,exp_stress,color='blue',marker='.', label='Experimental data')
    ax[0].plot(sim_strain,sim_stress,color='red',label='Intial simulation data')
    ax[0].legend()
    ax[0].grid()
    

    exp_file = 'noise_data.txt'
    sim_file = 'interpolated_simulation_data.txt'
    exp_stress_1 = np.loadtxt(exp_file,usecols=(0))
    exp_strain_1 = np.loadtxt(exp_file,usecols=(1))
    sim_stress_1 = np.loadtxt(sim_file,usecols=(0))
    sim_strain_1 = np.loadtxt(sim_file,usecols=(1))
    
    youngs_modulus = exp_stress_1[3]/exp_strain_1[3]
    exp_stress = np.array([])
    exp_strain = np.array([])
    for i in range(len(exp_stress_1)):
        if exp_stress_1[i] >= 500:
            exp_stress = np.append(exp_stress,exp_stress_1[i])
            exp_strain = np.append(exp_strain,exp_strain_1[i])

    sim_stress = np.array([])
    sim_strain = np.array([])
    for i in range(len(sim_stress_1)):
        if sim_strain_1[i] >= exp_strain[0]:
            sim_stress = np.append(sim_stress,sim_stress_1[i])
            sim_strain = np.append(sim_strain,sim_strain_1[i])
    
   
    

    if before_residual_error > 1e-3:
        
        x = np.array([0.8,0.7,0.8,0.4], dtype = np.longdouble)
        levenberg_residual_error = np.zeros((300))
        levenberg,youngs_modulus = levenberg_marquardt(x,youngs_modulus,exp_stress[:len(sim_stress)],exp_strain[:len(sim_strain)],sim_stress,sim_strain, levenberg_residual_error, lamda = 0.001, f_tol = 1e-3, eta = 0.25, max_iteration = 200, lamda_iteration = 20)
        print('parameters',levenberg)
        print('E',youngs_modulus)
        sim_stress ,sim_strain = nonlinear_fem(youngs_modulus,levenberg[0],levenberg[1],levenberg[2],levenberg[3])
    
        with open('simulation_data.txt', 'w') as f:
            f.write('0')
            f.write('\t\t0\n')
            for s,st in zip(sim_stress,sim_strain):
                print(s,st,file = f)

        interpolation()
        after_residual_error = residual_error()
        yield_stress = levenberg[0]
        hardening_modulus = levenberg[1]
        delta_y = levenberg[2]
        eta = levenberg[3]
        print('last residual_error',after_residual_error)
        ax[1].scatter(exp_strain_1,exp_stress_1,color = 'blue',marker = '.', label = 'Experimental data')
        ax[1].plot(sim_strain,sim_stress,color = 'red', label = 'Final simulation data')
        ax[1].legend()
        ax[1].grid()
        ax[1].set_title('Predicted data and True value')
        ax[1].set_xlabel('Strain')
        ax[1].set_ylabel('Stress')
        ax[0].set_title('Inital guess and True value')
        ax[0].set_xlabel('Strain')
        ax[0].set_ylabel('Stress')
        fig.savefig('Initial guess plot and Predicted plot')
        plt.close()
        

    training_testing()

    end_time = time.time()
    levenberg_residual_error = levenberg_residual_error[np.nonzero(levenberg_residual_error)] 
    print('residual_error',levenberg_residual_error)
    x_mse = range(len(levenberg_residual_error)) 
    plt.plot(x_mse,levenberg_residual_error, color = 'blue')
    plt.title('Mean squared error vs Number of iterations ')
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean squared error')
    # plt.legend()
    plt.grid()
    plt.savefig('mse')
    plt.close()
    
    print('time in minutes',(end_time-start_time)/60)