import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt

from ppp_isotropic_nonlinear_hardening import *
from interpolation import *
from residual_error import *
from training_testing import *

start_time = time.time() 


'''
=================================================================================================================================================================================
                                                jacobian function
=================================================================================================================================================================================
'''
def jacobian(x,epsilon):
    '''
    =====================================================================
    jacobian of a function
    ---------------------------------------------------------------------
    partial_sigma_0 : partial derivative of first term
    partial_sigma_1 : partial derivative of second term
    partial_sigma_2 : partial derivative of third term
    partial_n : partial derivative of exponent term
    ---------------------------------------------------------------------
    return
    jacobian : array of partial deriavtives of the function
    =====================================================================
    '''
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
'''
=================================================================================================================================================================================
'''

'''
=================================================================================================================================================================================
                                                    quadratic function
=================================================================================================================================================================================
'''

def quadratic_model(delta_x,f_0,gradient_0,hessian_0):
    '''
    ======================================================================
    quadratic model
    Description : Levenberg Marquardt method can be described using trust region 
    framework. For spherical trust region it should be solved at each iteration.
    So we define a model function 
    ----------------------------------------------------------------------
    delta_x : step
    f_0 : function value f(delta_x=0)
    gradient_0 : gradient g(delta_x=0)
    hessian_0 : Hessian h(delta_x=0)
    ----------------------------------------------------------------------
    return
    m_1 : function f(delta_x)
    ======================================================================
    '''
    
    m_1 = f_0 + np.dot(gradient_0,delta_x)+0.5*np.dot(delta_x,np.dot(hessian_0,delta_x))
    return m_1
'''============================================================================================'''



'''
=================================================================================================================================================================================
                                        levenberg marquardt
=================================================================================================================================================================================
'''

def levenberg_marquardt(x,youngs_modulus,exp_stress,exp_strain,sim_stress,sim_strain,levenberg_residual_error,lamda, f_tol, eta, max_iteration, lamda_iteration):
    '''
    ======================================================================
    Levenberg-Marquardt method for solving 
    nonlinear least square problems
    ----------------------------------------------------------------------
    x : start parameter np.array([y1,...,yn])
    youngs_modulus : youngs odulus of the material
    jacobian : function for jacobian which returns np.array([[dri/dyj,...,drm/dyn)
    exp_stress : data points of experimental stress  np.array([y1,...,ym])
    exp_strain : data points of experimental strain  np.array([x1,...,xm])
    sim_stress : data points of stress genetrated by using FEM parameters are given by by user  np.array([y1,...,ym])
    sim_strain : data points of strain genetrated by using FEM parameters are given by by user  np.array([x1,...,xm])
    levenerg_residual_error : it stores the mean squared error of stress strain curve in each iteration
    lamda : initial value of lamda
    ftol : termination criterion
    eta : accuracy to accept step
    max_iteration : maximum_number of iterations
    lambda_itereration : maximum_number of lambda adjustions
    ----------------------------------------------------------------------
    return:
    FEM_parameters and youngs_modulus : algorithm computed parameters and youngs modulus
    FEM_parameters contains Yield stress, Linear hardening modulus, Non-linear hardening modulus and Hardening exponent
    ======================================================================
    '''
    no_of_data = len(exp_strain)  # Number of data points
    no_of_para = len(x)           # Number of parameters to identify
    print('='*80,file=f)
    print('\t\t\t\tLevenberg algorithm',file=f)
    
    # The performance of the nonlinear least square algorithms can be improved if the residuals
    # as well as the parameters are scaled. To scale or to normalize the residuals we use the
    # standard deviation of the experimental data

    standard_deviation = np.std(exp_stress)  # Standard deviation used for normalization
    x_0 = x
    residual_0 = (sim_stress-exp_stress)/standard_deviation         # compute Normalized residual
    f_0 = 0.5*(np.linalg.norm(residual_0)**2)                       # compute sum of squares
    jacobian_0 = jacobian(x_0,exp_strain)/standard_deviation        # compute normalized jacobian
    
    
    '''initilizing to start loop'''
    i = 1
    j = 1
    delta_f = 1
    loop = 1
    while ((abs(delta_f) > f_tol) and (i <= max_iteration) and (j <= lamda_iteration)):
        
        
        print('='*80,file=f)

        #
        # Try to compute a full step
        # By solving (jacobian.T*jacobian + lamda*identity)*delta_y = -jacobian.T*residual
        # Which is equivalent to solve the linear least square problem
        # min_x norm(([jacobian sqrt(lamda*identity)].T*delta_y) + [residual 0].T)**2
        #
        
        square_root_lamda = np.sqrt(lamda)
        square_root_lamda_identity = square_root_lamda*np.eye(no_of_para)

        jacobian_square_root_lamda_identity = np.hstack([jacobian_0,square_root_lamda_identity])      # Left hand side

        z = np.zeros(no_of_para) 
        residual_z = np.hstack([residual_0,z])       # Right hand side

        (delta_x,residual,rank,sv) = np.linalg.lstsq(jacobian_square_root_lamda_identity.T,-(residual_z), rcond=-1)

        #
        #
        # Warnings if delta_x is too small then the guess is wrong, guess is large. The program will exit 
        #
        #

        for i in range(len(delta_x)):
            if abs(delta_x[i]) < (1e-8):
                print(".!"*30)
                print("Guess is too large !!!")
                print("Please guess the parameters which is close to the real solution.")
                print(".!"*30)
                exit()

        
        #
        #
        # scaling parameter : here we normalized the parameters but while calling FEM program we have to scale the parameter to the real solution
        #
        #
        x_1 = x_0 + delta_x
        
        scaling_parameter = np.array([600,300,300,40])
        fem_parameter = np.array([x_1[0]*scaling_parameter[0],x_1[1]*scaling_parameter[1],x_1[2]*scaling_parameter[2],x_1[3]*scaling_parameter[3]])
        sim_stress,sim_strain = nonlinear_fem(youngs_modulus,x_1[0]*scaling_parameter[0],x_1[1]*scaling_parameter[1],x_1[2]*scaling_parameter[2],(x_1[3]*scaling_parameter[3]))

        # writing the output to the file 

        with open('simulation_data.txt', 'w') as a:
            a.write('0')
            a.write('\t\t0\n')
            for s,st in zip(sim_stress,sim_strain):
                print(s,st,file = a)
            a.close()

        # 
        # Again the simulation data and experimental data are not same do interpolate the FEM data respective to experimental data and writing to a file
        # 
        # 

        interpolation()
        filename = 'interpolated_simulation_data.txt'
        sim_stress_1 = np.loadtxt(filename,usecols=(0))
        sim_strain_1 = np.loadtxt(filename,usecols=(1))
        
        sim_stress = np.array([])
        sim_strain = np.array([])
        for i in range(len(sim_stress_1)):
            if sim_strain_1[i] >= exp_strain[0]:
                sim_stress = np.append(sim_stress,sim_stress_1[i])
                sim_strain = np.append(sim_strain,sim_strain_1[i])
        
        levenberg_residual_error[loop] = residual_error()  #  storing the mean squared error of each iteration
        
        #
        #  
        #    Update point, residuals, objective function, Hessian and Model 
        #
        #
        #
        
        
        
        residual_1 = (sim_stress[:len(exp_stress)] - exp_stress)/standard_deviation
        f_1 = 0.5*(np.linalg.norm(residual_1)**2)
        jacobian_1 = jacobian(x_1,exp_strain)/standard_deviation
        gradient_1 = np.inner(jacobian_1,residual_1)
        norm_gradient = np.linalg.norm(gradient_1)

        gradient_0 = np.inner(jacobian_0,residual_0)      # compute gradient
       
        hessian_0 = np.inner(jacobian_0,jacobian_0)       # Hessian matrix to solve quadratic model
       
        m_0 = f_0
        m_1 = quadratic_model(delta_x,f_0,gradient_0,hessian_0)        # caling quadratic model function


        #
        # Checking Accuracy
        #
        a = (f_0 - f_1)/(m_0 - m_1)  # it checks weather the step is accepted or not  and to adjust lamda
        delta_f = f_1 - f_0          # algorithm termination value
        
        # 
        # 
        #  Here it writes all the parameters at each iteration to see how the algorithm optimize the parameters
        # 
        # 

        print('='*80,file=f)
        print('\t\t\t\tloop : {}'.format(str(loop)),file=f)
        print(".."*20,file=f)
       
        print('Algorithm computed parameters',file=f)
        
        print(".."*20,file=f)
        print("Youngs modulus : {}".format(str(youngs_modulus)),file=f)
        print("Yield stress : {}".format(str(fem_parameter[0])),file=f)
        print("Linear Hardening modulus : {}".format(str(fem_parameter[1])),file=f)
        print("Non linear hardening modulus : {}".format(str(fem_parameter[2])),file=f)
        print("Hardening exponent : {}".format(str(fem_parameter[3])),file=f)
        
        print('='*80,file=f)
       

        #
        #
        # Adjusting lamda value to increase accuracy
        #
        #


        if a <= 0.25:                     # if a is less than 0.25 lamda will increase
            lamda = lamda*10
            

        else:
            if a > 0.75:
                lamda = lamda/10         # if a is greater than 0.75 lamda will decrease
                
            
        #
        # if a is positive then the step is accepted and parameters array, residual, jacobian will update
        #
        if a > eta:
            #print('step accepted')
            x_0 = x_1
            residual_0 = residual_1
            f_0 = f_1
            jacobian_0 = jacobian_1 
            j = 1
            i += 1

        #
        # if a is negative the step is not accepted and the parameter array, residual, jacobian are not changed. only lamda will change
        #

        else:
            #print('step rejected')
            delta_f = 1
            j += 1

        #
        # Warnings if the total iteration reaches the maximum and lamda iteration
        #

        if i > max_iteration:
            print('maximum number of iteration is reached')
        if j > lamda_iteration:
            print('maximum number of lamda iteration is reached')

        

        
        loop = loop +1
        
    #
    # optimized parameter wirting in to a file
    #
    
    print("\n\n",file=f)
    print(".."*50,file=f)
    print("Optimized parameter of the material",file=f)
    print(".."*50,file=f)
    print("Youngs modulus : {}".format(str(youngs_modulus)),file=f)
    print('Yield stress : {}'.format(str(x_1[0]*scaling_parameter[0])),file=f)
    print('Hardening modulus : {}'.format(str(x_1[1]*scaling_parameter[1])),file=f)
    print('Non-linear hardening modulus : {}'.format(str(x_1[2]*scaling_parameter[2])),file=f)
    print('Hardening exponent : {}'.format(str(x_1[3]*scaling_parameter[3])),file=f)
    print('='*80,file=f)
     
    return fem_parameter,youngs_modulus


'''
=================================================================================================================================================================================
'''


if __name__=="__main__":
    #
    # writing all output in to a document 
    #
    #
    with open('main_program.txt.txt', 'w') as f:
        print("Generated on : {}".format(str(dt.datetime.now())),file=f)
        print(("="*150),file=f)
        print(("="*150),file=f)
        print("\t\t\t\tParameter identification of an elasto-plastic materials using FEM program",file=f)
        print(("="*150),file=f)
        print(("="*150),file=f)
        print("Initial guess of the parameters",file=f)
        print(("."*20),file=f)

        #
        #
        # Getting guess for the parameters from the user
        #
        #

        print('Guess the parameter which is close to the real parameters')
        print("Enter young's modulus E : ")
        youngs_modulus = int(input())
        print("Enter yield stress : ")
        yield_stress = int(input())
        print("Enter linear hardening modulus : ")
        hardening_modulus = int(input())
        print("Enter Non-linear hardening modulus : ")
        delta_y = int(input())
        print("Enter hardening exponent : ")
        eta = int(input())

        #
        #
        #  writing guess in to the file
        #
        #
        print("Youngs modulus : {} ".format(str(youngs_modulus)),file=f)
        print("Yield stress : {}".format(str(yield_stress)),file=f)
        print("Linear Hardening modulus : {}".format(str(hardening_modulus)),file=f)
        print("Non linear hardening modulus : {}".format(str(delta_y)),file=f)
        print("Hardening exponent : {}".format(str(eta)),file=f)
        

        #
        #
        # Reading experimental data from a file
        #
        #

        exp_file = 'noise_data.txt'
        exp_stress = np.loadtxt(exp_file,usecols=(0))
        exp_strain = np.loadtxt(exp_file,usecols=(1))

        #
        # Calling Fem program to run the simulation for the initial guess and writing to a file for further use
        #
        #


        sim_stress ,sim_strain = nonlinear_fem(youngs_modulus,yield_stress,hardening_modulus,delta_y,eta)
        with open('simulation_data.txt', 'w') as b:
            b.write('0')
            b.write('\t\t0\n')
            for s,st in zip(sim_stress,sim_strain):
                print(s,st,file = b)
            b.close()

        #
        #
        # Here i am calling interpolation function and residual error
        # The experimental datas and FEM datas are not similar so, interpolation finds data points similar to experimental data 
        # And residual eroor calculates mean squarewd error between experimental data and sim data
        #
        #

        interpolation()
        before_residual_error = residual_error()
        
        #
        #
        # Plotting the experimental data and FEM data(initial guess) to visualize the difference
        #
        #
        fig,ax = plt.subplots(2,constrained_layout=True)
        ax[0].scatter(exp_strain,exp_stress,color='blue',marker='.', label='Experimental data')
        ax[0].plot(sim_strain,sim_stress,color='red',label='Intial simulation data')
        ax[0].legend()
        ax[0].grid()
        
        #
        #
        # For optimization with levenberg we are using Voce hardening law to find  the parameters in that voce law deals with data points only from yield stress
        # elastic part is not needed for voce law. i am asumming yield point by visualizing the experimental data. here i am seperating the data points for the 
        # further calculation.

        #
        # Reading the data points
        #

        exp_file = 'noise_data.txt'
        sim_file = 'interpolated_simulation_data.txt'
        exp_stress_1 = np.loadtxt(exp_file,usecols=(0))
        exp_strain_1 = np.loadtxt(exp_file,usecols=(1))
        sim_stress_1 = np.loadtxt(sim_file,usecols=(0))
        sim_strain_1 = np.loadtxt(sim_file,usecols=(1))

        #
        #
        #  Seperating the data points 
        #
        #
        
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


        #
        #
        # Warnings if the given guess is close to the real solution the program will exit 
        #
        #
        
        if before_residual_error < 600:
            print(".."*20)
            print("The guess is close to real solution")
            print(".."*20)
            exit()
        
        #
        #
        # Here the levenberg algorithm is initiated for the given parameters
        #
        #

        if before_residual_error > 1e-3:
            #
            #
            # X : normalize vector, instead of applying the parameters directly, normalizing vector is used to optimize the result close to the real solution  
            # Levenberg_residual_error : stores mean squared error at each iteration
            # Levenberg , youngs_modulus : calling the levenberg_marquardt algorithm fumction 
            #
            
            x = np.array([0.8,0.7,0.8,0.4])
            levenberg_residual_error = np.zeros((300))
            levenberg,youngs_modulus = levenberg_marquardt(x,youngs_modulus,exp_stress[:len(sim_stress)],exp_strain[:len(sim_strain)],sim_stress,sim_strain, levenberg_residual_error, lamda = 0.001, f_tol = 1e-3, eta = 0.0001, max_iteration = 200, lamda_iteration = 20)
            #
            #
            # Again calling the FEM function to with optimized parameter and saving that into the file 
            #
            #
            sim_stress ,sim_strain = nonlinear_fem(youngs_modulus,levenberg[0],levenberg[1],levenberg[2],levenberg[3])
        
            with open('simulation_data.txt', 'w') as c:
                c.write('0')
                c.write('\t\t0\n')
                for s,st in zip(sim_stress,sim_strain):
                    print(s,st,file = c)
                c.close()
            
            #
            #
            # Calling interpolation function to match the data with experimental data
            # And finding the mean squared error between the expermental and FEM data
            #

            interpolation()
            after_residual_error = residual_error()
            yield_stress = levenberg[0]
            hardening_modulus = levenberg[1]
            delta_y = levenberg[2]
            eta = levenberg[3]
            #
            #
            # plotting the experimental and FEM data to visualize the result
            #
            #
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
            
        #
        #
        # training and testing function is called. the experimental data is splited into testing and training data
        # And plotting the data to see the simulation curve fits with training and testing data
        #
        training_testing()

        #
        #
        # Here plotting the Mean squared error in each iterations to see how the error got reduced
        #
        #
        
        levenberg_residual_error = levenberg_residual_error[np.nonzero(levenberg_residual_error)] 
        x_mse = range(len(levenberg_residual_error)) 
        plt.plot(x_mse,levenberg_residual_error, color = 'blue')
        plt.title('Mean squared error vs Number of iterations ')
        plt.xlabel('Number of iterations')
        plt.ylabel('Mean squared error')
        plt.grid()
        plt.savefig('mse')
        plt.close()

        #
        #
        # Calculating the total time taken to complete the program
        #
        #
        end_time = time.time()
        total_time = (end_time-start_time)/60
        print('time in minutes : {}'.format(str(total_time)),file=f)
        f.close()