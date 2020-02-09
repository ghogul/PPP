import numpy as np
import matplotlib.pyplot as plt



'''
================================================================================================================================================================================================
                                                                  INITIAL PARAMETERS
================================================================================================================================================================================================
'''
'''***MATERIAL PARAMETER***'''
poission_ratio = 0.3
youngs_modulus = 210000 #Mpa


'''***GEOMETRICAL PARAMETER***'''
length = 70  #mm
radius = 5   #mm


'''***TIME PARAMETERS***'''
tau = 0
time_step = 0.1
total_time = 1

'''***EXTERNAL LOADING***'''
f_ext = 500  #Mpa
yield_stress = 350 #Mpa


'''***lame's constant***'''
mu = youngs_modulus/(2*(1+poission_ratio))
lamda = (poission_ratio*youngs_modulus)/((1+poission_ratio)*(1-2*poission_ratio))
k = youngs_modulus/(3*(1-(2*poission_ratio)))


'''***DISCRETIZATION PARAMETERS***'''
nelm = 4
isoparametric_edge = 4
d_o_f = 2
no_nodes = int((np.sqrt(nelm)+1)*(np.sqrt(nelm)+1))
# length_element = 2.5

'''***graphical representation of elements***'''
nelm_length = np.linspace(0,length,np.sqrt(nelm)+1)
nelm_radius = np.linspace(0,radius,np.sqrt(nelm)+1)
yy,xx = np.meshgrid(nelm_length,nelm_radius)

# fig, ax = plt.subplots()
# ax.scatter (xx,yy)
# plt.show()


'''***shape function for isoparametric element***'''
# E1 = 0
# E2 = 0
# N = ([[(1-E1)/4,(1-E2)/4],
#           [(1+E1)/4,(1-E2)/4],
#           [(1+E1)/4,(1+E2)/4],
#           [(1-E1)/4,(1+E2)/4]])
weight = 1

'''
================================================================================================================================================================================================
'''


'''
================================================================================================================================================================================================
                               deriving all elements coordinates for the calculating jacobian
================================================================================================================================================================================================
'''
def elements_coordinates(nelm,nelm_length,nelm_radius):
    all_ele_coord = np.zeros((nelm,4,2))
    loop = 0
    for j in range(int(np.sqrt(nelm))):
        for i in range(int(np.sqrt(nelm))):

            ele_coord = np.matrix(([[nelm_radius[i],nelm_length[j]],
                        [nelm_radius[i+1],nelm_length[j]],
                        [nelm_radius[i+1],nelm_length[j+1]],
                        [nelm_radius[i],nelm_length[j+1]]]))
            ele_coord = ele_coord.reshape((4,2))
            all_ele_coord_zeros = all_ele_coord[loop]+ele_coord  
            all_ele_coord[loop] = all_ele_coord_zeros
            loop += 1
    return all_ele_coord

'''
================================================================================================================================================================================================
'''

# area = all_ele_coord[0,1,0]*all_ele_coord[0,2,1]    

'''
================================================================================================================================================================================================
                                                          material rotuine
================================================================================================================================================================================================
'''

def material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,global_plastic_strain,yield_stress,mu,lamda):
    epsilon = np.dot(B_matrix , initial_displacement[i])
    strain = epsilon
    # print(strain)
    c_1 = (youngs_modulus)/(1-poission_ratio**2)
    C = c_1*np.matrix([[1,poission_ratio,0,0],
                       [poission_ratio,1,0,0],
                       [0,0,0,0],
                       [0,0,0,((1-poission_ratio)/2)]])
    trial_stress = np.dot(C , (strain - global_plastic_strain[i]))
    trial_stress_deviatoric = trial_stress - (np.sum(trial_stress)/3)
    trial_stress_equivalent = np.sqrt(((3/2) * (np.sum(trial_stress_deviatoric))**2))
    print(trial_stress_equivalent)
    if trial_stress_equivalent-yield_stress < 0:
        # print("elastic")
        stress = trial_stress
        return C, stress

    else:
        # print("plastic")
        delta_lamda = (trial_stress_equivalent-yield_stress)/(3*mu)
        current_stress = ((1/3)*np.sum(trial_stress)*np.ones(4).reshape(4,1)) + (((trial_stress_equivalent-(3*mu*delta_lamda)))/trial_stress_equivalent)*trial_stress_deviatoric
        current_stress_deviatoric = current_stress - (1/3)*np.sum(current_stress)
        current_stress_equivalent = np.sqrt ((3/2)*(np.sum(current_stress_deviatoric)**2))
        plastic_strain[i] = global_plastic_strain[i] + ((delta_lamda * 3/2) / current_stress_equivalent) * current_stress_deviatoric
        # print(plastic_strain[i])
        stress = current_stress
        c_t_1st = (1/3) * (3*lamda + 2*mu)*np.ones((4,4))
        identity_deviatoric = ((1/2)*(np.eye(4)+np.eye(4)))-((1/3)*np.ones((4,4)))
        c_t_2nd = (2*mu)*((trial_stress_equivalent-(3*mu*delta_lamda))/trial_stress_equivalent)*identity_deviatoric
        c_t_3rd = (3*mu / (trial_stress_equivalent)**2)*(trial_stress_deviatoric*trial_stress_deviatoric.reshape(1,4))
        C_t = c_t_1st + c_t_2nd - c_t_3rd
        return C_t, stress
'''
================================================================================================================================================================================================
'''
'''
================================================================================================================================================================================================
                                                      element rotuine
================================================================================================================================================================================================
'''
# element rotuine
def element_rotuine(radius):
    k_all_ele = np.zeros((nelm,isoparametric_edge*2,isoparametric_edge*2))
    all_ele_coord = elements_coordinates(nelm,nelm_length,nelm_radius)
    for i in range(nelm):
        k_all = np.zeros((isoparametric_edge*2,isoparametric_edge*2))
        for j in range(4):
            if j == 0:
                E1 = -((np.sqrt(1/3)))
                E2 = -((np.sqrt(1/3)))

            elif j == 1:
                E1 = +((np.sqrt(1/3)))
                E2 = -((np.sqrt(1/3)))

            elif j == 2:
                E1 = +((np.sqrt(1/3)))
                E2 = +((np.sqrt(1/3)))

            elif j == 3:
                E1 = -((np.sqrt(1/3)))
                E2 = +((np.sqrt(1/3)))
                
            derivative_N = 1/4*np.matrix([[-(1-E2),(1-E2),(1+E2),-(1+E2)],
                                          [-(1-E1),-(1+E1),(1+E1),(1-E1)]])
            x_y_ele = all_ele_coord[i]
            jacobi_1 = derivative_N*x_y_ele
            jacobi_inverse = np.linalg.inv(jacobi_1)
            B_1_matrix = jacobi_inverse*derivative_N
            ele_radius = (x_y_ele[0,0]+x_y_ele[1,0]+x_y_ele[2,0]+x_y_ele[3,0])/isoparametric_edge
            area = all_ele_coord[0,1,0]*all_ele_coord[0,2,1]    
            weight = 1
            N_1 = N_2 = N_3 = N_4 = 1/4
            B_matrix = np.matrix([[B_1_matrix[0,0],0,B_1_matrix[0,1],0,B_1_matrix[0,2],0,B_1_matrix[0,3],0],
                                [0,B_1_matrix[1,0],0,B_1_matrix[1,1],0,B_1_matrix[1,2],0,B_1_matrix[1,3]],
                                [N_1/ele_radius,0,N_2/ele_radius,0,N_3/ele_radius,0,N_4/ele_radius,0],
                                [B_1_matrix[1,0],B_1_matrix[0,0],B_1_matrix[1,1],B_1_matrix[0,1],B_1_matrix[1,2],B_1_matrix[0,2],B_1_matrix[1,3],B_1_matrix[0,3]]])
            c_matrix,stress_matrix = material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,global_plastic_strain,yield_stress,mu,lamda)
            k_1 = 2*np.pi*ele_radius*area
            k_2 = np.dot(c_matrix,B_matrix)
            k_3 = np.dot(np.transpose(B_matrix),k_2)
            k_all = k_all + weight*weight*k_1*k_3*np.linalg.det(jacobi_1)
            
        k_all_ele[i] = k_all
        
        
    return k_all_ele                   

'''
================================================================================================================================================================================================
'''
'''
================================================================================================================================================================================================
                                                      internal force matrix
================================================================================================================================================================================================
'''
# element rotuine
def internal_force(radius):
    internal_force_matrix_all_ele = np.zeros((nelm,isoparametric_edge*2,1))
    all_ele_coord = elements_coordinates(nelm,nelm_length,nelm_radius)
    for i in range(nelm):
        E1 = 0
        E2 = 0
        derivative_N = 1/4*np.matrix([[-(1-E2),(1-E2),(1+E2),-(1+E2)],
                                      [-(1-E1),-(1+E1),(1+E1),(1-E1)]])
        x_y_ele = all_ele_coord[i]
        jacobi_1 = derivative_N*x_y_ele
        jacobi_inverse = np.linalg.inv(jacobi_1)
        B_1_matrix = jacobi_inverse*derivative_N
        ele_radius = (x_y_ele[0,0]+x_y_ele[1,0]+x_y_ele[2,0]+x_y_ele[3,0])/isoparametric_edge
        area = all_ele_coord[0,1,0]*all_ele_coord[0,2,1]    
        weight = 2
        N_1 = N_2 = N_3 = N_4 = 1/4
        B_matrix = np.matrix([[B_1_matrix[0,0],0,B_1_matrix[0,1],0,B_1_matrix[0,2],0,B_1_matrix[0,3],0],
                            [0,B_1_matrix[1,0],0,B_1_matrix[1,1],0,B_1_matrix[1,2],0,B_1_matrix[1,3]],
                            [N_1/ele_radius,0,N_2/ele_radius,0,N_3/ele_radius,0,N_4/ele_radius,0],
                            [B_1_matrix[1,0],B_1_matrix[0,0],B_1_matrix[1,1],B_1_matrix[0,1],B_1_matrix[1,2],B_1_matrix[0,2],B_1_matrix[1,3],B_1_matrix[0,3]]])
        c_matrix,stress_matrix = material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,global_plastic_strain,yield_stress,mu,lamda)
        internal_force_matrix = weight*2*np.pi*radius*area*np.transpose(B_matrix)*stress_matrix*np.linalg.det(jacobi_1)
        internal_force_matrix_all_ele[i] = internal_force_matrix
        
    return internal_force_matrix_all_ele                       

'''
================================================================================================================================================================================================
'''



'''
================================================================================================================================================================================================
                                                               external force
================================================================================================================================================================================================
'''
# finding the external forces of the elements
def external_force(f_ext,nelm,i,external_force_ele,radius):
    # all_ele_coord = elements_coordinates(nelm,nelm_length,nelm_radius)
    if i < int(nelm-(np.sqrt(nelm))):
        external_force_ele[i] = (np.zeros((isoparametric_edge*2,1)))
        return (np.zeros((isoparametric_edge*2,1)))

    else:
        # x_y_ele = all_ele_coord[i]
        ele_radius = radius/int(np.sqrt(nelm))
        ele_length = radius/int(np.sqrt(nelm))
        external_force = 2*np.pi*ele_radius*((ele_length/2)*f_ext*(1/4))*np.transpose(np.matrix([0,0,0,0,0,4,0,4]))
        external_force_ele[i] = external_force
        return (external_force)   

'''
================================================================================================================================================================================================
'''
    
'''
================================================================================================================================================================================================
                                                      assignment matrix
================================================================================================================================================================================================
'''
def assignment_matrix(nelm,isoparametric_edge,d_o_f):
    # nodes of the elements
    x_cells = int(np.sqrt(nelm))
    y_cells = int(np.sqrt(nelm))
    elements = np.arange((x_cells+1)*(y_cells+1)).reshape(y_cells+1,x_cells+1)
    # for i in range(0,(x_cells+1),1):
    #     if i % 2 != 0:
    #         elements_1 = elements[i]
    #         elements[i] = elements_1[::-1]
    ID = ([])
    for i in range(y_cells):
        for j in range(x_cells):
            id = np.matrix([elements[i,j],elements[i,j+1],elements[i+1,j],elements[i+1,j+1]])
            ID = np.append(ID,id)
    summation = ([])  
    for i in ID:
        multi = np.array([(i*2),((i*2)+1)])
        summation = np.append(summation,multi)
    summation = summation.astype(int)
    a_column = summation.flatten().reshape(nelm,isoparametric_edge*d_o_f)
    # print(sum)
    #  assignment matrix for all the elements
    a_rows = (np.arange(isoparametric_edge*d_o_f)).astype(int)
    all_a = np.zeros((nelm,isoparametric_edge*d_o_f , no_nodes*d_o_f))
    for k in range(nelm):
        a = np.zeros((isoparametric_edge*d_o_f , no_nodes*d_o_f))
        for i,j in zip(a_rows,a_column[k]):
            a[i,j] = 1
        all_a[k] = a
    return all_a,summation

'''
================================================================================================================================================================================================
'''

'''
================================================================================================================================================================================================
                                                            assembly and newton-raphson 
================================================================================================================================================================================================
'''
all_a,summation = assignment_matrix(nelm,isoparametric_edge,d_o_f)
global_initial_displacement = np.random.rand((no_nodes*2),1)
a = np.array([])
for i in summation:
    a = np.append(a,global_initial_displacement[i])
initial_displacement = a.reshape((nelm,isoparametric_edge*2,1))
global_plastic_strain = np.zeros((nelm,isoparametric_edge,1))

while True:
    tau = tau + time_step
    if tau > (total_time-time_step):
        break
    plastic_strain = np.zeros((nelm,isoparametric_edge,1))
    while True:
        k_ele = element_rotuine(radius)
        # print(k_ele)
        internal_force_matrix_ele = internal_force(radius)
        external_force_ele = np.zeros((nelm,8,1))
        for i in range(nelm):
            external_force(f_ext,nelm,i,external_force_ele,radius)

        all_a,summation = assignment_matrix(nelm,isoparametric_edge,d_o_f)
        #assembly
        global_stiffness_matrix = np.zeros((no_nodes*2,no_nodes*2))
        global_internal_force_matrix = 0
        global_external_force_matrix = 0
        for i in range(nelm):
            assembly_1 = np.dot((np.transpose(all_a[i])),k_ele[i])
            assembly = np.dot(assembly_1,all_a[i])
            global_stiffness_matrix = global_stiffness_matrix + assembly
            global_internal_force_matrix = global_internal_force_matrix + np.dot((np.transpose(all_a[i])),internal_force_matrix_ele[i])
            global_external_force_matrix = global_external_force_matrix + np.dot((np.transpose(all_a[i])),external_force_ele[i])


        G_matrix = global_internal_force_matrix - (global_external_force_matrix*tau)


        reduced_global_stiffness_matrix_1 = global_stiffness_matrix[2:,2:]
        reduced_global_stiffness_matrix_2 = np.delete(reduced_global_stiffness_matrix_1,1,1)
        reduced_global_stiffness_matrix_3 = np.delete(reduced_global_stiffness_matrix_2,1,0)
        reduced_global_stiffness_matrix_4 = np.delete(reduced_global_stiffness_matrix_3,2,1)
        reduced_global_stiffness_matrix_5 = np.delete(reduced_global_stiffness_matrix_4,2,0)
        reduced_global_stiffness_matrix_6 = np.delete(reduced_global_stiffness_matrix_5,2,1)
        reduced_global_stiffness_matrix_7 = np.delete(reduced_global_stiffness_matrix_6,2,0)
        reduced_global_stiffness_matrix_8 = np.delete(reduced_global_stiffness_matrix_7,7,1)
        reduced_global_stiffness_matrix   = np.delete(reduced_global_stiffness_matrix_8,7,0)


        reduced_G_matrix_1 = np.delete(G_matrix,0,0)
        reduced_G_matrix_2 = np.delete(reduced_G_matrix_1,0,0)
        reduced_G_matrix_3 = np.delete(reduced_G_matrix_2,1,0)
        reduced_G_matrix_4 = np.delete(reduced_G_matrix_3,2,0)
        reduced_G_matrix_5 = np.delete(reduced_G_matrix_4,2,0)
        reduced_G_matrix   = np.delete(reduced_G_matrix_5,7,0)

        delta_displacement = np.dot(np.linalg.inv(reduced_global_stiffness_matrix),reduced_G_matrix)
        # print(delta_displacement)
        delta_displacement = np.insert(delta_displacement,0,0)
        delta_displacement = np.insert(delta_displacement,1,0)
        delta_displacement = np.insert(delta_displacement,3,0)
        delta_displacement = np.insert(delta_displacement,5,0)
        delta_displacement = np.insert(delta_displacement,6,0)
        delta_displacement = np.insert(delta_displacement,12,0)
        delta_displacement = delta_displacement.reshape(18,1)
        global_displacement = delta_displacement - global_initial_displacement

        b = np.array([])
        for j in summation:
            b = np.append(b,global_displacement[j])
        global_displacement_ele = b.reshape((nelm,isoparametric_edge*2,1))
        # print(global_displacement.shape)
        # print(inital_displacement)

        initial_displacement = np.copy(global_displacement_ele)
        # global_plastic_strain = plastic_strain
        # print(delta_displacement)
        if (np.linalg.norm(delta_displacement,np.inf) < (0.005 * np.linalg.norm(global_displacement,np.inf))) or (np.linalg.norm(reduced_G_matrix,np.inf) < 0.005*np.linalg.norm(global_internal_force_matrix,np.inf)):
            break


    



   





                                                                                                                             

# print(inital_displacement)











































































# print(k_global)

 
# while tau <(total_time-time_step):
#         tau = tau + time_step
#         # boundary condition
#         nodal_position = (isoparametric_edge*d_o_f)
#         initial_displacement = np.random.rand(8,1)
#         initial_displacement[0] = initial_displacement[1] = 0
#         print(element_rotuine(E1,E2,all_ele_coord,N,area))

# print(all_a[0])
































