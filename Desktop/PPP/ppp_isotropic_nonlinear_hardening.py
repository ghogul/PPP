import numpy as np
import matplotlib.pyplot as plt

def nonlinear_fem(youngs_modulus,yield_stress,hardening_modulus,delta_y,eta):

    '''
    ================================================================================================================================================================================================
                                                                    INITIAL PARAMETERS
    ================================================================================================================================================================================================
    '''
    '''***MATERIAL PARAMETER***'''
    # print('='*80)
    # print('\t\t\tFEM')
    # print('='*80)
    poission_ratio = 0.3
    #youngs_modulus = 21000 #Mpa
    #hardening_modulus = 200 #Mpa
    #delta_y = 200
    #eta = 10
    # print("poisiion_ratio",poission_ratio)
    # print("youngs_mpdulus",youngs_modulus)
    # print("yield_stress",yield_stress)
    # print("hardening_modulus",hardening_modulus)
    # print("delta_y",delta_y)
    # print("eta",eta)
    '''***GEOMETRICAL PARAMETER***'''
    length = 50  #mm
    radius = 0.5   #mm
    # print("length",length)
    # print("radius",radius)
    '''***TIME PARAMETERS***'''
    tau = 0
    time_step = 0.01
    total_time = 1
    # print("time_step",time_step)

    '''***EXTERNAL LOADING***'''
    f_ext = 75000  #Mpa
    #yield_stress = 490.25 #Mpa    mild steel
    #print("f_ext",f_ext)
    #print("yield_stress",yield_stress)
    #print('='*80)

    '''***lame's constant***'''
    mu = (youngs_modulus/(2*(1+poission_ratio)))
    lamda = (poission_ratio*youngs_modulus)/((1+poission_ratio)*(1-2*poission_ratio))
    bulk = youngs_modulus/(3*(1-(2*poission_ratio)))


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
    gauss_point = 4
    '''
    ================================================================================================================================================================================================
    '''


    '''
    ================================================================================================================================================================================================
                                deriving all elements coordinates for the calculating jacobian
    ================================================================================================================================================================================================
    '''
    def elements_coordinates(nelm,nelm_length,nelm_radius):
        '''
        
        
        '''
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

    def material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,j,global_plastic_strain,yield_stress,mu,lamda,bulk,alpha):
        # print("plastic strain",global_plastic_strain)
        epsilon = np.dot(B_matrix , initial_displacement[i])
        strain = epsilon
        strain_deviatoric = np.copy(strain)
        strain_deviatoric[:3] = (strain[:3] - ((1/3)*strain[1]+strain[2]+strain[3]))
        strain_equivalent[i,j] = (np.sqrt(((3/2) * np.dot(strain_deviatoric.T,strain_deviatoric)))).item()
        #print("strain eq",strain_equivalent[i,j])
        #print("strain",strain)
        #print("initial displacement",initial_displacement[i])
        c_1 = (youngs_modulus)/((1+poission_ratio)*(1-2*poission_ratio))
        C = c_1*np.array([[1-poission_ratio,poission_ratio,poission_ratio,0],
                        [poission_ratio,1-poission_ratio,poission_ratio,0],
                        [poission_ratio,poission_ratio,1-poission_ratio,0],
                        [0,0,0,((1-2*poission_ratio)/2)]])

        
        elastic_strain = strain - global_plastic_strain[i,j]
        elastic_strain_deviatoric = np.copy(elastic_strain)
        elastic_strain_deviatoric[:3] = (strain[:3] - ((1/3)*strain[1]+strain[2]+strain[3]))
        elastic_strain_equivalent = (np.sqrt(((3/2) * np.dot(elastic_strain_deviatoric.T,elastic_strain_deviatoric)))).item()

        #print("global plastic strain predictor",global_plastic_strain[i,j])
        # print("elastic strain", elastic_strain)
        # print("material rotuine")
        # print("Strain:",strain)
        # print("C:",C)
        trial_stress = np.dot(C,elastic_strain)
        
        #print("TRIAL:",trial_stress)
        trial_stress_deviatoric = np.copy(trial_stress)
        trial_stress_deviatoric[:3] = trial_stress[:3] - ((1/3)*(trial_stress[0]+trial_stress[1]+trial_stress[2]))
        #print("trial_deviatoric",trial_stress_deviatoric)
        n = np.divide(trial_stress_deviatoric,np.linalg.norm(trial_stress_deviatoric),out=np.zeros_like(trial_stress_deviatoric),where=np.linalg.norm(trial_stress_deviatoric)!=0)
        #print("n",n)
        trial_stress_equivalent = (np.sqrt(((3/2) * np.dot(trial_stress_deviatoric.T,trial_stress_deviatoric)))).item()
        # print(trial_stress)
        # print(strain)
        # print(C)
        # print(trial_stress)
        # # print(strain)
        #print("eq_stress",trial_stress_equivalent)
        #print("alpha",alpha_updated[i,j])
        #np.longdouble(eta)
        beta = (hardening_modulus*alpha[i,j]) + delta_y*(1-np.exp(-(eta)*alpha[i,j]))
        #print("Beta", beta)
        phi = np.linalg.norm(trial_stress_deviatoric) - (np.sqrt(2/3)*(yield_stress + beta))
        #print("phi",phi)
        if phi <= 0:
            # print("elastic")
            #print("C",C)
            stress_equivalent[i,j] = trial_stress_equivalent
            elastic_plastic_strain_equivalent[i,j] = elastic_strain_equivalent
            # stress = trial_stress
            return C, trial_stress

        else:
            # print("plastic")
            #print("mu",mu)
            delta_lamda = ((phi)/((2*mu)+((2/3)*hardening_modulus)))
            #print("delta lamda",delta_lamda)
            plastic_strain[i,j] = global_plastic_strain[i,j] + (delta_lamda * n) 
            current_plastic = plastic_strain[i,j]
            plastic_strain_deviatoric = np.copy(plastic_strain[i,j])
            plastic_strain_deviatoric[:3] = (current_plastic[:3] - ((1/3)*current_plastic[1]+current_plastic[2]+current_plastic[3]))
            elastic_plastic_strain_equivalent[i,j] = (np.sqrt(((3/2) * np.dot(plastic_strain_deviatoric.T,plastic_strain_deviatoric)))).item()
            #print("global plastic strain corrector",plastic_strain[i,j])
            current_stress = C*(strain - plastic_strain[i,j])
            # print(np.asscalar((trial_stress_equivalent-(3*mu*delta_lamda))/trial_stress_equivalent))
            # current_1 = ((bulk*(strain[0]+strain[1]+strain[2]))*np.ones(4)).reshape(4,1)
            # current_1[3] = 0
            # print("current_1",current_1)
            # current_stress = current_1 + (trial_stress_deviatoric - (2*mu*delta_lamda*n))
            #print("current stress ",current_stress)
            current_stress_deviatoric = np.copy(current_stress)
            current_stress_deviatoric[:3] = current_stress[:3] - ((1/3)*(current_stress[0]+current_stress[1]+current_stress[2]))
            current_stress_equivalent = (np.sqrt ((3/2)*(np.dot(current_stress_deviatoric.T,current_stress_deviatoric)))).item()
            stress_equivalent[i,j] = current_stress_equivalent
            #print("current_stress_deviatoric", current_stress_deviatoric)
            #print("current stress eq",current_stress_equivalent)
            # print("global plastic strain",global_plastic_strain[i])
            
            alpha_updated[i,j] =  (np.sqrt(((3/2) * np.dot(plastic_strain_deviatoric.T,plastic_strain_deviatoric)))).item()
            #print("Alpha", alpha_updated[i,j])
            # stress = current_stress
            beta_1 = 1-((np.divide(phi,np.linalg.norm(trial_stress_deviatoric),out=np.zeros_like(trial_stress_deviatoric),where=np.linalg.norm(trial_stress_deviatoric)!=0))*(1/(1+(hardening_modulus/3*mu))))
            beta_2 = (1-(np.divide(phi,np.linalg.norm(trial_stress_deviatoric),out=np.zeros_like(trial_stress_deviatoric),where=np.linalg.norm(trial_stress_deviatoric)!=0)))*(1/(1+(hardening_modulus/3*mu)))
            #print("beta 1 ",beta_1)
            #print("beta 2",beta_2)
            identity = np.ones((4,4))
            identity[3,0] = identity[3,1] = identity[3,2] = identity[3,3] = identity[0,3] = identity[1,3] = identity[2,3] = 0
            identity_deviatoric = ((1/2)*(np.eye(4)+np.eye(4)))-((1/3)*np.ones((4,4)))
            identity_deviatoric[3,0] = identity_deviatoric[3,1] = identity_deviatoric[3,2] = identity_deviatoric[0,3] = identity_deviatoric[1,3] = identity_deviatoric[2,3] = 0
            identity_deviatoric[3,3] = 0.5
            n_tensor = np.dot(n,n.T)
            #print("n_tensor", n_tensor)
            c_t_1st = bulk * identity
            #print("c_t_1st",c_t_1st)
            c_t_2nd = 2*mu*beta_1*identity_deviatoric
            #print("c_t_2nd",c_t_2nd)
            c_t_3rd = 2*mu*beta_2*n_tensor
            #print("c_t_3nd",c_t_3rd)
            c_t_deviatoric = c_t_2nd - c_t_3rd
            #print("c_t_dev",c_t_deviatoric)
            C_t = c_t_deviatoric + c_t_1st



            
            
            
            #print("C_t",C_t)
            return C_t, current_stress
    '''
    ================================================================================================================================================================================================
    '''
    '''
    ================================================================================================================================================================================================
                                                        element rotuine
    ================================================================================================================================================================================================
    '''
    # element rotuine
    def element_rotuine(radius,gauss_point):
        internal_force_matrix_all_ele = np.zeros((nelm,isoparametric_edge*2,1))
        k_all_ele = np.zeros((nelm,isoparametric_edge*2,isoparametric_edge*2))
        all_ele_coord = elements_coordinates(nelm,nelm_length,nelm_radius)
        for i in range(nelm):
            #print("elements",i)
            internal_force_matrix = 0
            k_all = np.zeros((isoparametric_edge*2,isoparametric_edge*2))
            for j in range(gauss_point):
                #print("gauss point",j)
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
                # print("deri_n", derivative_N)
                x_y_ele = all_ele_coord[i]
                jacobi_1 = derivative_N*x_y_ele
                jacobi_inverse = np.linalg.inv(jacobi_1)
                B_1_matrix = jacobi_inverse*derivative_N
                ele_radius = (x_y_ele[0,0]+x_y_ele[1,0]+x_y_ele[2,0]+x_y_ele[3,0])/isoparametric_edge
                area = all_ele_coord[0,1,0]*all_ele_coord[0,2,1]    
                weight = 1
                # N_1 = N_2 = N_3 = N_4 = 1/4
                B_matrix = np.matrix([[B_1_matrix[0,0],0,B_1_matrix[0,1],0,B_1_matrix[0,2],0,B_1_matrix[0,3],0],
                                    [0,B_1_matrix[1,0],0,B_1_matrix[1,1],0,B_1_matrix[1,2],0,B_1_matrix[1,3]],
                                    [((1-E1)*(1-E2)/4)/ele_radius,0,((1+E1)*(1-E2)/4)/ele_radius,0,((1+E1)*(1+E2)/4)/ele_radius,0,((1-E1)*(1+E2)/4)/ele_radius,0],
                                    [B_1_matrix[1,0],B_1_matrix[0,0],B_1_matrix[1,1],B_1_matrix[0,1],B_1_matrix[1,2],B_1_matrix[0,2],B_1_matrix[1,3],B_1_matrix[0,3]]])
                c_matrix,stress_matrix = material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,j,global_plastic_strain,yield_stress,mu,lamda,bulk,alpha)
                k_1 = 2*np.pi*ele_radius*area
                k_2 = np.dot(c_matrix,B_matrix)
                k_3 = np.dot(np.transpose(B_matrix),k_2)
                k_all = k_all + weight*weight*k_1*k_3*np.linalg.det(jacobi_1)

                internal_force_matrix = internal_force_matrix + weight*weight*2*np.pi*ele_radius*area*np.dot(np.transpose(B_matrix),stress_matrix)*np.linalg.det(jacobi_1)
                #print("internal element",internal_force_matrix)
            k_all_ele[i] = k_all
            internal_force_matrix_all_ele[i] = internal_force_matrix
            
            
        return k_all_ele, internal_force_matrix_all_ele              

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

    global_displacement = np.zeros((no_nodes*d_o_f,1)) 
    global_plastic_strain = np.zeros((nelm,gauss_point,isoparametric_edge,1))

    G_matrix = 0
    # print(global_displacement)
    # global_displacement = np.random.random((18,1))*0.04
    a = np.array([])
    for i in summation:
        a = np.append(a,global_displacement[i])
    initial_displacement = a.reshape((nelm,isoparametric_edge*d_o_f,1))
    all_stress = np.array([])
    all_strain = np.array([])
    all_elastic_plastic_strain = np.array([])
    alpha = np.zeros((nelm,gauss_point,1))
    time = 0
    while True:
        tau = tau + time_step
        time += 1
        #print("="*60)
        #print("time",tau)
        if tau > (total_time-time_step):
            break
        plastic_strain = np.zeros((nelm,gauss_point,isoparametric_edge,1))
        alpha_updated = np.zeros((nelm,gauss_point,1))
        stress_equivalent = np.zeros((nelm,gauss_point,1))
        strain_equivalent = np.zeros((nelm,gauss_point,1))
        elastic_plastic_strain_equivalent = np.zeros((nelm,gauss_point,1))
        k = 0
    # for i in range(0,1,10):
        #print("="*30)
        # plastic_strain = np.zeros((nelm,isoparametric_edge,1))
        loop = 0
        while True:
            loop += 1
            #print("="*30)
            #print("loop",loop)
            #print("="*30)
            # print("plastic strain", plastic_strain)
            k_ele,internal_force_matrix_ele = element_rotuine(radius,gauss_point)
            # print(k_ele)
            # internal_force_matrix_ele
            external_force_ele = np.zeros((nelm,isoparametric_edge*d_o_f,1))
            for i in range(nelm):
                external_force(f_ext,nelm,i,external_force_ele,radius)

            # print("external force",external_force_ele)
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

            
            # print("global stiffness",global_stiffness_matrix)
            G_matrix =  (global_internal_force_matrix - (global_external_force_matrix*tau))
            #print("global internal",global_internal_force_matrix)
            #print("global external",global_external_force_matrix*tau)
            # print("G matrix",G_matrix)

            reduced_global_stiffness_matrix_1 = global_stiffness_matrix[2:,2:]
            reduced_global_stiffness_matrix_2 = np.delete(reduced_global_stiffness_matrix_1,1,1)
            reduced_global_stiffness_matrix_3 = np.delete(reduced_global_stiffness_matrix_2,1,0)
            reduced_global_stiffness_matrix_4 = np.delete(reduced_global_stiffness_matrix_3,2,1)
            reduced_global_stiffness_matrix_5 = np.delete(reduced_global_stiffness_matrix_4,2,0)
            reduced_global_stiffness_matrix_6 = np.delete(reduced_global_stiffness_matrix_5,2,1)
            reduced_global_stiffness_matrix_7 = np.delete(reduced_global_stiffness_matrix_6,2,0)
            reduced_global_stiffness_matrix_8 = np.delete(reduced_global_stiffness_matrix_7,7,1)
            reduced_global_stiffness_matrix   = np.delete(reduced_global_stiffness_matrix_8,7,0)
            #print("reduced stiffness matrix",reduced_global_stiffness_matrix)

            reduced_G_matrix_1 = np.delete(G_matrix,0,0)
            reduced_G_matrix_2 = np.delete(reduced_G_matrix_1,0,0)
            reduced_G_matrix_3 = np.delete(reduced_G_matrix_2,1,0)
            reduced_G_matrix_4 = np.delete(reduced_G_matrix_3,2,0)
            reduced_G_matrix_5 = np.delete(reduced_G_matrix_4,2,0)
            reduced_G_matrix   = np.delete(reduced_G_matrix_5,7,0)
            #print("reduced g",reduced_G_matrix)
            delta_displacement = np.dot(np.linalg.inv(reduced_global_stiffness_matrix),(-reduced_G_matrix))
            # print("stiffness matrix",np.linalg.inv(reduced_global_stiffness_matrix))
            # print("delta displacement",delta_displacement)
            delta_displacement = np.insert(delta_displacement,0,0)
            delta_displacement = np.insert(delta_displacement,1,0)
            delta_displacement = np.insert(delta_displacement,3,0)
            delta_displacement = np.insert(delta_displacement,5,0)
            delta_displacement = np.insert(delta_displacement,6,0)
            delta_displacement = np.insert(delta_displacement,12,0)
            delta_displacement = delta_displacement.reshape(18,1)
            #print("delta displacement",delta_displacement)
            # print("global_before",global_displacement)
            global_displacement =  global_displacement + delta_displacement 
            #print("global_displacement",global_displacement)
            # print("delta displac",delta_displacement.shape)
            # print("reduced g matrix",reduced_G_matrix)
            # print(global_internal_force_matrix)
            b = np.array([])
            for j in summation:
                b = np.append(b,global_displacement[j])
            # print('i')
            global_displacement_ele = b.reshape((nelm,isoparametric_edge*d_o_f,1))
            
            # print(global_displacement_ele)
            # print(loop)
            initial_displacement = np.copy(global_displacement_ele)
            alpha = alpha_updated
            #print("alpha",alpha)
            # print("plastic strain",plastic_strain)
            # global_plastic_strain = plastic_strain
            # print("global plastic strain", global_plastic_strain)
            # print(initial_displacement)
            # global_plastic_strain = plastic_strain
            # print(delta_displacement)
            # print("delta displacement",np.linalg.norm(delta_displacement,np.inf))
            # print("global dispkacement",(0.005 * np.linalg.norm(global_displacement,np.inf)))
            # print("reduced G",(np.linalg.norm(reduced_G_matrix,np.inf)))
            # print("global internal", (0.005*np.linalg.norm(global_internal_force_matrix,np.inf)))
            # print("plastic strain",plastic_strain)
            if (np.linalg.norm(delta_displacement,np.inf) < (0.005 * np.linalg.norm(global_displacement,np.inf))) and ((np.linalg.norm(reduced_G_matrix,np.inf)) < (0.005*np.linalg.norm(global_internal_force_matrix,np.inf))):
                break
        global_plastic_strain = plastic_strain
        all_stress = np.append(all_stress,np.sum(stress_equivalent)/(nelm*gauss_point))
        all_strain = np.append(all_strain,np.sum(strain_equivalent)/(nelm*gauss_point))
        all_elastic_plastic_strain = np.append(all_elastic_plastic_strain,np.sum(elastic_plastic_strain_equivalent)/(nelm*gauss_point))

        # global_plastic_strain = np.random.random((8,1))



    #print("stress",all_stress)
    #print("strain",all_strain)

    #print("stress",all_stress.size)
    #print("strain",all_strain.size)
    #print("elastic_plastic",all_elastic_plastic_strain)

    # filename = "STRESS STRAIN.txt"

    # exp_strain = np.loadtxt(filename, usecols=(1))
    # exp_stress = np.loadtxt(filename, usecols=(0))
    # filename = "STRESS STRAIN.txt"
    # x = np.loadtxt(filename, usecols=(0))
    # y = np.loadtxt(filename, usecols=(1))
    # print(x.size)
    # print(y.size)
    # plt.plot(y,x)
    #plt.scatter(exp_strain,exp_stress)
    #plt.scatter(all_strain,all_stress)
    #plt.plot(all_elastic_plastic_strain,all_stress)
    #plt.show()
    return all_stress,all_strain