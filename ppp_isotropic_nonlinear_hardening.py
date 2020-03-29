import numpy as np
import matplotlib.pyplot as plt

def nonlinear_fem(youngs_modulus,yield_stress,hardening_modulus,delta_y,eta):

    '''
    ==================================================================
                            INITIAL PARAMETERS
    ..................................................................
    youngs_modulus : youngs_modulus of the material
    yield_stress : yield stress of the material
    hardening_modulus : Linear hardening modulus
    delta_y : Non-linear hardening modulus
    eta : Hardening exponent 
    ..................................................................
    return:
    all_stress : equivalent stress in each iteration
    all_strain : equivalent strain in each iteration
    ==================================================================
    '''
    '''***MATERIAL PARAMETER***'''
    
    poission_ratio = 0.3
    
    '''***GEOMETRICAL PARAMETER***'''
    length = 50  #mm
    radius = 0.5   #mm
    
    '''***TIME PARAMETERS***'''
    tau = 0
    time_step = 0.01
    total_time = 1
    

    '''***EXTERNAL LOADING***'''
    f_ext = 75000  #Mpa
    

    '''***lame's constant***'''
    mu = (youngs_modulus/(2*(1+poission_ratio)))
    lamda = (poission_ratio*youngs_modulus)/((1+poission_ratio)*(1-2*poission_ratio))
    bulk = youngs_modulus/(3*(1-(2*poission_ratio)))


    '''***DISCRETIZATION PARAMETERS***'''
    nelm = 4
    isoparametric_edge = 4
    d_o_f = 2
    no_nodes = int((np.sqrt(nelm)+1)*(np.sqrt(nelm)+1))
    

    '''***graphical representation of elements***'''
    nelm_length = np.linspace(0,length,np.sqrt(nelm)+1)
    nelm_radius = np.linspace(0,radius,np.sqrt(nelm)+1)
    yy,xx = np.meshgrid(nelm_length,nelm_radius)

    


    '''***shape function for isoparametric element***'''
    
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
        ======================================================================
                        element coordinates
        ......................................................................
        nelm : Number of elements
        nelm_length : element length  Z-direction in axisymmetric axis
        nelm_radius : element radius  r-direction in axisymmetric axis
        ......................................................................
        return:
        all_ele_coord : it returns coordinates of all elements
        ======================================================================
        '''
        all_ele_coord = np.zeros((nelm,4,2))    # it stores all element coordinates in one array
        loop = 0
        for j in range(int(np.sqrt(nelm))):
            for i in range(int(np.sqrt(nelm))):
                ele_coord = np.matrix(([[nelm_radius[i],nelm_length[j]],
                            [nelm_radius[i+1],nelm_length[j]],                  # ele_coord : coordinated of each element
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

       

    '''
    ================================================================================================================================================================================================
                                                            material rotuine
    ================================================================================================================================================================================================
    '''

    def material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,j,global_plastic_strain,yield_stress,mu,lamda,bulk,alpha):
        '''
        ======================================================================
                        material routine
        ......................................................................
        poission ratio : poission ration of the material
        youngs modulus : youngs modulus of the material
        B_matrix : strain-displacement matrix
        initial_displacement : initial displacement to find strain
        i : assigning the value of the variable according to the gauss-point
        j : assigning the value of the variable according to the gauss-point
        global_plastic_strain : global plastic strain is to calculate the strain 
        yield_stress : yield stress of the material 
        mu : lames constant
        lamda : lames constant
        bulk : lames constant
        alpha : hardening varibale
        ......................................................................
        return:
        C : stress-strain matrix of the elastic part
        C_t : algoritm consistent tangent stress-strain matrix of the plastic part
        stress : equavilant stress of that time
        ======================================================================
        '''

        #
        #
        # First, calculate the strain by giving the displacement
        # 
        # 
        
        epsilon = np.dot(B_matrix , initial_displacement[i])
        strain = epsilon
        strain_deviatoric = np.copy(strain)
        strain_deviatoric[:3] = (strain[:3] - ((1/3)*strain[1]+strain[2]+strain[3]))
        strain_equivalent[i,j] = (np.sqrt(((3/2) * np.dot(strain_deviatoric.T,strain_deviatoric)))).item()

        #
        # 
        # calculating stress --strain relationship matrix to find stress for the computed strain
        # to check weather the stress is below the yield or above
        # 
        c_1 = (youngs_modulus)/((1+poission_ratio)*(1-2*poission_ratio))
        C = c_1*np.array([[1-poission_ratio,poission_ratio,poission_ratio,0],
                        [poission_ratio,1-poission_ratio,poission_ratio,0],
                        [poission_ratio,poission_ratio,1-poission_ratio,0],
                        [0,0,0,((1-2*poission_ratio)/2)]])

        #
        # 
        #  calculating the elastic strain and its equivalent to find the trial stress
        # 
        # 
        elastic_strain = strain - global_plastic_strain[i,j]
        elastic_strain_deviatoric = np.copy(elastic_strain)
        elastic_strain_deviatoric[:3] = (strain[:3] - ((1/3)*strain[1]+strain[2]+strain[3]))
        elastic_strain_equivalent = (np.sqrt(((3/2) * np.dot(elastic_strain_deviatoric.T,elastic_strain_deviatoric)))).item()

        # 
        #  calculating the trial stress 
        #  n : vector normal to find the direction
        #  trial stress equivalent : finding the equivalent stress 
        #  Beta : exponential function for Non-linear hardening
        #  phi : condition to check elastic or plastic region   
        # 
        
        trial_stress = np.dot(C,elastic_strain)
        trial_stress_deviatoric = np.copy(trial_stress)
        trial_stress_deviatoric[:3] = trial_stress[:3] - ((1/3)*(trial_stress[0]+trial_stress[1]+trial_stress[2]))
        
        n = np.divide(trial_stress_deviatoric,np.linalg.norm(trial_stress_deviatoric),out=np.zeros_like(trial_stress_deviatoric),where=np.linalg.norm(trial_stress_deviatoric)!=0)
        
        trial_stress_equivalent = (np.sqrt(((3/2) * np.dot(trial_stress_deviatoric.T,trial_stress_deviatoric)))).item()
        beta = (hardening_modulus*alpha[i,j]) + delta_y*(1-np.exp(-(eta)*alpha[i,j]))
        phi = np.linalg.norm(trial_stress_deviatoric) - (np.sqrt(2/3)*(yield_stress + beta))
        
        if phi <= 0:
            # phi is less than or zero means its in the elastic region and returns stress strain relationship matrix
            
            stress_equivalent[i,j] = trial_stress_equivalent
            elastic_plastic_strain_equivalent[i,j] = elastic_strain_equivalent
            
            return C, trial_stress

        else:
            #  phi is greater than zero then its in the plastic region  and returns C_t algorithmic tangent stress-strain matrix
            #  delta_lamda : plastic multiplier
            
            delta_lamda = ((phi)/((2*mu)+((2/3)*hardening_modulus)))
            # 
            # 
            # plastic corrector
            # here computing plastic strain 
            # and stress for the respected strain 
            # 
           
            plastic_strain[i,j] = global_plastic_strain[i,j] + (delta_lamda * n) 
            current_plastic = plastic_strain[i,j]
            plastic_strain_deviatoric = np.copy(plastic_strain[i,j])
            plastic_strain_deviatoric[:3] = (current_plastic[:3] - ((1/3)*current_plastic[1]+current_plastic[2]+current_plastic[3]))
            elastic_plastic_strain_equivalent[i,j] = (np.sqrt(((3/2) * np.dot(plastic_strain_deviatoric.T,plastic_strain_deviatoric)))).item()
            current_stress = C*(strain - plastic_strain[i,j])
            current_stress_deviatoric = np.copy(current_stress)
            current_stress_deviatoric[:3] = current_stress[:3] - ((1/3)*(current_stress[0]+current_stress[1]+current_stress[2]))
            current_stress_equivalent = (np.sqrt ((3/2)*(np.dot(current_stress_deviatoric.T,current_stress_deviatoric)))).item()
            stress_equivalent[i,j] = current_stress_equivalent
            
            # 
            # here hardening variable is apdated for the current plastic strain 
            # 
            
            alpha_updated[i,j] =  (np.sqrt(((3/2) * np.dot(plastic_strain_deviatoric.T,plastic_strain_deviatoric)))).item()

            # 
            #  computing fourth order identity tensor and algorithmic tangent stress -strain matrix
            # 
            
            beta_1 = 1-((np.divide(phi,np.linalg.norm(trial_stress_deviatoric),out=np.zeros_like(trial_stress_deviatoric),where=np.linalg.norm(trial_stress_deviatoric)!=0))*(1/(1+(hardening_modulus/3*mu))))
            beta_2 = (1-(np.divide(phi,np.linalg.norm(trial_stress_deviatoric),out=np.zeros_like(trial_stress_deviatoric),where=np.linalg.norm(trial_stress_deviatoric)!=0)))*(1/(1+(hardening_modulus/3*mu)))
            
            identity = np.ones((4,4))
            identity[3,0] = identity[3,1] = identity[3,2] = identity[3,3] = identity[0,3] = identity[1,3] = identity[2,3] = 0
            identity_deviatoric = ((1/2)*(np.eye(4)+np.eye(4)))-((1/3)*np.ones((4,4)))
            identity_deviatoric[3,0] = identity_deviatoric[3,1] = identity_deviatoric[3,2] = identity_deviatoric[0,3] = identity_deviatoric[1,3] = identity_deviatoric[2,3] = 0
            identity_deviatoric[3,3] = 0.5
            n_tensor = np.dot(n,n.T)
            c_t_1st = bulk * identity
            c_t_2nd = 2*mu*beta_1*identity_deviatoric
            c_t_3rd = 2*mu*beta_2*n_tensor
            c_t_deviatoric = c_t_2nd - c_t_3rd
            
            C_t = c_t_deviatoric + c_t_1st

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
        '''
        ======================================================================
                        element routine
        ......................................................................
        radius : radius of the specimen 
        gauss_point : number of gauss points here gauss point is 4
        ......................................................................
        return:
        k_all_ele : stiffness matrix of all elements
        internal_force_matrix_all_ele : internal matrix of all elements
        ======================================================================
        '''

        # 
        # internal_force_matrix_all_ele : storing all internal force matrix in an array
        # k_all_ele : storing all stiffness matrix in an array
        # 
        internal_force_matrix_all_ele = np.zeros((nelm,isoparametric_edge*2,1))
        k_all_ele = np.zeros((nelm,isoparametric_edge*2,isoparametric_edge*2))
        all_ele_coord = elements_coordinates(nelm,nelm_length,nelm_radius)
        for i in range(nelm):
            
            internal_force_matrix = 0
            k_all = np.zeros((isoparametric_edge*2,isoparametric_edge*2))
            for j in range(gauss_point):
                
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
                    
                derivative_N = 1/4*np.matrix([[-(1-E2),(1-E2),(1+E2),-(1+E2)],    # derivative of the shape function 
                                            [-(1-E1),-(1+E1),(1+E1),(1-E1)]])
                
                x_y_ele = all_ele_coord[i]                                        # coordinates of the each element
                jacobi_1 = derivative_N*x_y_ele                                   # Jacobian of the each element 
                jacobi_inverse = np.linalg.inv(jacobi_1)
                B_1_matrix = jacobi_inverse*derivative_N
                ele_radius = (x_y_ele[0,0]+x_y_ele[1,0]+x_y_ele[2,0]+x_y_ele[3,0])/isoparametric_edge
                area = all_ele_coord[0,1,0]*all_ele_coord[0,2,1]    
                weight = 1
                # 
                # computing the strain-displacement matrix
                # 
                B_matrix = np.matrix([[B_1_matrix[0,0],0,B_1_matrix[0,1],0,B_1_matrix[0,2],0,B_1_matrix[0,3],0],
                                    [0,B_1_matrix[1,0],0,B_1_matrix[1,1],0,B_1_matrix[1,2],0,B_1_matrix[1,3]],
                                    [((1-E1)*(1-E2)/4)/ele_radius,0,((1+E1)*(1-E2)/4)/ele_radius,0,((1+E1)*(1+E2)/4)/ele_radius,0,((1-E1)*(1+E2)/4)/ele_radius,0],
                                    [B_1_matrix[1,0],B_1_matrix[0,0],B_1_matrix[1,1],B_1_matrix[0,1],B_1_matrix[1,2],B_1_matrix[0,2],B_1_matrix[1,3],B_1_matrix[0,3]]])
                
                # 
                # calling matrial rotuine 
                # 
                c_matrix,stress_matrix = material_rotuine(poission_ratio, youngs_modulus,B_matrix,initial_displacement,i,j,global_plastic_strain,yield_stress,mu,lamda,bulk,alpha)
                # 
                #  computing element stiffness matrix
                # 
                k_1 = 2*np.pi*ele_radius*area
                k_2 = np.dot(c_matrix,B_matrix)
                k_3 = np.dot(np.transpose(B_matrix),k_2)
                k_all = k_all + weight*weight*k_1*k_3*np.linalg.det(jacobi_1)
                # 
                #  computing element element internal force matrix
                # 

                internal_force_matrix = internal_force_matrix + weight*weight*2*np.pi*ele_radius*area*np.dot(np.transpose(B_matrix),stress_matrix)*np.linalg.det(jacobi_1)
            # 
            # 
            #  Assigning all elment stiffness matrix and internal force matrix to an single three dimensional array
            # 
            # 
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
    
    def external_force(f_ext,nelm,i,external_force_ele,radius):
        '''
        ======================================================================
                        external force
        ......................................................................
        f_ext : external force acting on the specimen 
        nelm :  number of elements
        i : iteration number to store the value in an array
        external_force_ele : stores the external force acting on each element
        radius : radius of the specimen 
        ......................................................................
        return:
        external force : external force acting on each element
        ======================================================================
        '''
        # 
        #  External force will act on elements which has directly contact with external load
        # 
        if i < int(nelm-(np.sqrt(nelm))):
            external_force_ele[i] = (np.zeros((isoparametric_edge*2,1)))
            return (np.zeros((isoparametric_edge*2,1)))

        else:
            
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
        '''
        ======================================================================
                        assignment matrix
        ......................................................................
        nelm :  number of elements
        isoparametric_edge : rectangle element has four edge
        d_o_f : degree of freedom of each node
        ......................................................................
        return:
        all_a : its an array stores the assignment matrix for all elements
        summation : its just a flattened array of all elements coordinates
        ======================================================================
        '''
        # 
        #  Defining number of elements in rows and columns
        # 
        x_cells = int(np.sqrt(nelm))
        y_cells = int(np.sqrt(nelm))
        elements = np.arange((x_cells+1)*(y_cells+1)).reshape(y_cells+1,x_cells+1)
        
        # 
        #  ID : stores all the coordinates of each element 
        # 

        ID = ([])
        for i in range(y_cells):
            for j in range(x_cells):
                id = np.matrix([elements[i,j],elements[i,j+1],elements[i+1,j],elements[i+1,j+1]])
                ID = np.append(ID,id)

        # 
        #  summation : Each node has two degrees of freedom so each node has two number 
        # 
        summation = ([])  
        for i in ID:
            multi = np.array([(i*2),((i*2)+1)])
            summation = np.append(summation,multi)
        summation = summation.astype(int)
        a_column = summation.flatten().reshape(nelm,isoparametric_edge*d_o_f)

        # 
        #  all_a : stores all the element assignment matrix (a.T*k*a) a matrix has ones of corresponding nodes of the element
        # 
        
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
    # 
    #  Calling assignment matrix function for assembly
    # 
    
    
    all_a,summation = assignment_matrix(nelm,isoparametric_edge,d_o_f)
    # 
    #  global displacement : initiating displacement to start
    #  global_plastic-strain : initiating plastic strain to start
    # 

    global_displacement = np.zeros((no_nodes*d_o_f,1)) 
    global_plastic_strain = np.zeros((nelm,gauss_point,isoparametric_edge,1))

    # 
    #  splitting the global displacements according to elements
    # 
    G_matrix = 0
    a = np.array([])
    for i in summation:
        a = np.append(a,global_displacement[i])
    initial_displacement = a.reshape((nelm,isoparametric_edge*d_o_f,1))

    # 
    #  initiating array to store all stress and strain taht is computed for each displacement 
    # 
    all_stress = np.array([])
    all_strain = np.array([])
    all_elastic_plastic_strain = np.array([])
    
    # 
    #  initiating time loop 
    # 
    
    alpha = np.zeros((nelm,gauss_point,1))
    time = 0
    while True:
        tau = tau + time_step
        time += 1
        # 
        #  loop breaking condition
        # 
        if tau > (total_time-time_step):
            break

        # 
        #   initiating these array to store the values in each iteration and update these variables in the next iteration
        #   plastic strain : An array stores all plastic strain of converged iteration
        #   alpha updated : An array stores hardening variable 
        #   stress_equivalent : An array stores all equivalent stress 
        #   strain_equivalent : An array stores all equivalent strain 
        # 
        plastic_strain = np.zeros((nelm,gauss_point,isoparametric_edge,1))
        alpha_updated = np.zeros((nelm,gauss_point,1))
        stress_equivalent = np.zeros((nelm,gauss_point,1))
        strain_equivalent = np.zeros((nelm,gauss_point,1))
        elastic_plastic_strain_equivalent = np.zeros((nelm,gauss_point,1))
        # k = 0
        # 
        # 
        #   initiating loop to find displacements
        # 
        # 
        loop = 0
        while True:
            loop += 1
            # 
            #  Calling element routine it will return stiffness matrix and internal force matrix
            # 
            
            k_ele,internal_force_matrix_ele = element_rotuine(radius,gauss_point)
            # 
            #  Calling external force function 
            #  external_force_ele : an array stores external force acting on each element
            # 
            external_force_ele = np.zeros((nelm,isoparametric_edge*d_o_f,1))
            for i in range(nelm):
                external_force(f_ext,nelm,i,external_force_ele,radius)

            # 
            #   calling assignment matrix for assembly
            # 
            all_a,summation = assignment_matrix(nelm,isoparametric_edge,d_o_f)
            # 
            #   Assembly of all elements stiffness matrix, internal force matrix and external force matrix
            #   global_stiffness_matrix : assembly of each element stiffness matrix
            #   global_internal_force_matrix :  assembly of each element internal force matrix
            #   global_external_force_matrix : assembly of each element external force matrix
            # 
            global_stiffness_matrix = np.zeros((no_nodes*2,no_nodes*2))
            global_internal_force_matrix = 0
            global_external_force_matrix = 0
            for i in range(nelm):
                assembly_1 = np.dot((np.transpose(all_a[i])),k_ele[i])
                assembly = np.dot(assembly_1,all_a[i])
                global_stiffness_matrix = global_stiffness_matrix + assembly
                global_internal_force_matrix = global_internal_force_matrix + np.dot((np.transpose(all_a[i])),internal_force_matrix_ele[i])
                global_external_force_matrix = global_external_force_matrix + np.dot((np.transpose(all_a[i])),external_force_ele[i])

            # 
            #  G_matrix : residual matrix (difference between internal and external force)
            # 
            G_matrix =  (global_internal_force_matrix - (global_external_force_matrix*tau))

            # 
            # 
            #  Applying  boundary conditions 
            #  reducing the global stiffess matrix acoording to boundary condition
            # 
            reduced_global_stiffness_matrix_1 = global_stiffness_matrix[2:,2:]
            reduced_global_stiffness_matrix_2 = np.delete(reduced_global_stiffness_matrix_1,1,1)
            reduced_global_stiffness_matrix_3 = np.delete(reduced_global_stiffness_matrix_2,1,0)
            reduced_global_stiffness_matrix_4 = np.delete(reduced_global_stiffness_matrix_3,2,1)
            reduced_global_stiffness_matrix_5 = np.delete(reduced_global_stiffness_matrix_4,2,0)
            reduced_global_stiffness_matrix_6 = np.delete(reduced_global_stiffness_matrix_5,2,1)
            reduced_global_stiffness_matrix_7 = np.delete(reduced_global_stiffness_matrix_6,2,0)
            reduced_global_stiffness_matrix_8 = np.delete(reduced_global_stiffness_matrix_7,7,1)
            reduced_global_stiffness_matrix   = np.delete(reduced_global_stiffness_matrix_8,7,0)
            # 
            # 
            #  Applying  boundary conditions 
            #  reducing the G matrix acoording to boundary condition
            # 

            reduced_G_matrix_1 = np.delete(G_matrix,0,0)
            reduced_G_matrix_2 = np.delete(reduced_G_matrix_1,0,0)
            reduced_G_matrix_3 = np.delete(reduced_G_matrix_2,1,0)
            reduced_G_matrix_4 = np.delete(reduced_G_matrix_3,2,0)
            reduced_G_matrix_5 = np.delete(reduced_G_matrix_4,2,0)
            reduced_G_matrix   = np.delete(reduced_G_matrix_5,7,0)

            # 
            #  newton raphson scheme  F = K.U   U = K^-1.G_matrix
            # 
            
            delta_displacement = np.dot(np.linalg.inv(reduced_global_stiffness_matrix),(-reduced_G_matrix))

            # 
            #  After computing the displacement inserting zeros in to the known displacements
            # 
            
            delta_displacement = np.insert(delta_displacement,0,0)
            delta_displacement = np.insert(delta_displacement,1,0)
            delta_displacement = np.insert(delta_displacement,3,0)
            delta_displacement = np.insert(delta_displacement,5,0)
            delta_displacement = np.insert(delta_displacement,6,0)
            delta_displacement = np.insert(delta_displacement,12,0)
            delta_displacement = delta_displacement.reshape(18,1)

            # 
            # 
            #   Total displacement : periviously computed displacement + current displacement of the stress  
            # 
            # 
            global_displacement =  global_displacement + delta_displacement 
            # 
            # 
            #   splitting the global displacement into element wise displaceent for next time step
            # 
            # 
            b = np.array([])
            for j in summation:
                b = np.append(b,global_displacement[j])
            
            global_displacement_ele = b.reshape((nelm,isoparametric_edge*d_o_f,1))
            
            # 
            # 
            #  displacements and hardening variables are updated for the next time step
            # 
            # 
            
            initial_displacement = np.copy(global_displacement_ele)
            alpha = alpha_updated

            # 
            #  
            #   Warnings if the loop exceeds more than 1000 iteration warning message will print in the terminal
            #   1000 is the maximum iteration because during algorithm computed parameters some parameters may in negative, in that case it will take more iterations   
            #   than normal iteration
            
            if loop > 1000:
                print(".!"*30)
                print("Guess is too large or small !!!")
                print("Please guess the parameters which is close to the real solution.")
                print(".!"*30)
                exit()
            # 
            # 
            #  Termination criteria for the Newton-Raphson scheme 
            # 
            # 
            if (np.linalg.norm(delta_displacement,np.inf) < (0.005 * np.linalg.norm(global_displacement,np.inf))) and ((np.linalg.norm(reduced_G_matrix,np.inf)) < (0.005*np.linalg.norm(global_internal_force_matrix,np.inf))):
                break
        # 
        # 
        #   if the plastic region is initiated plastic strain will update for the next time step
        #   all_stress : storing stress for the each time step for stress strain curve
        #   all_strain : storing strain for the each time step for stress strain curve
        global_plastic_strain = plastic_strain
        all_stress = np.append(all_stress,np.sum(stress_equivalent)/(nelm*gauss_point))
        all_strain = np.append(all_strain,np.sum(strain_equivalent)/(nelm*gauss_point))
        all_elastic_plastic_strain = np.append(all_elastic_plastic_strain,np.sum(elastic_plastic_strain_equivalent)/(nelm*gauss_point))

        
    return all_stress,all_strain