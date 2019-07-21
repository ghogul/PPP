import numpy as np

a=np.matrix([[-1,0,1],[1,0,1],[0,1,0]])
b=np.matrix([[4,0,0],[0,1,0],[0,0,2]])
c=np.matrix([[-1,0,1],[0,0,1],[1,1,-1]])
print(np.matmul(c,b,a))




