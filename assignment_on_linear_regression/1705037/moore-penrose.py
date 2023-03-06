import numpy as np
from numpy import random
np.set_printoptions(precision=3)

#Take the dimensions of matrix n, m as input.
n = int(input("Input n "))
m = int(input("Input m "))

#Produce a random n x m matrix A. For the purpose of demonstrating, every cell of A must be an integer.
A = random.randint(100, size=(n,m))

#Perform Singular Value Decomposition using NumPy’s library function
U,D,V_T = np.linalg.svd(A)
D = np.diag(D) #np.linalg.svd returns a diagonal 1D array for D. We have to convert it to a square array first 

#Calculate the Moore-Penrose Pseudoinverse using NumPy’s library function
A_pinv_lib = np.linalg.pinv(A)
print("A_pinv_lib\n",A_pinv_lib)

#Calculate the Moore-Penrose Pseudoinverse again using Eq. 2.47
D_pinv = np.zeros(A.shape) #According to the book, the dimension of D_pinv is nxm

for i in range(D.shape[0]):
  for j in range(D.shape[1]):
    if D[i,j]!=0:
      D_pinv[i,j] = 1/D[i,j] #the pseudoinverse D+ of a diagonal matrix D is obtained by taking the reciprocal of its nonzero elements

D_pinv = np.matrix.transpose(D_pinv) #then taking the transpose of the resulting matrix

#Eqn 2.47 - A_pinv = V * D+ * U_T
V = np.matrix.transpose(V_T)
U_T = np.matrix.transpose(U)

A_pinv = np.dot(V,D_pinv) # V * D+
A_pinv = np.dot(A_pinv,U_T) # V * D+ * U_T
print("A_pinv\n",A_pinv)


#Check if these two inverses are equal (np.allclose will come in handy)
if np.allclose(A_pinv,A_pinv_lib):
  print("Equal")
else:
  print("Not Equal")