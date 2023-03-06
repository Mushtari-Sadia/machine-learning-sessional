import numpy as np
from numpy import random
np.set_printoptions(precision=3)
n = int(input("Input the dimension of nxn matrix "))

#generating invertible nxn matrix
A = random.randint(100, size=(n, n))
while np.linalg.det(A)==0:
  A = random.randint(100, size=(n, n))

print("original matrix:\n",A,"\n\n")

#eigen decomposition
eigen_values, eigen_vectors = np.linalg.eig(A)
# print("eigen values",eigen_values)
print("eigen vectors",eigen_vectors)

rec_A = np.dot(eigen_vectors,np.diag(eigen_values)) #V*diag(lambda)
rec_A = np.dot(rec_A,np.linalg.inv(eigen_vectors)) #V*diag(lambda)*V^-1
print("reconstructed matrix:\n",rec_A,"\n\n")

if np.allclose(A,rec_A):
  print("Successfully reconstructed")
else:
  print("Reconstruction unsuccessful")

