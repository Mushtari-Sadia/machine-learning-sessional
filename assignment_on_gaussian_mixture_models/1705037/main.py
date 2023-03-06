from GMM import *
from plot import *
import warnings
warnings.filterwarnings("ignore")

#load data
data = []
with open('Assignment 3 Materials/data6D.txt') as f:
    for line in f.readlines():
        data.append([float(x) for x in line.split()])
# print(data)
X = np.array(data)

# As the number of components (or, the number of gaussian distributions, k)
# is usually unknown, you will assume a range for k. For example, from 1 to 10.

converged_log_likelihood = []
num_clusters = 10
for k in range(1,num_clusters+1):
    clusters,log_likelihood_val = EM(X, k)
    converged_log_likelihood.append(log_likelihood_val)
    print("k = "+str(k)+" : converged log likelihood = ", log_likelihood_val)
    print()

print("converged_log_likelihood = ",converged_log_likelihood)

#log likelihood plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
ax.set_title('Converged log likelihood vs k')
ax.plot(np.arange(1, num_clusters+1), converged_log_likelihood)
ax.scatter(np.arange(1, num_clusters+1), converged_log_likelihood)
ax.set_xlabel('Clusters')
ax.set_xticks(np.arange(1, num_clusters+1))
ax.set_ylabel('Converged log likelihood')
# plt.savefig('converged log likelihood vs clusters.png')
plt.show()


