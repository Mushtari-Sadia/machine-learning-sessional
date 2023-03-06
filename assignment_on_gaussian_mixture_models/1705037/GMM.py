import numpy as np
import traceback
from scipy.stats import multivariate_normal
def initialize(X, k):
    clusters = []
    mu_k = np.random.uniform(low=-10.0, high=10.0, size=(k,X.shape[1]))

    for i in range(k):
        clusters.append({
            'pi_k': 1.0 / k,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })

    return clusters

def expectation(X, clusters):
    global sum_of_clusters
    sum_of_clusters = np.zeros(X.shape[0], dtype=np.float64)
    for cluster in clusters:
        pi = cluster['pi_k']
        mu = cluster['mu_k']
        cov = cluster['cov_k']
        cluster['gamma_nk'] = pi * multivariate_normal.pdf(X, mu, cov)
        sum_of_clusters += cluster['gamma_nk']

    for cluster in clusters:
        cluster['gamma_nk'] /= sum_of_clusters
        cluster['gamma_nk'] = np.expand_dims(cluster['gamma_nk'], 1)
    return clusters

def maximization(X, clusters):
    N = X.shape[0]
    for cluster in clusters:
        N_k = np.sum(cluster['gamma_nk'])
        cluster['pi_k'] = N_k / N
        cluster['mu_k'] = np.sum(cluster['gamma_nk'] * X, axis=0) / N_k
        cluster['cov_k'] = np.dot((cluster['gamma_nk'] * (X - cluster['mu_k'])).T, X - cluster['mu_k']) / N_k
    return clusters

def log_likelihood(X, clusters):
    return np.sum(np.log(sum_of_clusters))

def EM(X, k, max_iter=1000,animation=False):
    while True:
        try:
            clusters = initialize(X, k)

            log_likelihood_list = []

            for i in range(max_iter):
                clusters = expectation(X, clusters)
                clusters = maximization(X, clusters)
                log_likelihood_list.append(log_likelihood(X, clusters))


                try:
                    if log_likelihood_list[-1] - log_likelihood_list[-2] < 1e-6:
                        print("Converged at iteration ",i)
                        break
                except:
                    continue

            break
        except:
            # if animation:
            #     traceback.print_exc()
            continue
    return clusters,log_likelihood_list[-1]