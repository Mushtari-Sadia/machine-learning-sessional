import matplotlib.colors as colors
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Ellipse
from GMM import *
import traceback

def create_cluster_animation(X, k_star,max_iter=1000):
    colorset = ['blue', 'red', 'black', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray']
    images = []
    while True:
        try:
            clusters = initialize(X, k_star)

            log_likelihood_list = []
            for i in range(max_iter):
                # print("Iteration: ", i)
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.scatter(X[:, 0], X[:, 1], c=colorset[0], marker='o')
                clusters = expectation(X, clusters)
                clusters = maximization(X, clusters)
                log_likelihood_list.append(log_likelihood(X, clusters))

                idx = 0

                for cluster in clusters:
                    mu = cluster['mu_k']
                    cov = cluster['cov_k']

                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                    vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
                    theta = np.arctan2(vy, vx)

                    color = colors.to_rgba(colorset[idx])

                    for cov_factor in range(1, 4):
                        ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2,
                                      height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=np.degrees(theta),
                                      linewidth=2)
                        ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
                        ax.add_artist(ell)

                    ax.scatter(cluster['mu_k'][0], cluster['mu_k'][1], c=colorset[idx], s=1000, marker='+')
                    idx += 1
                    # print("before canvas draw")
                fig.canvas.draw()
                plt.show()
                time.sleep(0.1)
                # print("after canvas draw")

                # print("showing plot...")

                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                images.append(image)

                try:
                    if log_likelihood_list[-1] - log_likelihood_list[-2] < 1e-6:
                        print("Converged at iteration ",i)
                        break
                except:
                    continue

            break
        except:
            print("mu_k caused error. reinitializing")
            continue
    return images

def create_cluster_animation_for_nD(X, k_star,V,M,trueX,max_iter=1000):
    colorset = ['blue', 'red', 'black', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray']
    images = []
    while True:
        try:
            clusters = initialize(trueX, k_star)

            log_likelihood_list = []
            for i in range(max_iter):
                # print("Iteration: ", i)
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.scatter(X[:, 0], X[:, 1], c=colorset[0], marker='o')
                clusters = expectation(trueX, clusters)
                clusters = maximization(trueX, clusters)
                log_likelihood_list.append(log_likelihood(trueX, clusters))

                idx = 0
                # Model
                # er
                # u = V.(u - M)
                # Model
                # er
                # sigma = V.sigma.V.T
                for cluster in clusters:
                    # print("V.shape",V.shape)
                    # print("M.shape",M.shape)
                    # print("cluster['mu_k'].shape",cluster['mu_k'].shape)
                    # print("cluster['cov_k'].shape",cluster['cov_k'].shape)

                    mu = np.dot(V,cluster['mu_k']-M)
                    cov = np.dot(np.dot(V,cluster['cov_k']),V.T)

                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                    vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
                    theta = np.arctan2(vy, vx)

                    color = colors.to_rgba(colorset[idx])

                    for cov_factor in range(1, 4):
                        ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2,
                                      height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=np.degrees(theta),
                                      linewidth=2)
                        ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
                        ax.add_artist(ell)

                    ax.scatter(cluster['mu_k'][0], cluster['mu_k'][1], c=colorset[idx], s=1000, marker='+')
                    idx += 1
                    # print("before canvas draw")
                fig.canvas.draw()
                plt.show()
                time.sleep(0.1)
                # print("after canvas draw")

                # print("showing plot...")

                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                images.append(image)

                try:
                    if log_likelihood_list[-1] - log_likelihood_list[-2] < 1e-6:
                        print("Converged at iteration ",i)
                        break
                except:
                    continue

            break
        except:
            traceback.print_exc()
            print("mu_k caused error. reinitializing")
            continue
    return images
