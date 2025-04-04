import numpy as np
from joblib import Parallel, delayed
import cvxpy as cp

# constructs n^2 x n^2 covariance matrix based on 2D RBF kernel
def cov_2d(n, L):

    # create grid indices
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y)  # coordinate grids

    # flatten coordinates to get all (x, y) positions
    coords = np.column_stack((xx.ravel(), yy.ravel()))

    # compute pairwise squared Euclidean distances using broadcasting
    diff = coords[:, None, :] - coords[None, :, :]  # shape (n^2, n^2, 2)
    dist_sq = np.sum(diff ** 2, axis=-1)  # sum over x and y differences -> shape (n^2, n^2)

    cov = np.exp(-dist_sq / (L ** 2))

    return cov

# reconstructs a single s-sparse vector with CVX
def reconstruct(A, n, s, L):
    
    # ground truth    
    xt = np.zeros(n*n)
    pos = np.random.choice(np.arange(n*n), s, replace=False)
    xt[pos] = np.abs(np.random.normal(0, 1, s))

    y = A @ xt # measurements
    x = cp.Variable(n*n)
    sigma = 0.001 * cp.norm(A @ xt, 2)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.norm(A @ x - y, 2) <= sigma, x>=0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    error = np.linalg.norm(x.value - xt, ord=2) / np.linalg.norm(xt, ord=2)
    
    return error

# computes probabilities of success for different number of measurements
def compute_results(n_measurements, n, s, L, n_runs):
    p_success = []

    for m in n_measurements:

        baseline_A = np.random.randn(m, n*n)        

        # generate A
        if L == 0:  # uncorrelated case
            A = baseline_A
        else:  # correlated case
            cov = cov_2d(n, L)
            cov_stable = cov + np.eye(n*n) * 0.0001
            cov_sqrt = np.linalg.cholesky(cov_stable)
            A = baseline_A @ cov_sqrt.T

        # calculate errors for n_runs
        errors = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct)(A, n, s, L) for _ in range(n_runs)))

        p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction

    return p_success

np.random.seed(0)

# parameters
m = 400
n_measurements = range(10, m)  # number of measurements across which to test
n = 20
s = 40
lengths = [0, 1, np.sqrt(2), 2, np.sqrt(5), 2*np.sqrt(2), 3, np.sqrt(10), np.sqrt(13)]  # 0 represents random Gaussian
n_runs = 100

p_success_dict = {L: compute_results(n_measurements, n, s, L, n_runs) for L in lengths}

np.savez('measurements_2D.npz', lengths=lengths, n_measurements=n_measurements, p_success_dict=p_success_dict)

### Plot results ###

import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.labelsize': 12, 
    'axes.titlesize': 14,  
    'xtick.labelsize': 12,  
    'ytick.labelsize': 12,  
    'legend.fontsize': 12,  
    'font.size': 12  
})

lengths = [0, 1, np.sqrt(2), 2, np.sqrt(5), 2 * np.sqrt(2), 3, np.sqrt(10), np.sqrt(13)]

color_map = {
    0: 'blue',            
    1: 'orange',          
    np.sqrt(2): 'brown',  
    2: 'dodgerblue',           
    np.sqrt(5): 'green',   
    2 * np.sqrt(2): 'limegreen',  
    3: 'gold',             
    np.sqrt(10): 'purple',      
    np.sqrt(13): 'red' 
}

sqrt_labels = {
    np.sqrt(2): r'\sqrt{2}',
    np.sqrt(5): r'\sqrt{5}',
    2 * np.sqrt(2): r'2\sqrt{2}',
    np.sqrt(10): r'\sqrt{10}',
    np.sqrt(13): r'\sqrt{13}'
}

# format labels correctly
def format_label(L):
    if L in sqrt_labels:
        return rf"$L = {sqrt_labels[L]}$"  
    else:
        return rf"$L = {int(L)}$"
        
# load measurement results
data = np.load('measurements_results_2D_cvx.npz', allow_pickle=True)
n_measurements = data['n_measurements']
p_success_dict = data['p_success_dict'].item()

# plot probability vs. number of measurements
plt.figure()
for key in lengths:
    plt.plot(n_measurements, np.ravel(p_success_dict[key]), label=format_label(key), color=color_map[key])
plt.legend()
plt.xlabel('$m$')
plt.ylabel('Probability of successful reconstruction')
plt.title(r'$n\times n=20\times20$, $s=40$')

plt.show()
