import numpy as np
from joblib import Parallel, delayed
import pylops
from pylops.optimization.sparsity import spgl1 as spgl1_pylops
import spgl1
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


# reconstructs a single s-sparse vector
def reconstruct(A, n, s, L):
    # ground truth
    xt = np.zeros(n*n)
    pos = np.random.choice(np.arange(n*n), s, replace=False)
    xt[pos] = np.abs(np.random.normal(0, 1, s))
    # xt[pos] = np.random.normal(0, 1, s)
    
    # CVX
    y = A @ xt # measurements
    x = cp.Variable(n*n)
    sigma = 0.001 * cp.norm(A @ xt, 2)
    objective = cp.Minimize(cp.norm(x, 1))
    # constraints = [cp.norm(A @ x - y, 2) <= sigma]
    constraints = [cp.norm(A @ x - y, 2) <= sigma, x>=0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    error = np.linalg.norm(x.value - xt, ord=2) / np.linalg.norm(xt, ord=2)

    # # measurements
    # y = Aop * xt
    # sigma = 0.001 * np.linalg.norm(Aop * xt)

    # # reconstruction and error
    # xinv = spgl1_pylops(A=Aop, b=y, sigma=sigma, iter_lim=5000)[0]
    # # xinv = spgl1_pylops(A=Aop, b=y, sigma=0.1, iter_lim=2000)[0]
    # error = np.linalg.norm(xinv - xt, ord=2) / np.linalg.norm(xt, ord=2)
    return error
    

def compute_results(Aop, n, sparsity_levels, n_runs, L):
    p_success = []

    for s in sparsity_levels:
        errors = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct)(A, n, s, L) for _ in range(n_runs)))
        p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction

    return p_success


np.random.seed(0)

# parameters
m = 200 #200
n = 20 #20
s = 200 #200
sparsity_levels = range(1, s)
lengths = [0, 1, np.sqrt(2), 2, np.sqrt(5), 2*np.sqrt(2), 3, np.sqrt(10), np.sqrt(13)]  # 0 represents random Gaussian
# lengths = [0, np.sqrt(13)]  # 0 represents random Gaussian
n_runs = 100

# Main loop for processing
p_success_dict = {}
baseline_A = np.random.randn(m, n*n)
for L in lengths:
    if L == 0:  # uncorrelated case
        A = baseline_A
    else:  # correlated case
        cov = cov_2d(n, L)
        cov_stable = cov + np.eye(n*n) * 0.0001
        cov_sqrt = np.linalg.cholesky(cov_stable)
        A = baseline_A @ cov_sqrt.T
    # Aop = pylops.MatrixMult(A)
    p_success_dict[L] = compute_results(A, n, sparsity_levels, n_runs, L)

# save results
np.savez('sparsity_results_2D_cvx.npz', lengths=lengths, sparsity_levels=sparsity_levels, p_success_dict=p_success_dict)