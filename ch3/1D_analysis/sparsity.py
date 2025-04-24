import numpy as np
from joblib import Parallel, delayed
import cvxpy as cp

'''numerical experiment for the effect of 1D smoothing in the measurement matrix across varying sparsity levels
(Fig. 3.5 in thesis), including the option to constrain signal classes (Fig. 3.8 and Fig. 3.10)'''

# reconstructs a single s-sparse vector with CVX
def reconstruct(A, N, s, L):
    # ground truth
    xt = np.zeros(N)
    pos = np.random.choice(np.arange(N), s, replace=False)
    xt[pos] = np.random.normal(0, 1, s)
    
    # xt[pos] = np.abs(np.random.normal(0, 1, s))  # nonnegativity constraint

    # separation distance constraint
    # def generate_sparse_signal(N, s, L):

    #     if L == 0:
    #         pos = np.random.choice(np.arange(N), s, replace=False)
    #         pos.sort()
    #     else:
    #         # generate initial positions ensuring no overlap
    #         available_positions = np.arange(0, N - (L * (s - 1)))
    #         selected_positions = np.random.choice(available_positions, s, replace=False)
    #         selected_positions.sort()

    #         # add spacing
    #         pos = selected_positions + np.arange(0, s * L, L)

    #         # ensure indices are within bounds and integers
    #         pos = pos[pos < N]
    #         pos = pos.astype(int)

    #     # create sparse vector
    #     xt = np.zeros(N)
    #     xt[pos] = np.random.normal(0, 1, len(pos))
    #     return xt

    # xt = generate_sparse_signal(N, s, L)
    
    y = A @ xt # measurements
    x = cp.Variable(N)
    sigma = 0.001 * cp.norm(A @ xt, 2)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.norm(A @ x - y, 2) <= sigma]
    # constraints = [cp.norm(A @ x - y, 2) <= sigma, x>=0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    error = np.linalg.norm(x.value - xt, ord=2) / np.linalg.norm(xt, ord=2)
    
    return error
    

def compute_results(A, N, sparsity_levels, n_runs, L):
    p_success = []

    # calculate s_max for the given N and L
    # s_max = (N - L) // (L + 1)

    for s in sparsity_levels:
        # if s exceeds s_max, assign p_success = 0
        # if s > s_max:
        #     p_success.append(0.0)
        #     continue

        # calculate errors for n_runs at sparsity level s
        # try:
            errors = np.array(Parallel(n_jobs=4)(
                delayed(reconstruct)(A, N, s, L) for _ in range(n_runs)))
            p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction
        # except ValueError as e:
        #     p_success.append(0.0)  # failure for this configuration

    return p_success

np.random.seed(0)

# parameters
m = 100
N = 200
s = 100
sparsity_levels = range(1, s)
lengths = [0, 1, 2, 3, 4]  # 0 represents random Gaussian
n_runs = 100

# main loop for processing
p_success_dict = {}
for L in lengths:
    if L == 0:  # uncorrelated case
        A = np.random.randn(m, N)
    else:  # correlated case
        mean = np.zeros(N)
        ind = np.arange(N)
        cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
        A = np.random.multivariate_normal(mean, cov, m)
    p_success_dict[L] = compute_results(A, N, sparsity_levels, n_runs, L)

# save results
np.savez('sparsity.npz', lengths=lengths, sparsity_levels=sparsity_levels, p_success_dict=p_success_dict)
