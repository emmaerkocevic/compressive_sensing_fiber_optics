import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
import cvxpy as cp

'''numerical experiment for the effect of 1D smoothing in the measurement matrix, simultanously varying 
measurement numbers and sparsity levels (Fig. 3.6 in thesis)'''

# reconstructs a single s-sparse vector with CVX
def reconstruct(A, xt):
    
    y = A @ xt # measurements
    x = cp.Variable(N)
    sigma = 0.001 * cp.norm(A @ xt, 2)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.norm(A @ x - y, 2) <= sigma]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    error = np.linalg.norm(x.value - xt, ord=2) / np.linalg.norm(xt, ord=2)

    return error

# computes probabilities of success for a given combination of m, N, s
def run_experiment(m, N, s, L, n_runs):
    
    # generate A
    if L==0:  # uncorrelated case
        A = np.random.randn(m, N)
    else:  # correlated case
        mean = np.zeros(N)
        ind = np.arange(N)
        cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
        A = np.random.multivariate_normal(mean, cov, m)

    # calculate errors for n_runs
    errors = []
    for _ in range(n_runs):
        # ground truth
        xt = np.zeros(N)
        pos = np.random.choice(np.arange(N), s, replace=False)
        xt[pos] = np.random.normal(0, 1, s)

        errors.append(reconstruct(A, xt))

    p_success = np.mean(np.array(errors) <= 0.05)  # probability of successful reconstruction
    delta = m / N
    rho = s / m

    return delta, rho, p_success

def compute_results(n_measurements, N, L, n_runs):
    
    # define tasks for parallel execution over m and s
    tasks = []
    for m in n_measurements:
        delta = m / N
        sparsity_levels = range(5, m)
        for s in sparsity_levels:
            rho = s / m
            tasks.append((m, N, s, n_runs))

    # run all tasks in parallel
    results = Parallel(n_jobs=4)(
        delayed(run_experiment)(m, N, s, L, n_runs) for m, N, s, n_runs in tasks
    )

    return results

np.random.seed(0)

# parameters
N = 200
n_measurements = [m for m in range(10, 201, 10)]
n_runs = 100
lengths = [0, 1, 2]

# main loop
results_dict = {}
for L in lengths:
    results = compute_results(n_measurements, n, L, n_runs)
    results_dict[L] = results

# save results
np.savez('phase_transition.npz', lengths=lengths, results_dict=results_dict)
