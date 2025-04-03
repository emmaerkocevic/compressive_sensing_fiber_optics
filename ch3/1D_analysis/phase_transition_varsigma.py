import numpy as np
from joblib import Parallel, delayed
import pylops
from pylops.optimization.sparsity import spgl1
from collections import defaultdict
import cvxpy as cp


# reconstructs a single s-sparse vector
def reconstruct(A, xt):

    # # measurements
    # y = Aop * xt
    # sigma = 0.001 * np.linalg.norm(Aop * xt)

    # # reconstruction and error
    # xinv = spgl1(A=Aop, b=y, sigma=sigma, iter_lim=5000)[0]
    # error = np.linalg.norm(xinv - xt, ord=2) / np.linalg.norm(xt, ord=2)
    
    # CVX
    y = A @ xt # measurements
    x = cp.Variable(n)
    sigma = 0.001 * cp.norm(A @ xt, 2)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.norm(A @ x - y, 2) <= sigma]
    # constraints = [cp.norm(A @ x - y, 2) <= sigma, x>=0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    error = np.linalg.norm(x.value - xt, ord=2) / np.linalg.norm(xt, ord=2)

    return error


# computes probabilities of success for a given combination of m, n, s
def run_experiment(m, n, s, L, n_runs):
    
    # generate A
    if L==0:  # uncorrelated case
        A = np.random.randn(m, n)
    else:  # correlated case
        mean = np.zeros(n)
        ind = np.arange(n)
        cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
        A = np.random.multivariate_normal(mean, cov, m)
    # Aop = pylops.MatrixMult(A)

    # calculate errors for n_runs
    errors = []
    for _ in range(n_runs):
        # ground truth
        xt = np.zeros(n)
        pos = np.random.choice(np.arange(n), s, replace=False)
        xt[pos] = np.random.normal(0, 1, s)

        errors.append(reconstruct(A, xt))

    p_success = np.mean(np.array(errors) <= 0.05)  # probability of successful reconstruction
    delta = m / n
    rho = s / m

    return delta, rho, p_success


def compute_results(n_measurements, n, L, n_runs):
    
    # define tasks for parallel execution over m and s
    tasks = []
    for m in n_measurements:
        delta = m / n
        sparsity_levels = range(5, m)
        for s in sparsity_levels:
            rho = s / m
            tasks.append((m, n, s, n_runs))

    # run all tasks in parallel
    results = Parallel(n_jobs=4)(
        delayed(run_experiment)(m, n, s, L, n_runs) for m, n, s, n_runs in tasks
    )

    # # group (rho, p_success) by delta for p_success in [0.4, 0.6]
    # delta_dict = defaultdict(list)
    # for delta, rho, p_success in results:
    #     if 0.4 <= p_success <= 0.6:
    #         delta_dict[delta].append((rho, p_success))
    
    # # compute averages for each delta, with 0 if no valid points exist
    # avg_results = [
    #     (
    #         delta, 
    #         np.mean([rho for rho, _ in delta_dict[delta]]) if delta_dict[delta] else 0,
    #         np.mean([p_success for _, p_success in delta_dict[delta]]) if delta_dict[delta] else 0
    #     )
    #     for delta in sorted(set(delta for delta, _, _ in results))
    # ]

            
    # return results, avg_results
    return results


np.random.seed(0)

# parameters
n = 200
n_measurements = [m for m in range(10, 201, 10)]
n_runs = 100
lengths = [0, 1, 2]

# main loop
results_dict = {}
# avg_results_dict = {}
for L in lengths:
    # results, avg_results = compute_results(n_measurements, n, L, n_runs)
    results = compute_results(n_measurements, n, L, n_runs)
    results_dict[L] = results
    # avg_results_dict[L] = avg_results

# save results
np.savez('phase_transition_results_varsigma_cvx.npz', lengths=lengths, results_dict=results_dict)
# np.savez('phase_transition_avg_results1.npz', lengths=lengths, avg_results_dict=avg_results_dict)