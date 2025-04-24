import numpy as np
from joblib import Parallel, delayed
import cvxpy as cp

'''numerical experiment for the effect of 1D smoothing in the measurement matrix across varying measurement numbers
(Fig. 3.4 in thesis), including the option to constrain signal classes (Fig. 3.7 and Fig. 3.9)'''

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
    #     xt[pos] = np.random.normal(0, 1, len(pos))  # use len(pos) in case some positions were trimmed
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

# computes probabilities of success for different number of measurements
def compute_results(n_measurements, N, s, L, n_runs):
    p_success = []

    for m in n_measurements:

        # generate A
        if L == 0:  # uncorrelated case
            A = np.random.randn(m, N)
        else:  # correlated case
            mean = np.zeros(N)
            ind = np.arange(N)
            cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
            A = np.random.multivariate_normal(mean, cov, m)

        # calculate errors for n_runs
        errors = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct)(A, N, s, L) for _ in range(n_runs)))

        p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction

    return p_success

np.random.seed(0)

# parameters
m = 200
n_measurements = range(10, m)  # number of measurements across which to test
N = 200
s = 20
lengths = [0, 1, 2, 3, 4]  # 0 represents random Gaussian
n_runs = 100

p_success_dict = {L: compute_results(n_measurements, N, s, L, n_runs) for L in lengths}

np.savez('measurements.npz', lengths=lengths, n_measurements=n_measurements, p_success_dict=p_success_dict)
