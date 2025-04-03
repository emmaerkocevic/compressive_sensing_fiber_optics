import numpy as np
from joblib import Parallel, delayed
import pylops
from pylops.optimization.sparsity import spgl1 as spgl1_pylops
import spgl1
import cvxpy as cp


# reconstructs a single s-sparse vector
def reconstruct(A, n, s, L):
    # ground truth    
    xt = np.zeros(n)
    pos = np.random.choice(np.arange(n), s, replace=False)
    xt[pos] = np.random.normal(0, 1, s)
    
    # xt[pos] = np.abs(np.random.normal(0, 1, s))

    # def generate_sparse_signal(n, s, L):
    #     # validate inputs
    #     if L < 0:
    #         raise ValueError("L must be non-negative")
    #     if s * L > n:
    #         raise ValueError(f"s * L = {s * L} exceeds the size of the vector n = {n}")
        
    #     # special case: L = 0 (no spacing constraint)
    #     if L == 0:
    #         pos = np.random.choice(np.arange(n), s, replace=False)
    #         pos.sort()
    #     else:
    #         # generate initial positions ensuring no overlap
    #         available_positions = np.arange(0, n - (L * (s - 1)))
    #         selected_positions = np.random.choice(available_positions, s, replace=False)
    #         selected_positions.sort()
    
    #         # add spacing
    #         pos = selected_positions + np.arange(0, s * L, L)
    
    #         # ensure indices are within bounds
    #         pos = pos[pos < n]
    
    #     # ensure indices are integers
    #     pos = pos.astype(int)  # convert to integer type for indexing
    
    #     # create sparse vector
    #     xt = np.zeros(n)
    #     xt[pos] = np.random.normal(0, 1, len(pos))  # use len(pos) in case some positions were trimmed
    #     return xt, pos
    
    # xt, pos = generate_sparse_signal(n, s, L)
    
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

    # # measurements
    # y = Aop * xt
    # sigma = 0.001 * np.linalg.norm(Aop * xt)

    # # reconstruction and error
    # xinv = spgl1_pylops(A=Aop, b=y, sigma=sigma, iter_lim=5000)[0]
    # # xinv = spgl1_pylops(A=Aop, b=y, sigma=sigma, iter_lim=5000, project=spgl1.norm_l1nn_project)[0]
    # error = np.linalg.norm(xinv - xt, ord=2) / np.linalg.norm(xt, ord=2)
    return error


# computes probabilities of success for different number of measurements
def compute_results(n_measurements, n, s, L, n_runs):
    p_success = []

    for m in n_measurements:

        # generate A
        if L == 0:  # uncorrelated case
            A = np.random.randn(m, n)
        else:  # correlated case
            mean = np.zeros(n)
            ind = np.arange(n)
            cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
            A = np.random.multivariate_normal(mean, cov, m)
        # Aop = pylops.MatrixMult(A)

        # calculate errors for n_runs
        errors = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct)(A, n, s, L) for _ in range(n_runs)))

        p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction

    return p_success


np.random.seed(0)

# parameters
m = 200
# n_measurements = np.linspace(10,m,10).astype(int)
n_measurements = range(10, m)  # number of measurements across which to test
n = 200
s = 20
lengths = [0, 1, 2, 3, 4]  # 0 represents random Gaussian
n_runs = 100

p_success_dict = {L: compute_results(n_measurements, n, s, L, n_runs) for L in lengths}

np.savez('measurements_results_cvx_2.npz', lengths=lengths, n_measurements=n_measurements, p_success_dict=p_success_dict)