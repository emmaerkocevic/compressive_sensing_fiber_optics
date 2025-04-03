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
    #     # Validate inputs
    #     if L < 0:
    #         raise ValueError("L must be non-negative")
    #     if s * L > n:
    #         raise ValueError(f"s * L = {s * L} exceeds the size of the vector n = {n}")

    #     # Special case: L = 0 (no spacing constraint)
    #     if L == 0:
    #         pos = np.random.choice(np.arange(n), s, replace=False)
    #         pos.sort()
    #     else:
    #         # Generate initial positions ensuring no overlap
    #         available_positions = np.arange(0, n - (L * (s - 1)))
    #         selected_positions = np.random.choice(available_positions, s, replace=False)
    #         selected_positions.sort()

    #         # Add spacing
    #         pos = selected_positions + np.arange(0, s * L, L)

    #         # Ensure indices are within bounds
    #         pos = pos[pos < n]

    #     # Ensure indices are integers
    #     pos = pos.astype(int)  # Convert to integer type for indexing

    #     # Create sparse vector
    #     xt = np.zeros(n)
    #     xt[pos] = np.random.normal(0, 1, len(pos))  # Use len(pos) in case some positions were trimmed
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
    # sigma = 0.001 * np.linalg.norm(Aop * xt, ord=2)

    # # reconstruction and error
    # xinv = spgl1_pylops(A=Aop, b=y, sigma=sigma, iter_lim=5000)[0]
    # # xinv = spgl1_pylops(A=Aop, b=y, sigma=sigma, iter_lim=5000, project=spgl1.norm_l1nn_project)[0]
    # error = np.linalg.norm(xinv - xt, ord=2) / np.linalg.norm(xt, ord=2)
    return error
    

def compute_results(A, n, sparsity_levels, n_runs, L):
    p_success = []

    # Calculate s_max for the given n and L
    # s_max = (n - L) // (L + 1)

    for s in sparsity_levels:
        # If s exceeds s_max, assign p_success = 0
        # if s > s_max:
        #     print(f"Skipping s={s}, L={L}: Cannot fit within n={n}. Setting p_success to 0.")
        #     p_success.append(0.0)  # Probability of success is 0
        #     continue

        # calculate errors for n_runs at sparsity level s
        # try:
            errors = np.array(Parallel(n_jobs=4)(
                delayed(reconstruct)(A, n, s, L) for _ in range(n_runs)))
            p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction
        # except ValueError as e:
        #     print(f"Error during reconstruction with s={s}, L={L}: {e}")
        #     p_success.append(0.0)  # Failure for this configuration

    return p_success



np.random.seed(0)

# parameters
m = 100
n = 200
s = 100
sparsity_levels = range(1, s)
# sparsity_levels = np.linspace(1,s,10).astype(int)
lengths = [0, 1, 2, 3, 4]  # 0 represents random Gaussian
n_runs = 100

# Main loop for processing
p_success_dict = {}
for L in lengths:
    if L == 0:  # uncorrelated case
        # sparsity_levels = range(1, s + 1)  # Include all sparsity levels up to s
        A = np.random.randn(m, n)
    else:  # correlated case
        # s_max = (n + L) // (L + 1)
        # sparsity_levels = range(1, s + 1)  # Include all sparsity levels up to s
        mean = np.zeros(n)
        ind = np.arange(n)
        cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
        A = np.random.multivariate_normal(mean, cov, m)
    # Aop = pylops.MatrixMult(A)
    p_success_dict[L] = compute_results(A, n, sparsity_levels, n_runs, L)

# save results
np.savez('sparsity_results_cvx_2.npz', lengths=lengths, sparsity_levels=sparsity_levels, p_success_dict=p_success_dict)