import numpy as np
from joblib import Parallel, delayed
import cvxpy as cp
import scipy.stats as stats
from scipy.stats import expon, uniform, triang, norm
from scipy.interpolate import interp1d

'''numerical experiment for different intensity distributions across varying sparsity levels (Fig. 4.5 in thesis)'''

# bimodal distribution functions
def bimodal_pdf(x, loc1, loc2, scale):
    return 0.5 * norm.pdf(x, loc1, scale) + 0.5 * norm.pdf(x, loc2, scale)

def bimodal_cdf(x, loc1, loc2, scale):
    return 0.5 * norm.cdf(x, loc1, scale) + 0.5 * norm.cdf(x, loc2, scale)

def bimodal_ppf(q, loc1, loc2, scale, num_points=1000):
    x_vals = np.linspace(0, 5, num_points)
    ppf_interp = interp1d(bimodal_cdf(x_vals, loc1, loc2, scale), x_vals, fill_value="extrapolate")
    return ppf_interp(np.clip(q, 0, 1))

# constructs n^2 x n^2 covariance matrix based on 2D RBF kernel
def cov_2d(n, L):
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    cov = np.exp(-dist_sq / (L ** 2))
    return cov

# reconstructs a single s-sparse vector using CVX
def reconstruct(A, n, s, L):
    xt = np.zeros(n * n)
    pos = np.random.choice(np.arange(n * n), s, replace=False)
    xt[pos] = np.abs(np.random.normal(0, 1, s))

    y = A @ xt
    x = cp.Variable(n * n)
    sigma = 0.000001 * cp.norm(A @ xt)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [cp.norm(A @ x - y, 2) <= sigma, x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    error = np.linalg.norm(x.value - xt) / np.linalg.norm(xt)
    return error

# computes probabilities of success for different sparsity levels
def compute_results(A, n, sparsity_levels, n_runs, L):
    p_success = []
    for s in sparsity_levels:
        errors = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct)(A, n, s, L) for _ in range(n_runs)))
        p_success.append(np.mean(errors <= 0.05))  # probability of successful reconstruction
    return p_success

np.random.seed(0)

# parameters
m = 200
n = 20
s = 200
sparsity_levels = range(1, s)
#sparsity_levels = range(1, s, 20)
lengths = [1]
n_runs = 100
distributions = ['mean-zero normal', 'exponential', 'uniform', 'triangular', 'shifted normal', 'bimodal']

# main loop: compute results
results = {}
for L in lengths:
    results[L] = {}
    
    # generate covariance matrix if L > 0
    if L > 0:
        cov = cov_2d(n, L)
        cov_stable = cov + np.eye(n * n) * 0.0001
        cov_sqrt = np.linalg.cholesky(cov_stable)
    else:
        cov_sqrt = None  # no smoothing

    for dist in distributions:
        # generate measurement matrix based on distribution
        baseline_A = np.random.randn(m, n * n)  # N(0,I)
        smoothed_A = baseline_A @ cov_sqrt.T if L > 0 else baseline_A
        u = stats.norm.cdf(smoothed_A)  # Unif[0,1]

        if dist == 'mean-zero normal':
            A = smoothed_A
        elif dist == 'exponential':
            A = stats.expon.ppf(u, scale=1)
        elif dist == 'uniform':
            A = np.sqrt(12) * u
        elif dist == 'triangular':
            A = stats.triang.ppf(u, c=1, loc=0, scale=np.sqrt(18))
        elif dist == 'shifted normal':
            A = stats.norm.ppf(u, loc=3, scale=1)
        elif dist == 'bimodal':
            # parameters for bimodal distribution
            d = (np.sqrt(2) + 2) / 2
            c = np.sqrt((4 - d ** 2) / d ** 2)
            mu1, mu2, sigma = 1.5, 1.5 + d, c * d / 2
            A = bimodal_ppf(u, mu1, mu2, sigma)

        # compute success probabilities
        results[L][dist] = compute_results(A, n, sparsity_levels, n_runs, L)

# save results
np.savez('sparsity_results_nonneg.npz',
         sparsity_levels=sparsity_levels,
         lengths=lengths,
         distributions=distributions,
         results=results)

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

colors = ['black', 'blue', 'orange', 'green', 'red', 'purple']

# Load results
data = np.load('sparsity_results_nonneg_3apr.npz', allow_pickle=True)
sparsity_levels = data['sparsity_levels']
distributions = data['distributions']
results = data['results'].item()  # Convert to dictionary

# Plot results
plt.figure()
for i, dist in enumerate(distributions):
    plt.plot(sparsity_levels, results[1][dist], label=dist, color=colors[i])
plt.xlabel('$s$')
plt.ylabel('Probability of successful reconstruction')
plt.title(r'$m=200$, $n\times n=20\times 20$, $L=1$')
plt.legend()
plt.show()
