import numpy as np
from joblib import Parallel, delayed
import scipy.stats as stats
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from scipy.interpolate import interp1d

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

# reconstructs a single s-sparse vector
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

# computes probabilities of success for different number of measurements
def compute_results(n_measurements, n, s, L, n_runs, dist_type):
    p_success = []
    
    if L > 0:
        cov = cov_2d(n, L)
        cov_stable = cov + np.eye(n * n) * 0.0001
        cov_sqrt = np.linalg.cholesky(cov_stable)
    else:
        cov_sqrt = None  # No smoothing for L=0

    for m in n_measurements:
        baseline_A = np.random.randn(m, n * n)  # N(0,I)
        smoothed_A = baseline_A @ cov_sqrt.T if L > 0 else baseline_A
        u = stats.norm.cdf(smoothed_A)  # Unif[0,1]
        
        if dist_type == 'mean-zero normal':
            A = smoothed_A
        elif dist_type == 'exponential':
            A = stats.expon.ppf(u, scale=1)
        elif dist_type == 'uniform':
            A = np.sqrt(12) * u
        elif dist_type == 'triangular':
            A = stats.triang.ppf(u, c=1, loc=0, scale=np.sqrt(18))
        elif dist_type == 'shifted normal':
            A = stats.norm.ppf(u, loc=3, scale=1)
            # A[A < 0] = np.random.normal(loc=3, scale=1, size=np.sum(A < 0))
        elif dist_type == 'bimodal':
            # parameters for bimodal distribution
            d = (np.sqrt(2) + 2) / 2
            c = np.sqrt((4 - d ** 2) / d ** 2)
            mu1, mu2, sigma = 1.5, 1.5 + d, c * d / 2
            A = bimodal_ppf(u, mu1, mu2, sigma)

        errors = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct)(A, n, s, L) for _ in range(n_runs)))

        p_success.append(np.mean(errors <= 0.05))

    return p_success

np.random.seed(0)

# Parameters
m = 400
n_measurements = range(10, m)
# n_measurements = range(10, m, 20)
n = 20
s = 40
n_runs = 100
lengths = [1]
distributions = ['mean-zero normal', 'exponential', 'uniform', 'triangular', 'shifted normal', 'bimodal']

# Dictionary to store results
results = {}

# Compute and store results
for L in lengths:
    results[L] = {}
    for dist in distributions:
        results[L][dist] = compute_results(n_measurements, n, s, L, n_runs, dist)

# Save all results in a single file
np.savez('measurements_results_nonneg_3apr.npz', 
         n_measurements=n_measurements, 
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
data = np.load('measurements_results_nonneg_3apr.npz', allow_pickle=True)

n_measurements = data['n_measurements']
lengths = data['lengths']
distributions = data['distributions']
results = data['results'].item()  # Convert to dictionary

# Plot results
plt.figure()
for i, dist in enumerate(distributions):
    plt.plot(n_measurements, results[lengths[0]][dist], label=dist, color=colors[i])
plt.xlabel('$m$')
plt.ylabel('Probability of successful reconstruction')
plt.title(r'$n\times n=20\times 20$, $s=40$, $L=1$')
plt.legend()
plt.show()
