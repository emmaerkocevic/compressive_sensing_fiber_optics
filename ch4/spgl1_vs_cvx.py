import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pylops
from pylops.optimization.sparsity import spgl1 as spgl1_pylops
import cvxpy as cp

'''comparison of reconstructions with spgl1 versus cvxpy (Fig. 4.6 in thesis)'''

def reconstruct_spgl1(Aop, n, s, noise):
    xt = np.zeros(n)
    pos = np.random.choice(np.arange(n), s, replace=False)  # positions of non-zeros
    xt[pos] = np.abs(np.random.normal(0, 1, s))  # ground truth

    y = Aop * xt + noise  # measurements
    sigma = np.linalg.norm(noise, ord=2)  # noise level
    xinv = spgl1_pylops(A=Aop, b=y, sigma=sigma)[0]  # solve with SPGL1
    error = np.linalg.norm(xinv - xt, ord=2) / np.linalg.norm(xt, ord=2)
    return error

def reconstruct_cvx(A, n, s, noise):
    xt = np.zeros(n)
    pos = np.random.choice(np.arange(n), s, replace=False)  # positions of non-zeros
    xt[pos] = np.abs(np.random.normal(0, 1, s))  # ground truth

    y = A @ xt + noise # measurements
    x = cp.Variable(n)
    sigma = cp.norm(noise, 2) # noise level
    objective = cp.Minimize(cp.norm1(x))
    constraints = [cp.norm(A @ x - y, 2) <= sigma]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    error = np.linalg.norm(x.value - xt, ord=2) / np.linalg.norm(xt, ord=2)
    return error

def compute_results(n_measurements, n, s, n_runs, mean):
    nmse_spgl1, nmse_cvx = [], []  # normalized mean-square error
    
    for m in n_measurements:
        A = np.random.normal(loc=mean, scale=1, size=(m, n))  # measurement matrix with N(0,1) or N(20,1) i.i.d. entries
        Aop = pylops.MatrixMult(A)
        
        variance_dB = -20  # noise variance in dB
        variance_linear = 10 ** (variance_dB / 10)  # convert dB to linear scale
        noise = np.random.normal(loc=0, scale=np.sqrt(variance_linear), size=m)
        
        errors_spgl1 = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct_spgl1)(Aop, n, s, noise) for _ in range(n_runs)))
        errors_cvx = np.array(Parallel(n_jobs=4)(
            delayed(reconstruct_cvx)(A, n, s, noise) for _ in range(n_runs)))
        
        nmse_spgl1.append(np.mean(errors_spgl1))
        nmse_cvx.append(np.mean(errors_cvx))
    
    return nmse_spgl1, nmse_cvx


# parameters
np.random.seed(0)
n_measurements = range(20, 85, 5)  # different numbers of measurements
n, s, n_runs = 100, 5, 100  # signal length, sparsity, and number of runs

# compute results for both N(0,1) and N(20,1) matrices
nmse_mean0_spgl1, nmse_mean0_cvx = compute_results(n_measurements, n, s, n_runs, mean=0)
nmse_mean20_spgl1, nmse_mean20_cvx = compute_results(n_measurements, n, s, n_runs, mean=20)

# save results
np.savez('results_mean0.npz', n_measurements=n_measurements, nmse_spgl1=nmse_mean0_spgl1, nmse_cvx=nmse_mean0_cvx)
np.savez('results_mean20.npz', n_measurements=n_measurements, nmse_spgl1=nmse_mean20_spgl1, nmse_cvx=nmse_mean20_cvx)

# plot results
plt.figure()
plt.plot(n_measurements, nmse_mean0_cvx, label='N(0,1) CVX', color='red')
plt.plot(n_measurements, nmse_mean0_spgl1, label='N(0,1) SPGL1', color='red', linestyle='dashed')
plt.plot(n_measurements, nmse_mean20_cvx, label='N(20,1) CVX', color='blue')
plt.plot(n_measurements, nmse_mean20_spgl1, label='N(20,1) SPGL1', color='blue', linestyle='dashed')
plt.xlim(20, 80)
plt.legend()
plt.xlabel('$m$')
plt.ylabel('NMSE')
plt.title(r'$n=100, s=5$')
plt.show()
