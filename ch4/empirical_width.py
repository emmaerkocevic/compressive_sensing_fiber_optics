import numpy as np
import matplotlib.pyplot as plt

'''greedy approximation of the empirical width for varying measurement numbers and sparsity levels (Fig. 4.2 in thesis'''

plt.rcParams.update({
    'axes.labelsize': 12,
    'axes.titlesize': 14,  
    'xtick.labelsize': 12,  
    'ytick.labelsize': 12,  
    'legend.fontsize': 12,  
    'font.size': 12 
})

# generates rademacher sequence
def rademacher(size):
    return np.random.choice([-1, 1], size=size)

def compute_supremum(m, n, s, distribution, num_trials=100):
    
    suprema = []

    for _ in range(num_trials):
        
        # generate exponential or bernoulli matrix
        if distribution == 'exponential':
            A = np.random.exponential(scale=0.5, size=(m, n * n))
        elif distribution == 'bernoulli':
            A = np.random.binomial(1, 0.5, size=(m, n * n))

        # compute Rademacher-weighted sum
        epsilon = rademacher(m)[:, np.newaxis]
        h = (1 / np.sqrt(m)) * np.sum(epsilon * A, axis=0)

        # find optimal u and compute <h,u_opt>
        top_s_indices = np.argsort(np.abs(h))[-s:]
        u_opt = np.zeros(n * n)
        u_opt[top_s_indices] = np.sign(h[top_s_indices])
        u_opt /= np.linalg.norm(u_opt, 2)  # normalize
        suprema.append(np.dot(h, u_opt))

    return np.mean(suprema)

# parameters
n = 10
s_fixed = 10
m_fixed = 50
s_values = range(1, 100)
m_values = range(1, 100)

# empirical width across sparsity levels
suprema_s_exp = [compute_supremum(m_fixed, n, s, distribution='exponential') for s in s_values]
suprema_s_ber = [compute_supremum(m_fixed, n, s, distribution='bernoulli') for s in s_values]

# empirical width across number of measurements
suprema_m_exp = [compute_supremum(m, n, s_fixed, distribution='exponential') for m in m_values]
suprema_m_ber = [compute_supremum(m, n, s_fixed, distribution='bernoulli') for m in m_values]

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(m_values, suprema_m_exp, color='red', label='Exp(1/2)')
axs[0].plot(m_values, suprema_m_ber, color='blue', label='Ber(1/2)')
axs[0].set_xlabel("m")
axs[0].set_ylabel(r"$\hat{W}_m(\mathbf{\Sigma}_s,\mathbf{a})$")
axs[0].set_title(r"$s=10$, $n=100$")
axs[0].legend()

axs[1].plot(s_values, suprema_s_exp, color='red', label='Exp(1/2)')
axs[1].plot(s_values, suprema_s_ber, color='blue', label='Ber(1/2)')
axs[1].set_xlabel("s")
axs[1].set_ylabel(r"$\hat{W}_m(\mathbf{\Sigma}_s,\mathbf{a})$")
axs[1].set_title(r"$m=50$, $n=100$")
axs[1].legend()

plt.tight_layout()
plt.show()
