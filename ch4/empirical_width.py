import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'axes.labelsize': 12,
    'axes.titlesize': 14,  
    'xtick.labelsize': 12,  
    'ytick.labelsize': 12,  
    'legend.fontsize': 12,  
    'font.size': 12 
})


def rademacher(size):
    return np.random.choice([-1, 1], size=size)


def compute_supremum(m, n, s, distribution, num_trials=100):
    suprema = []

    for _ in range(num_trials):
        if distribution == 'exponential':
            A = np.random.exponential(scale=2, size=(m, n * n))
        elif distribution == 'bernoulli':
            A = np.random.binomial(1, 0.5, size=(m, n * n))

        epsilon = rademacher(m)[:, np.newaxis]
        Z = (1 / np.sqrt(m)) * np.sum(epsilon * A, axis=0)

        top_s_indices = np.argsort(np.abs(Z))[-s:]
        u_opt = np.zeros(n * n)
        u_opt[top_s_indices] = np.sign(Z[top_s_indices])
        suprema.append(np.dot(Z, u_opt))

    return np.mean(suprema)


# Parameters
n = 10
s_fixed = 10
m_fixed = 50
s_values = range(1, 100)
m_values = range(1, 100)

# 1. Empirical width vs. sparsity level
suprema_s_exp = [compute_supremum(m_fixed, n, s, distribution='exponential') for s in s_values]
suprema_s_ber = [compute_supremum(m_fixed, n, s, distribution='bernoulli') for s in s_values]

# 2. Empirical width vs. number of measurements
suprema_m_exp = [compute_supremum(m, n, s_fixed, distribution='exponential') for m in m_values]
suprema_m_ber = [compute_supremum(m, n, s_fixed, distribution='bernoulli') for m in m_values]

# Plotting both plots side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: width vs. number of measurements
axs[0].plot(m_values, suprema_m_exp, color='red', label='Exp(2)')
axs[0].plot(m_values, suprema_m_ber, color='blue', label='Ber(1/2)')
axs[0].set_xlabel("m")
axs[0].set_ylabel(r"$\hat{W}_m(\mathbf{\Sigma}_s,\mathbf{a})$")
axs[0].set_title(r"$s=10$, $n=100$")
axs[0].legend()

# Right plot: width vs. sparsity level
axs[1].plot(s_values, suprema_s_exp, color='red', label='Exp(2)')
axs[1].plot(s_values, suprema_s_ber, color='blue', label='Ber(1/2)')
axs[1].set_xlabel("s")
axs[1].set_ylabel(r"$\hat{W}_m(\mathbf{\Sigma}_s,\mathbf{a})$")
axs[1].set_title(r"$m=50$, $n=100$")
axs[1].legend()

plt.tight_layout()
plt.show()
