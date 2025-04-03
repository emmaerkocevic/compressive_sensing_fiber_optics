import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.labelsize': 12,  # Axis labels
    'axes.titlesize': 14,  # Title
    'xtick.labelsize': 12,  # X-axis tick labels
    'ytick.labelsize': 12,  # Y-axis tick labels
    'legend.fontsize': 12,  # Legend
    'font.size': 12  # General font size
})


# ### BPDN guarantee constants ###
# delta_2s = np.linspace(0, 1, 1000)
# C_0 = 2 * (1 - (1 - np.sqrt(2)) * delta_2s) / (1 - (1 + np.sqrt(2)) * delta_2s)
# C_1 = 4 * np.sqrt(1 + delta_2s) / (1 - (1 + np.sqrt(2)) * delta_2s)

# plt.figure()
# plt.plot(delta_2s, C_0, label='$C_0$', color='blue')
# plt.plot(delta_2s, C_1, label='$C_1$', color='red')
# plt.axvline(np.sqrt(2)-1, label=r'$\sqrt{2}-1$', color='black')
# plt.xlim(0, 1)
# plt.ylim(-500, 500)
# plt.xlabel('$\delta_{2s}$')
# plt.legend()
# plt.show()



### smoothing kernel ###
# distances = np.arange(-10, 11, 1)  # i-j
# lengths = [1, 2, 3, 4]
# colors = ['blue', 'orange', 'green', 'red', 'purple']

# # plot kernel for each L
# plt.figure(figsize=(6.4, 5.4))
# for L in lengths:
#     kernel = np.exp(-distances**2 / L**2)
#     plt.plot(distances, kernel, label=f'L={L}', color=colors[L])
# plt.axhline(np.exp(-1), label=r'$e^{-1}$', color='black')
# plt.xticks(np.arange(-10, 11, 2))
# plt.title(r'$\Sigma_{ij} = \exp\left(-(i-j)^2 / L^2\right)$')
# plt.xlabel('$i-j$')
# plt.xlim(-10, 10)
# plt.legend()
# plt.tight_layout()
# plt.show()



### smoothing of 1D speckle ###
# n = 200
# speckle_baseline = np.random.randn(1, n)
# lengths = [0, 1, 2, 3, 4]
# ind = np.arange(n)

# plt.figure(figsize=(6.4, 5.4))
# for i, L in enumerate(lengths):
#     if L == 0:  # no correlation
#         speckle = speckle_baseline
#     else:
#         cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
#         cov_sqrt = np.linalg.cholesky(cov)
#         speckle = speckle_baseline @ cov_sqrt.T

#     ax = plt.subplot(len(lengths), 1, i + 1)
#     plt.plot(ind, speckle.flatten(), color=colors[i])
#     plt.xlim(-1, n)
#     plt.xticks([1, 50, 100, 150, 200])
#     plt.yticks([-2, 0, 2])
#     ax.text(1.01, 0.5, f'L={L}', transform=ax.transAxes,
#             verticalalignment='center', horizontalalignment='left')
    
# plt.xlabel('Pixel number', fontsize=14)
# plt.figtext(0.0, 0.5, 'Pixel value', ha='center', va='center', rotation='vertical', fontsize=14)

# plt.tight_layout()
# plt.show()

# ### covariance matrix ###
# n = 200
# ind = np.arange(n)
# L = 2
# cov = np.exp(-(1 / L) ** 2 * (ind[:, np.newaxis] - ind[np.newaxis, :]) ** 2)
# plt.figure()
# plt.imshow(cov)

# ### theoretical upper bound on L ###
# s = 100
# sparsity_levels = np.arange(1, s)
# upper_bound = np.log((2*sparsity_levels-1)/(np.sqrt(2)-1))**(-1/2)

# plt.figure()
# plt.plot(sparsity_levels, upper_bound, linestyle='dashed',
#          label=r"$\left(\ln\frac{2s-1}{\sqrt{2}-1}\right)^{-\frac{1}{2}}$", color='blue')
# plt.fill_between(sparsity_levels, 0, upper_bound, alpha=0.25, color='blue')
# plt.xlabel('$s$')
# plt.ylabel('$L$')
# plt.yscale('log')
# plt.xlim(0, s)
# plt.legend(fontsize=14)
# plt.show()


### 2D distance metric ###
n = 5 # grid size

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(n+1)-0.5, minor=True)
ax.set_yticks(np.arange(n+1)-0.5, minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# fill grid with (p_i, (xi, yi))
for p_i in range(n * n):
    xi, yi = divmod(p_i, n)  # convert index to 2D coordinates
    ax.text(yi, n - xi - 1, f"$p_{{{p_i}}}$\n({xi},{yi})", 
            ha='center', va='center', fontsize=16)

plt.xlim(-0.5, n-0.5)
plt.ylim(-0.5, n-0.5)
plt.show()
