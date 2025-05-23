import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import expon, uniform, triang, norm
from scipy.interpolate import interp1d

'''creates patterns with different intensity distributions, their sample covariance matrices, and pixel-wise differences
(Fig. 4.3 in thesis)'''

plt.rcParams.update({
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 12
})

# bimodal distribution and parameters
def bimodal_pdf(x, loc1, loc2, scale):
    return 0.5 * norm.pdf(x, loc1, scale) + 0.5 * norm.pdf(x, loc2, scale)

def bimodal_cdf(x, loc1, loc2, scale):
    return 0.5 * norm.cdf(x, loc1, scale) + 0.5 * norm.cdf(x, loc2, scale)

def bimodal_ppf(q, loc1, loc2, scale, num_points=1000):
    x_vals = np.linspace(0, 5, num_points)
    ppf_interp = interp1d(bimodal_cdf(x_vals, loc1, loc2, scale), x_vals, fill_value="extrapolate")
    return ppf_interp(np.clip(q, 0, 1))

d = (np.sqrt(2) + 2) / 2  # distance between peaks
c = np.sqrt((4 - d ** 2) / d ** 2)
mu1, mu2, sigma = 1.5, 1.5 + d, c * d / 2  # mean of peak 1, mean of peak 2 and standard deviation

# model covariance matrix
def cov_2d(n, L):
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    dist_sq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    return np.exp(-dist_sq / (L ** 2))

n = 80  # n-by-n pixel pattern
L = np.sqrt(13)  # smoothing length
cov = cov_2d(n, L)
cov_stable = cov + np.eye(n * n) * 0.0001
cov_sqrt = np.linalg.cholesky(cov_stable)

# generate speckle patterns
np.random.seed(0)
m = 10000  # number of realizations
baseline_speckles = np.random.randn(m, n * n)  # N(0,I)
smoothed_speckles = baseline_speckles @ cov_sqrt.T if L > 0 else baseline_speckles  # N(0,Sigma)
u_mat = stats.norm.cdf(smoothed_speckles)  # standard uniform

# inverse transform sampling on u_mat to obtain different distributions
distributions = {
    "Exponential": expon.ppf(u_mat, scale=1),
    "Uniform": np.sqrt(12) * u_mat,
    "Triangular": triang.ppf(u_mat, c=1, loc=0, scale=np.sqrt(18)),
    "Normal": norm.ppf(u_mat, loc=3, scale=1),
    "Bimodal": bimodal_ppf(u_mat, mu1, mu2, sigma),
}

# compute sample covariance matrices
sample_covs = {name: np.cov(data, rowvar=False) for name, data in distributions.items()}

# compute the pixel-wise differences for all sample covariances
diff_matrices = {}
for name, sample_cov in sample_covs.items():
    # diff = np.linalg.norm(cov - sample_cov, axis=1) / np.linalg.norm(cov, axis=1)
    diff = np.linalg.norm(cov - sample_cov, axis=1) / n
    diff_2d = diff.reshape(n, n)
    diff_matrices[name] = diff_2d

# set common color scales
cov_vmin = min(np.min(mat) for mat in sample_covs.values())
cov_vmax = max(np.max(mat) for mat in sample_covs.values())

diff_vmin = min(np.min(mat) for mat in diff_matrices.values())
diff_vmax = max(np.max(mat) for mat in diff_matrices.values())

fig, axes = plt.subplots(2, len(sample_covs), figsize=(4 * len(sample_covs), 8))

# sample covariance matrices
for ax, (name, sample_cov) in zip(axes[0], sample_covs.items()):
    im_cov = ax.imshow(sample_cov, cmap='hot', vmin=cov_vmin, vmax=cov_vmax)
    ax.set_title(f"{name}")
    ax.set_xticks([0, 3000, 6000])
    ax.set_yticks([0, 3000, 6000])

# pixel-wise differences
for ax, (name, diff_2d) in zip(axes[1], diff_matrices.items()):
    im_diff = ax.imshow(diff_2d, cmap='Greys', vmin=diff_vmin, vmax=diff_vmax)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 20, 40, 60, 80])

plt.tight_layout(rect=[0, 0, 0.95, 1])
cbar_ax_cov = fig.add_axes([0.96, 0.55, 0.015, 0.38])
cbar_ax_diff = fig.add_axes([0.96, 0.07, 0.015, 0.38])

fig.colorbar(im_cov, cax=cbar_ax_cov)
fig.colorbar(im_diff, cax=cbar_ax_diff)

plt.show()

# show an example pattern for each distribution, along with its histogram of intensities
example_images = {name: data[0, :].reshape((n, n)) for name, data in distributions.items()}

vmin, vmax = min(np.min(img) for img in example_images.values()), max(np.max(img) for img in example_images.values())

fig, axs = plt.subplots(2, len(distributions), figsize=(18, 6), gridspec_kw={'width_ratios': [1] * len(distributions)})
for i, (title, img_data) in enumerate(example_images.items()):

    # pattern
    img = axs[0, i].imshow(img_data, vmin=vmin, vmax=vmax)
    axs[0, i].set_title(title)
    axs[0, i].axis("off")

    # histogram
    x_vals = np.linspace(np.min(img_data), np.max(img_data), 1000)
    axs[1, i].hist(img_data.ravel(), bins=20, density=True, alpha=0.5, color='blue')

    # overlay theoretical density function
    if title == "Exponential":
        axs[1, i].plot(x_vals, expon.pdf(x_vals), 'r-')
    elif title == "Uniform":
        axs[1, i].plot(x_vals, uniform.pdf(x_vals, 0, np.sqrt(12)), 'r-')
    elif title == "Triangular":
        axs[1, i].plot(x_vals, triang.pdf(x_vals, c=1, loc=0, scale=np.sqrt(18)), 'r-')
    elif title == "Normal":
        axs[1, i].plot(x_vals, norm.pdf(x_vals, loc=3, scale=1), 'r-')
    elif title == "Bimodal":
        axs[1, i].plot(x_vals, bimodal_pdf(x_vals, mu1, mu2, sigma), 'r-')

    axs[1, i].set_xlim(0, 6)
    axs[1, i].set_xticks([0, 2, 4, 6])
    axs[1, i].set_ylim(0.0, 1.0)
    axs[1, i].set_yticks([0.0, 1.0])

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.53, 0.01, 0.35])
fig.colorbar(img, cax=cbar_ax)

plt.show()
