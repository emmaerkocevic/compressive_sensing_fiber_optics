import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import expon, uniform, triang, norm
from scipy.interpolate import interp1d

plt.rcParams.update({
    'axes.labelsize': 12,  
    'axes.titlesize': 14, 
    'xtick.labelsize': 12, 
    'ytick.labelsize': 12,
    'legend.fontsize': 12,  
    'font.size': 12 
})

# covariance matrix
def cov_2d(n, L):
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)  
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    dist_sq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    return np.exp(-dist_sq / (L ** 2))

# bimodal distribution functions
def bimodal_pdf(x, loc1, loc2, scale):
    return 0.5 * norm.pdf(x, loc1, scale) + 0.5 * norm.pdf(x, loc2, scale)

def bimodal_cdf(x, loc1, loc2, scale):
    return 0.5 * norm.cdf(x, loc1, scale) + 0.5 * norm.cdf(x, loc2, scale)

def bimodal_ppf(q, loc1, loc2, scale, num_points=1000):
    x_vals = np.linspace(0, 5, num_points)
    ppf_interp = interp1d(bimodal_cdf(x_vals, loc1, loc2, scale), x_vals, fill_value="extrapolate")
    return ppf_interp(np.clip(q, 0, 1))

# parameters
n = 80
L = np.sqrt(13)
m = 5000

# compute covariance matrix
cov = cov_2d(n, L)
cov_stable = cov + np.eye(n * n) * 0.0001
cov_sqrt = np.linalg.cholesky(cov_stable)

# parameters for bimodal distribution
d = (np.sqrt(2) + 2) / 2
c = np.sqrt((4 - d ** 2) / d ** 2)
mu1, mu2, sigma = 1.5, 1.5 + d, c * d / 2

np.random.seed(0)

# generate speckle patterns
baseline_speckles = np.random.randn(m, n * n)
smoothed_speckles = baseline_speckles @ cov_sqrt.T if L > 0 else baseline_speckles
u_mat = stats.norm.cdf(smoothed_speckles)

# apply different transformations to generate distributions
distributions = {
    "Exponential": expon.ppf(u_mat, scale=1),
    "Uniform": np.sqrt(12) * u_mat,
    "Triangular": triang.ppf(u_mat, c=1, loc=0, scale=np.sqrt(18)),
    "Normal": norm.ppf(u_mat, loc=3, scale=1),
    "Bimodal": bimodal_ppf(u_mat, mu1, mu2, sigma),
}

# compute empirical covariance matrices
sample_covariances = {name: np.corrcoef(data, rowvar=False) for name, data in distributions.items()}

# Apply mask to sample covariances
mask_threshold = 1 / (5 * np.e)
sample_covariances_masked = {name: np.where(sample_cov >= mask_threshold, sample_cov, 0) for name, sample_cov in sample_covariances.items()}

# extract middle row covariance
middle_pixel_index = (n // 2) * n + (n // 2)
middle_row_indices = np.arange((n // 2) * n, (n // 2 + 1) * n)

cov_middle_values = {
    "Model": cov[middle_pixel_index, middle_row_indices],
    **{name: sample_cov[middle_pixel_index, middle_row_indices] for name, sample_cov in sample_covariances.items()}
}

cov_middle_values_masked = {
    "Model": cov[middle_pixel_index, middle_row_indices],
    **{name: sample_cov[middle_pixel_index, middle_row_indices] for name, sample_cov in sample_covariances_masked.items()}
}

colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown']

# plot middle row covariance comparison
plt.figure(figsize=(10, 6))
x_vals = np.arange(n)

for i, (name, values) in enumerate(cov_middle_values.items()):
    plt.plot(x_vals, values, label=name, color=colors[i])  # Assign colors from the list

plt.xlabel("Pixel number")
plt.ylabel("Covariance")
plt.legend()
plt.show()

# plot middle row covariance comparison
plt.figure(figsize=(10, 6))
x_vals = np.arange(n)

for name, values in cov_middle_values_masked.items():
    plt.plot(x_vals, values, '--', label=f"{name} (Masked)")
    
plt.xlabel("Pixel number")
plt.ylabel("Covariance")
plt.legend()
plt.show()

# Compute differences for all sample covariances
diff_matrices = {}
for name, cov_sample in sample_covariances.items():
    diff = np.linalg.norm(cov - cov_sample, axis=1) / np.linalg.norm(cov, axis=1)
    diff_2d = diff.reshape(n, n)
    diff_matrices[name] = diff_2d  # Store for plotting

diff_matrices_masked = {name: np.linalg.norm(cov - cov_sample, axis=1) / np.linalg.norm(cov, axis=1) for name, cov_sample in sample_covariances_masked.items()}
for name in diff_matrices_masked:
    diff_matrices_masked[name] = diff_matrices_masked[name].reshape(n, n)

# Determine a common color scale
vmin = min(np.min(diff) for diff in diff_matrices.values())  # Minimum across all diffs
vmax = max(np.max(diff) for diff in diff_matrices.values())  # Maximum across all diffs

# Create figure with 5 side-by-side subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Plot each difference matrix
for ax, (name, diff_2d) in zip(axes, diff_matrices.items()):
    im = ax.imshow(diff_2d, cmap='Greys', vmin=vmin, vmax=vmax)  # Shared color scale
    ax.set_title(name)  # Set subplot title
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 20, 40, 60, 80])

# Add a single colorbar on the right
cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.08, pad=0.03, shrink=0.8)

# Show the plot
plt.show()

# Determine a common color scale
vmin = min(np.min(diff) for diff in diff_matrices_masked.values())  # Minimum across all diffs
vmax = max(np.max(diff) for diff in diff_matrices_masked.values())  # Maximum across all diffs

# Create figure with 5 side-by-side subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, (name, diff_2d) in zip(axes, diff_matrices_masked.items()):
    im = ax.imshow(diff_2d, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title(f"{name} (Masked)")
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 20, 40, 60, 80])
    
# Add a single colorbar on the right
cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.08, pad=0.03, shrink=0.8)

# Show the plot
plt.show()

# extract first row from each transformed matrix for visualization
first_row_images = {name: data[0, :].reshape((n, n)) for name, data in distributions.items()}

# define color range
vmin, vmax = min(np.min(img) for img in first_row_images.values()), max(np.max(img) for img in first_row_images.values())

# create figure and axes for visualization
fig, axs = plt.subplots(2, len(distributions), figsize=(18, 6), gridspec_kw={'width_ratios': [1] * len(distributions)})

# loop through distributions for visualization
for i, (title, img_data) in enumerate(first_row_images.items()):
    img = axs[0, i].imshow(img_data, vmin=vmin, vmax=vmax)
    axs[0, i].set_title(title)
    axs[0, i].axis("off")

    # plot histogram
    x_vals = np.linspace(np.min(img_data), np.max(img_data), 1000)
    axs[1, i].hist(img_data.ravel(), bins=20, density=True, alpha=0.5, color='blue')

    # overlay theoretical PDF
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
