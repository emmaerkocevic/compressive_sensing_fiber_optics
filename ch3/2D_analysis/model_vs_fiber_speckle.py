import numpy as np
import matplotlib.pyplot as plt
import pyMMF
import scipy.optimize as opt


def speckle_field(number_of_modes, mode_profiles, npoints, square_pixels):

    # circular field
    E_out = np.zeros((npoints, npoints, number_of_modes), dtype=complex)
    for k in range(number_of_modes):
        A = np.random.uniform(0, 1)  # random amplitude
        phi = np.random.uniform(-np.pi, np.pi)  # random phase
        E_out[:, :, k] = A * np.exp(1j * phi) * mode_profiles[:, :, k]
    E_out = np.sum(E_out, axis=2).astype(complex)
    intensity = np.abs(E_out) ** 2

    # cropped square field
    center = npoints // 2
    x_min, x_max = center - square_pixels // 2, center + square_pixels // 2
    y_min, y_max = center - square_pixels // 2, center + square_pixels // 2
    E_out_cropped = E_out[y_min:y_max, x_min:x_max]
    intensity_cropped = np.abs(E_out_cropped) ** 2

    return intensity, intensity_cropped


def sample_correlation(n_realizations, number_of_modes, mode_profiles, npoints, square_pixels):
    intensity_flat = np.zeros((n_realizations, square_pixels * square_pixels))

    for i in range(n_realizations):
        intensity_cropped = speckle_field(number_of_modes, mode_profiles, npoints, square_pixels)[1]
        intensity_flat[i] = intensity_cropped.flatten()

    sample_corr = np.corrcoef(intensity_flat, rowvar=False)

    return sample_corr


def model_covariance(n, L):
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y)  # coordinate grids
    coords = np.column_stack((xx.ravel(), yy.ravel()))  # flatten coordinates
    diff = coords[:, None, :] - coords[None, :, :]  # pairwise coordinate differences
    dist_sq = np.sum(diff ** 2, axis=-1)  # squared Euclidean distances
    model_cov = np.exp(-dist_sq / (L ** 2))
    return model_cov


# find the value of L for which the model covariance best resembles the sample correlation
def estimate_L(n, sample_corr):

    def loss_function(L, n, sample_cov):
        model_cov = model_covariance(n, L)
        return np.linalg.norm(sample_corr - model_cov, 'fro')  # minimize the Frobenius norm

    # estimate L
    L0 = 1.0
    result = opt.minimize(loss_function, L0, args=(n, sample_corr), bounds=[(0.1, 10)])
    L_opt = result.x[0]
    
    
    # find closest L = sqrt(Lx^2 + Ly^2)
    max_a_b = int(np.ceil(L_opt)) + 1  # search limit
    L_values = sorted({np.sqrt(a**2 + b**2) for a in range(1, max_a_b) for b in range(1, max_a_b)})
    L_opt = min(L_values, key=lambda x: abs(x - L_opt))

    # residual = np.linalg.norm(sample_corr - model_covariance(n, L_opt), 'fro') / np.linalg.norm(model_covariance(n, L_opt), 'fro')
    residual = np.linalg.norm(sample_corr - model_covariance(n, L_opt), 'fro')

    return L_opt, residual


# fiber parameters
radius = 20  # core radius
wl = 1  # wavelength in vacuum
NA = 0.22  # numerical aperture
areaSize = 2 * radius  # area size
square_pixels = 80  # size of the square mask in pixels
npoints = int((square_pixels * areaSize) / (2 * radius / np.sqrt(2)))  # image resolution for the circular field
n1 = 1.46  # core refractive index

# create and configure fiber object
profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
profile.initStepIndex(n1=n1, a=radius, NA=NA)
solver = pyMMF.propagationModeSolver()
solver.setIndexProfile(profile)
solver.setWL(wl)

# solve for modes
modes = solver.solve(solver='SI')
number_of_modes = modes.number
mode_profiles = modes.getModeMatrix(npola=1).reshape((npoints, npoints, number_of_modes))

# compute sample correlation matrix, find L_opt, and compute model covariance matrix
n_realizations = 10000
sample_corr = sample_correlation(n_realizations, number_of_modes, mode_profiles, npoints, square_pixels)
L_opt, residual = estimate_L(square_pixels, sample_corr)
print(f"Optimal L: {L_opt:.2f}"); print(f"Difference: {residual:.2f}")
model_cov = model_covariance(square_pixels, L_opt)

# calculate the difference between the model and sample correlations for each pixel
# difference = np.linalg.norm(model_cov - sample_corr, axis=1) / np.linalg.norm(model_cov, axis=1)
difference = np.linalg.norm(model_cov - sample_corr, axis=1) / square_pixels
difference_2d = difference.reshape(square_pixels, square_pixels)

vmin = min(np.min(model_cov), np.min(sample_corr)); vmax = max(np.max(model_cov), np.max(sample_corr))

# plot heatmap of the model covariance, sample correlation, and pixel-wise differences
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# model covariance
im1 = axes[0].imshow(model_cov, cmap='hot', vmin=vmin, vmax=vmax)
axes[0].set_title("Model covariance matrix")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# sample correlation
im2 = axes[1].imshow(sample_corr, cmap='hot', vmin=vmin, vmax=vmax)
axes[1].set_title("Sample correlation matrix")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# pixel-wise differences
im3 = axes[2].imshow(difference_2d, cmap='Greys')
axes[2].set_title('Difference per pixel')
axes[2].set_xticks([0, 20, 40, 60, 80])
axes[2].set_yticks([0, 20, 40, 60, 80])
fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# compute center index in the speckle pattern from flattened representation
center_index = (square_pixels // 2) * square_pixels + square_pixels // 2

# extract the middle row from the model and sample correlation matrices
center_index_model = model_cov[center_index, :].reshape(square_pixels, square_pixels)
center_index_sample = sample_corr[center_index, :].reshape(square_pixels, square_pixels)

vmin = min(np.min(center_index_model), np.min(center_index_sample)); vmax = max(np.max(center_index_model), np.max(center_index_sample))

# plot heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# model
im1 = axes[0].imshow(center_index_model, cmap='hot', vmin=vmin, vmax=vmax)
axes[0].set_title("Model")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# fiber
im2 = axes[1].imshow(center_index_sample, cmap='hot', vmin=vmin, vmax=vmax)
axes[1].set_title("Fiber")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


cov_stable = model_cov + np.eye(square_pixels * square_pixels) * 0.0001
cov_sqrt = np.linalg.cholesky(cov_stable)

# model speckle
np.random.seed(0)
baseline = np.random.randn(square_pixels * square_pixels)  # N(0,I)
model_speckle = baseline @ cov_sqrt.T  # N(0,Sigma)
model_speckle_2d = model_speckle.reshape(square_pixels, square_pixels)

example_intensity = speckle_field(number_of_modes, mode_profiles, npoints, square_pixels)[0]

# plot example of model and fiber speckle
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # adjust width ratios for different subplot sizes

# model speckle
ax0 = fig.add_subplot(gs[0])  # smaller subplot
im0 = ax0.imshow(model_speckle_2d,
                 extent=[-square_pixels // 2, square_pixels // 2, -square_pixels // 2, square_pixels // 2])
ax0.set_title("Model speckle")
plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

# circular fiber speckle overlaid with square rectangle
ax1 = fig.add_subplot(gs[1])  # larger subplot
im1 = ax1.imshow(example_intensity, extent=[-npoints // 2, npoints // 2, -npoints // 2, npoints // 2])
ax1.set_title("Fiber speckle")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
rect = plt.Rectangle((-square_pixels // 2, -square_pixels // 2), square_pixels, square_pixels,
                     linewidth=2,  edgecolor='red', facecolor='none')
ax1.add_patch(rect)

plt.tight_layout()
plt.show()
