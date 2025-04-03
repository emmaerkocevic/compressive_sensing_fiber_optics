import numpy as np
import matplotlib.pyplot as plt
import pyMMF
import scipy.optimize as opt

# simulates fiber speckle
def speckle_field(number_of_modes, mode_profiles, npoints, square_pixels):
    E_out = np.zeros((npoints, npoints, number_of_modes), dtype=complex)
    for k in range(number_of_modes):
        A = np.random.uniform(0, 1)  # random amplitude in [0, 1]
        phi = np.random.uniform(-np.pi, np.pi)  # random phase in [-π, π]
        E_out[:, :, k] = A * np.exp(1j * phi) * mode_profiles[:, :, k]
    E_out = np.sum(E_out, axis=2).astype(complex)  # sum over modes
    intensity = np.abs(E_out) ** 2
    
    # crop to square region
    center = npoints // 2
    x_min, x_max = center - square_pixels // 2, center + square_pixels // 2
    y_min, y_max = center - square_pixels // 2, center + square_pixels // 2
    E_out_cropped = E_out[y_min:y_max, x_min:x_max]
    intensity_cropped = np.abs(E_out_cropped) ** 2
    
    return intensity, intensity_cropped

# sample correlation matrix
def sample_corr(n_realizations, number_of_modes, mode_profiles, npoints, square_pixels):
    intensity_flat = np.zeros((n_realizations, square_pixels * square_pixels))
    for i in range(n_realizations):
        intensity_cropped = speckle_field(number_of_modes, mode_profiles, npoints, square_pixels)[1]
        intensity_flat[i] = intensity_cropped.flatten()
    corr_mat = np.corrcoef(intensity_flat, rowvar=False)
    return corr_mat

# model covariance matrix
def cov_2d(n, L):
    x = np.arange(n)
    y = np.arange(n)
    xx, yy = np.meshgrid(x, y)  # coordinate grids
    coords = np.column_stack((xx.ravel(), yy.ravel()))  # flatten coordinates
    diff = coords[:, None, :] - coords[None, :, :]  # pairwise coordinate differences
    dist_sq = np.sum(diff ** 2, axis=-1)  # squared Euclidean distances
    cov = np.exp(-dist_sq / (L ** 2))
    return cov

# finds L for which cov_2d best resembles sample_corr
def optimal_L(n, corr_mat):
    
    def loss_function(L, n, corr_mat):
        cov_model = cov_2d(n, L)  # Compute theoretical covariance
        return np.linalg.norm(corr_mat - cov_model, 'fro')  # Frobenius norm
    
    # optimize L
    initial_L = 1.0
    result = opt.minimize(loss_function, initial_L, args=(n, corr_mat), bounds=[(0.1, 10)])
    optimal_L_cts = result.x[0]  # optimal continuous L

    # find closest discrete L = sqrt(a^2 + b^2)
    max_a_b = int(np.ceil(optimal_L_cts)) + 1  # set reasonable search limit
    L_values = sorted({np.sqrt(a**2 + b**2) for a in range(1, max_a_b) for b in range(1, max_a_b)})  # generate valid L values
    optimal_L = min(L_values, key=lambda x: abs(x - optimal_L_cts))  # find closest discrete L
    
    difference = np.linalg.norm(corr_mat - cov_2d(n, optimal_L_cts), 'fro') / np.linalg.norm(corr_mat, 'fro')

    return optimal_L, difference

# fiber parameters
radius = 20  # core radius
wl = 1  # wavelength in vacuum
NA = 0.22  # numerical aperture
areaSize = 2 * radius  # area size 
square_pixels = 80 # mask size in pixels
npoints = int((square_pixels * areaSize) / (2 * radius / np.sqrt(2)))  # orignal image resolution
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

example_intensity = speckle_field(number_of_modes, mode_profiles, npoints, square_pixels)[0]

corr_mat = sample_corr(1000, number_of_modes, mode_profiles, npoints, square_pixels)

optimal_L, difference = optimal_L(square_pixels, corr_mat)
print(f"Optimal L: {optimal_L:.2f}")
print(f"Difference: {difference:.2f}")

cov_model = cov_2d(square_pixels, optimal_L)  # model covariance based on optimal L
cov_stable = cov_model + np.eye(square_pixels * square_pixels) * 0.0001
cov_sqrt = np.linalg.cholesky(cov_stable)

# model speckle
np.random.seed(0)
baseline = np.random.randn(square_pixels * square_pixels)  # N(0, I)
model_speckle_flat = baseline @ cov_sqrt.T  # N(0, Sigma)
model_speckle = model_speckle_flat.reshape(square_pixels, square_pixels)

# compute center index in the speckle pattern from flattened representation
center_index = (square_pixels // 2) * square_pixels + square_pixels // 2

# extract the middle row from the model and sample correlation matrices
middle_row_model = cov_model[center_index, :].reshape(square_pixels, square_pixels)[square_pixels // 2, :]
middle_row_sample = corr_mat[center_index, :].reshape(square_pixels, square_pixels)[square_pixels // 2, :]

# plot comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(middle_row_model, label="model", color='red')
ax.plot(middle_row_sample, label="fiber", color='blue')
ax.set_xlabel("Pixel number")
ax.set_ylabel("Correlation")
ax.legend()
plt.show()

diff = np.linalg.norm(cov_model - corr_mat, axis=1) / np.linalg.norm(cov_model, axis=1)
diff_2d = diff.reshape(square_pixels,square_pixels)

# plot NMSE of correlations between model and fiber speckle
plt.figure()
im = plt.imshow(diff_2d, cmap='hot')
plt.title(r'$||\mathbf{\tilde\Sigma}_{i\cdot}-\mathbf{\bar{\Sigma}}_{i\cdot}||_2\enspace/\enspace||\mathbf{\tilde\Sigma}_{i\cdot}||_2$')
plt.xticks([0, 20, 40, 60, 80])
plt.yticks([0, 20, 40, 60, 80])
plt.colorbar(im)
plt.show()

# Create a figure with 1 row and 2 columns using GridSpec for layout
fig = plt.figure(figsize=(12, 5))

# Define a GridSpec layout with 1 row and 2 columns
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # Adjust width ratios for different subplot sizes

# first subplot (smaller size)
ax0 = fig.add_subplot(gs[0])  # Smaller subplot
im0 = ax0.imshow(model_speckle, extent=[-square_pixels // 2, square_pixels // 2, -square_pixels // 2, square_pixels // 2])
ax0.set_title("Model speckle")
plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)  # Add colorbar

# second subplot (normal size)
ax1 = fig.add_subplot(gs[1])  # Larger subplot
im1 = ax1.imshow(example_intensity, extent=[-npoints // 2, npoints // 2, -npoints // 2, npoints // 2])
ax1.set_title("Fiber speckle")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)  # Add colorbar
rect = plt.Rectangle((-square_pixels // 2, -square_pixels // 2), square_pixels, square_pixels, linewidth=2, edgecolor='red', facecolor='none')
ax1.add_patch(rect)

plt.tight_layout()
plt.show()
