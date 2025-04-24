import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

'''plots results from this data folder (Figures 3.4-3.10 in thesis)'''

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# sparsity
data = np.load('sparsity.npz', allow_pickle=True) # change to 'sparsity_xL.npz' or 'sparsity_xgeq0.npz' for constrained signal classes
lengths = data['lengths']
sparsity_levels = data['sparsity_levels']
p_success_dict = data['p_success_dict'][()]

plt.figure()
for key in p_success_dict:
    plt.plot(sparsity_levels, np.ravel(p_success_dict[key]), label=f'$L = {int(key)}$', color=colors[int(key)])
plt.legend()  
plt.xlabel('$s$')
plt.ylabel('Probability of successful reconstruction')
plt.title('$m=100$, $N=200$')

# measurements
data = np.load('measurements.npz', allow_pickle=True) # change to 'measurements_xL.npz' or 'measurements_xgeq0.npz' for constrained signal classes
lengths = data['lengths']
n_measurements = data['n_measurements']
p_success_dict = data['p_success_dict'][()]

plt.figure()
for key in p_success_dict:
    plt.plot(n_measurements, np.ravel(p_success_dict[key]), label=f'$L = {int(key)}$', color=colors[int(key)])
plt.legend()    
plt.xlabel('$m$')
plt.ylabel('Probability of successful reconstruction')
plt.title('$N=200$, $s=20$')

# phase transition
data1 = np.load('phase_transition.npz', allow_pickle=True)
results_dict = data1['results_dict'][()]

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
axes1 = axes1.ravel()  # flatten axes array for easier indexing

for idx, L in enumerate(sorted(results_dict.keys())):
    results = results_dict[L]
    
    # extract (delta, rho, p_success) and prepare grid for interpolation
    delta, rho, p_success = zip(*results)
    delta = np.array(delta)
    rho = np.array(rho)
    p_success = np.array(p_success)

    delta_i, rho_i = np.meshgrid(
        np.linspace(0, 1, 100),  # min(delta), max(delta)?
        np.linspace(0, 1, 100)   # min(rho), max(rho)?
    )
    p_success_i = griddata((delta, rho), p_success, (delta_i, rho_i), method='cubic')

    # grayscale image
    ax = axes1[idx]
    im = ax.imshow(
        p_success_i,
        extent=(0, 1, 0, 1),
        origin='lower',
        cmap='Greys',
        aspect='auto'
    )

   # overlay the 50% contour on the interpolated grid
    contour = ax.contour(
        delta_i, rho_i, p_success_i,
        levels=[0.5], colors=[colors[idx]], linewidths=2
    )
    ax.clabel(contour, fmt="50%%", colors=[colors[idx]])

    ax.set_title(f'$L = {L}$')
    ax.set_xlabel(r'$\delta = m / N$')
    ax.set_ylabel(r'$\rho = s / m$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.5)
    
fig1.tight_layout()
cbar_ax = fig1.add_axes([0.92, 0.15, 0.02, 0.7])
fig1.colorbar(im, cax=cbar_ax, label='Probability of successful reconstruction')
plt.show()
