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
data = np.load('sparsity_results_nonneg_3apr.npz', allow_pickle=True)
sparsity_levels = data['sparsity_levels']
distributions = data['distributions']
results = data['results'].item()  # Convert to dictionary

# Plot results
plt.figure()
for i, dist in enumerate(distributions):
    plt.plot(sparsity_levels, results[1][dist], label=dist, color=colors[i])
plt.xlabel('$s$')
plt.ylabel('Probability of successful reconstruction')
plt.title(r'$m=200$, $n\times n=20\times 20$, $L=1$')
plt.legend()
plt.show()

