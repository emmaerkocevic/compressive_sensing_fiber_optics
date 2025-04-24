import numpy as np
import matplotlib.pyplot as plt

'''plots results from this data folder (Fig. 3.13 and Fig. 3.14 in thesis)'''

plt.rcParams.update({
    'axes.labelsize': 12, 
    'axes.titlesize': 14,  
    'xtick.labelsize': 12,  
    'ytick.labelsize': 12,  
    'legend.fontsize': 12,  
    'font.size': 12  
})

lengths = [0, 1, np.sqrt(2), 2, np.sqrt(5), 2 * np.sqrt(2), 3, np.sqrt(10), np.sqrt(13)]

color_map = {
    0: 'blue',            
    1: 'orange',          
    np.sqrt(2): 'brown',  
    2: 'dodgerblue',           
    np.sqrt(5): 'green',   
    2 * np.sqrt(2): 'limegreen',  
    3: 'gold',             
    np.sqrt(10): 'purple',      
    np.sqrt(13): 'red' 
}

sqrt_labels = {
    np.sqrt(2): r'\sqrt{2}',
    np.sqrt(5): r'\sqrt{5}',
    2 * np.sqrt(2): r'2\sqrt{2}',
    np.sqrt(10): r'\sqrt{10}',
    np.sqrt(13): r'\sqrt{13}'
}

def format_label(L):
    if L in sqrt_labels:
        return rf"$L = {sqrt_labels[L]}$"  
    else:
        return rf"$L = {int(L)}$"
        
# load measurement results
data = np.load('measurements_results_2D_cvx.npz', allow_pickle=True)
n_measurements = data['n_measurements']
p_success_dict = data['p_success_dict'].item()

# plot probabilities across measurement numbers
plt.figure()
for key in lengths:
    plt.plot(n_measurements, np.ravel(p_success_dict[key]), label=format_label(key), color=color_map[key])
plt.legend()
plt.xlabel('$m$')
plt.ylabel('Probability of successful reconstruction')
plt.title(r'$n\times n=20\times20$, $s=40$')
plt.show()
        
# load sparsity results
data = np.load('sparsity_results_2D_cvx.npz', allow_pickle=True)
sparsity_levels = data['sparsity_levels']
p_success_dict = data['p_success_dict'].item()

# plot probabilities across sparsity levels
plt.figure()
for key in lengths:
    plt.plot(sparsity_levels, np.ravel(p_success_dict[key]), label=format_label(key), color=color_map[key])
plt.legend()
plt.xlabel('$s$')
plt.ylabel('Probability of successful reconstruction')
plt.title(r'$m=200$, $n\times n=20\times20$')
plt.show()
