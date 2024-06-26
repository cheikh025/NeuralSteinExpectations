import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('whitegrid')

HOME = "experiments/GMM_results/"
NAMES_AND_MARKERS = {"Langevin": ("LMC",'s'), "HMC": ("HMC",'^'), "CF": ("CF",'D'), "NCV": ("NCV",'P'), 
         "NSE_diff": ("NSE (D)", '*'), "NSE_grad": ("NSE (G)",'v')}

# Load the data
df = pd.read_csv(HOME + 'final_merged_GMM_results.csv')

# Calculate the means and standard deviations
means = df.groupby(['dim']).mean().reset_index()
stds = df.groupby(['dim']).std().reset_index()
means.to_csv(HOME + 'means.csv')
stds.to_csv(HOME + 'stds.csv')
# print(means)
# print(stds)

# Plot the data
plt.figure(figsize=(20, 10))


for i, method in enumerate(['Langevin', 'HMC', 'NSE_diff', 'NSE_grad', 'CF', 'NCV']):
    n, m = NAMES_AND_MARKERS[method]
    sns.lineplot(data=means, x='dim', y=method, label=n, marker=m, markersize=10)
    plt.fill_between(means['dim'], means[method] - stds[method], means[method] + stds[method], alpha=0.3)



sns.lineplot(data=means, x='dim', y='true_val', label='True Val', marker='o', markersize=10)
# for i, method in enumerate(['Langevin', 'HMC', 'stein_est_diff', 'stein_est_grad', 'CF', 'CF_on', 'NCV', 'NCV_on']):
#     sns.lineplot(data=means, x='dim', y=method, label=method + fr", $R^2$={Rsquared[method]:.5f}", marker=markers[i])

plt.xlabel('Dimension', fontsize=18)
plt.ylabel('Estimated Value', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.legend(fontsize=16)
plt.legend(loc='upper left')
plt.title(r'Estimated Value vs. Dimension for MoG', fontsize=20)
plt.savefig(HOME + 'GMM_estimated_value_vs_dimension.png')

