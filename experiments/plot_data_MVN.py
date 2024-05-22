import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('whitegrid')

HOME = "experiments/MVN_results/"
NAMES_AND_MARKERS = {"Langevin": ("LMC",'s'), "HMC": ("HMC",'^'), "CF": ("CF",'D'), "NCV": ("NCV",'P'), 
         "NSE_diff": ("NSE (D)", '*'), "NSE_grad": ("NSE (G)",'v')}


def compute_Rsquared(y_true, y_pred):
    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()
    ss_total = sum((y_true - sum(y_true) / len(y_true)) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2

# Load the data
df = pd.read_csv(HOME + 'merged_MVN_results.csv')

# Calculate the means and standard deviations
means = df.groupby(['dim']).mean().reset_index()
stds = df.groupby(['dim']).std().reset_index()
means.to_csv(HOME + 'means.csv')
stds.to_csv(HOME + 'stds.csv')
# print(means)
# print(stds)
# Calculate the R^2 values
Rsquared = {}
for method in ['Langevin', 'HMC', 'NSE_diff', 'NSE_grad', 'CF',  'NCV']:
    Rsquared[method] = compute_Rsquared(means['true_val'], means[method])

# Plot the data
plt.figure(figsize=(20, 10))


for i, method in enumerate(['Langevin', 'HMC', 'NSE_diff', 'NSE_grad', 'CF', 'NCV']):
    n, m = NAMES_AND_MARKERS[method]
    sns.lineplot(data=means, x='dim', y=method, label=n + fr", $R^2$={Rsquared[method]:.5f}", marker=m, markersize=10)
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
plt.title(r'Estimated Value vs. Dimension for $\mathcal{N}(3, 5I_d)$', fontsize=20)
plt.savefig(HOME + 'MVN_estimated_value_vs_dimension.pdf')

