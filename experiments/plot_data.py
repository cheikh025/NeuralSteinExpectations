import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_style('whitegrid')

HOME = "experiments/MVN_results/"

def compute_Rsquared(y_true, y_pred):
    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()
    ss_total = sum((y_true - sum(y_true) / len(y_true)) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2

# Load the data
df = pd.read_csv(HOME + 'merged_results.csv')

# Calculate the means and standard deviations
means = df.groupby(['dim']).mean().reset_index()
stds = df.groupby(['dim']).std().reset_index()
means.to_csv(HOME + 'means.csv')
stds.to_csv(HOME + 'stds.csv')
# print(means)
# print(stds)
# Calculate the R^2 values
Rsquared = {}
for method in ['Langevin', 'HMC', 'stein_est_diff', 'stein_est_grad', 'CF', 'CF_on', 'NCV', 'NCV_on']:
    Rsquared[method] = compute_Rsquared(means['true_val'], means[method])

# Plot the data
plt.figure(figsize=(20, 10))
markers = ['s', 'o', 'v', '*', 'd', 'D', 'p', 'P']
# plot with confidence intervals
for i, method in enumerate(['Langevin', 'HMC', 'stein_est_diff', 'stein_est_grad', 'CF', 'CF_on', 'NCV', 'NCV_on']):
    sns.lineplot(data=means, x='dim', y=method, label=method + fr", $R^2$={Rsquared[method]:.5f}", marker=markers[i])
    plt.fill_between(means['dim'], means[method] - stds[method], means[method] + stds[method], alpha=0.3)



sns.lineplot(data=means, x='dim', y='true_val', label='True Val', marker='o')
# for i, method in enumerate(['Langevin', 'HMC', 'stein_est_diff', 'stein_est_grad', 'CF', 'CF_on', 'NCV', 'NCV_on']):
#     sns.lineplot(data=means, x='dim', y=method, label=method + fr", $R^2$={Rsquared[method]:.5f}", marker=markers[i])

plt.xlabel('Dimension')
plt.ylabel('Estimated Value')
plt.legend()
plt.title(r'Estimated Value vs. Dimension for $\mathcal{N}(3, 5I_d)$')
plt.savefig(HOME + 'MVN_estimated_value_vs_dimension.png')
plt.show()

