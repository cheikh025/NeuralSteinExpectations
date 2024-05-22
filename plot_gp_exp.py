import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

x_idx = [7, 11, 15, 19, 23]
seeds = [10, 11, 12, 13, 14, 15]

cf_err = []
ncv_err = []
nse_d_err =[]
nse_g_err = []

df_list = []

for seed in seeds:
    fname = './results/GP_Results/FINAL_gp_results_samples_512_epochs_2000_seed_{}.csv'.format(seed)
    data = pd.read_csv(fname)
    
    data.drop_duplicates(subset = ['xstar_idx'], keep = 'first', inplace = True)
    data.sort_values(by=['xstar_idx'], inplace = True)
    # drop any methods
    data = data.drop(columns=['diff_cf', 'diff_ncv'])
    
    keys = [c for c in data if c.startswith('diff_')]
    print(keys)
    data = pd.melt(data, id_vars=['xstar_idx'], value_vars=keys, value_name='Error')



    data = data.rename(columns={'variable': 'Method'})
    data = data.rename(columns={'xstar_idx': 'x* Index'})
    
    # create a new column (Method) with the method name
    
    #rename methods
    data['Method'] = data['Method'].replace('diff_stein_diff', 'NSE(D)')
    data['Method'] = data['Method'].replace('diff_stein_grad', 'NSE(G)')
    #data['Method'] = data['Method'].replace('diff_ncv', 'NCV')
    data['Method'] = data['Method'].replace('diff_cf', 'CF')
    data['Method'] = data['Method'].replace('diff_varmin', 'NCV')
    #data['Method'] = data['Method'].replace('diff_varmin', 'VarMin')
    df_list.append(data.copy())

    print("Data: ", data)

plot_data = pd.concat(df_list)

print(plot_data['Error'].median())
print(plot_data['Error'].std())

import matplotlib 
matplotlib.rcParams.update({'font.size': 15})
# Plotting boxplot for the Mean absolute errors of each parameter and method using adjusted whiskers
plt.figure(figsize=(10, 6))
#sns.boxplot(x='x* Index', y='Error', hue='Method', data=data, whis=15)  
bp= sns.boxplot(x='Method', y='Error', hue = 'Method', data=plot_data, whis=15, showfliers = False)
#plt.xlabel('Query x* Index')
plt.xlabel('Method')
plt.ylabel('Mean Square Error')
plt.title('MSE for GP Mean Prediction (averaged over different x*)')
#plt.legend(title='Method', loc='upper left')  
plt.savefig('./results/GP_Results/gp_errors.png')

#print(bp['medians'])