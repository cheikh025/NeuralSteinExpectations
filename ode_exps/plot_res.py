import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('./ode_exps/data/lv_ode_errors.csv')
data['Error'] = data['Error'].str.extract(r'(\d+\.\d+)').astype(float)

#removemethod and MCOnMesh  
data = data[data['Method'] != 'MCOnMesh']
#rename methods
data['Method'] = data['Method'].replace('Stein', 'NSE(D)')
data['Method'] = data['Method'].replace('Stein_grad', 'NSE(G)')
#order methods NSE(D) NSE(G) NCV CF
data['Method'] = pd.Categorical(data['Method'], categories=['NSE(D)', 'NSE(G)', 'NCV', 'CF'])
data = data.sort_values(by='Method')

# Plotting boxplot for the Mean absolute errors of each parameter and method using adjusted whiskers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Parameter', y='Error', hue='Method', data=data, whis=15)  
plt.xlabel('Parameter')
plt.ylabel('Absolute Error')
plt.title('Absolute Error of each parameter and method')
plt.legend(title='Method', loc='upper left')  
plt.savefig('./ode_exps/lv_ode_errors.png')
