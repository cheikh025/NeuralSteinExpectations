from distributions import *
from utils import *
from ToyDistributions import *
import torch
import torch.distributions as tdist
from neuralStein import *
from LangevinSampler import *
import pandas as pd
import seaborn as sns
# Set the aesthetic style of the plots
sns.set(style="whitegrid", palette="pastel")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def square(x):
            return (x**2).sum(-1)

def identity(x):
    return x.sum(-1)

def exp_compare_dim_Gaussian():
    h = square
    dims = [1, 2, 3, 5, 10, 50]
    MEAN  = 3.0
    STD = 5.0
    true_moments = [(MEAN**2 + STD**2)*dim for dim in dims]

    stein_ests = []
    langevin_ests = []
    hmc_ests = []   
    stein_errors = []
    langevin_errors = []
    hmc_errors = []

    # set up distribution, a Multivariate Gaussian
    for dim in dims:
        dist = MultivariateNormalDistribution(mean = (MEAN+torch.zeros(dim)).to(device),
                                              covariance=((STD**2)*torch.eye(dim)).to(device)
                                              ) 
        # Evaluate Stein Expectation
        stein_est = evaluate_stein_expectation(dist, dim,(-10,10), 500*(int(dim/10)+1), h=h, epochs=1000*(int(dim/10)+1))
        langevin_est = eval_Langevin(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100,device=device)
        hmc_est = eval_HMC(dist = dist, dim = dim, h=h, num_samples=10, num_chains=100, device=device)

        # since the moment sums over each dimension, the true moment is the sum of the moments for each dimension
        true_moment = true_moments[dims.index(dim)]
        print("Dimension: ", dim)
        print(f'True moment: {true_moment}, Stein estimate: {stein_est}, Langevin estimate: {langevin_est}, HMC estimate: {hmc_est}')

        stein_error = abs(true_moment - stein_est)
        langevin_error = abs(true_moment - langevin_est)
        hmc_error = abs(true_moment - hmc_est)

        stein_ests.append(stein_est)
        langevin_ests.append(langevin_est)
        hmc_ests.append(hmc_est)

        stein_errors.append(stein_error)
        langevin_errors.append(langevin_error)
        hmc_errors.append(hmc_error)

    # plot the results
    plt.figure()
    plt.plot(dims, stein_ests, label='Stein')
    plt.plot(dims, langevin_ests, label='Langevin')
    plt.plot(dims, hmc_ests, label='HMC')
    plt.plot(dims, true_moments, label='True')
    plt.xlabel('Dimension')
    plt.ylabel('Estimated Moment')
    plt.legend(loc='best')
    plt.title('Estimated Moment vs. Dimension for N(3, 5*I_d)')
    #save figure
    plt.savefig('moment_comparison_dim_Gaussian.png')

def exp_compare_over_multiple_distributions():
    h = identity
    dim = 2
    n_iter = 15
    dim_ho = 1
    # set up distributions, MVN, MOG and HO
    def harmonic_oscillator(x):
        return  harmonic_oscillator_distribution_log(x, alpha=torch.tensor([1, 1]), beta=1.0)
    dists = {"HO" : CustomDistribution(harmonic_oscillator, dim_ho),
            "MVN" : MultivariateNormalDistribution(mean = torch.zeros(dim),
                                            covariance=(3**2)*torch.eye(dim)
                                            ), 
            "MOG" : Mixture(comps=[MultivariateNormalDistribution(mean = -2*torch.ones(dim),
                                            covariance=0.5*torch.eye(dim)),
                                MultivariateNormalDistribution(mean = 2*torch.ones(dim),
                                            covariance=0.5*torch.eye(dim)
                                            )],
                            pi=torch.tensor([0.5, 0.5])
                            )
            }
    
    # Evaluate Stein Expectation
    estimations = []
    for i in range(n_iter):
        print(f"Iteration: {i+1}")
        for name, dist in dists.items():
            # Evaluate estimations for both samplers
            if name == "HO":
                dim_ = dim_ho
            else :
                dim_ = dim
            hmc_est = eval_HMC(dist=dist, dim=dim_, h=h, num_samples=10, num_chains=100)
            langevin_est = eval_Langevin(dist=dist, dim=dim_, h=h, num_samples=10, num_chains=100)
            stein_est = evaluate_stein_expectation(dist, dim_,(-10,10), dim*300, h=h, epochs=1000)

            # Store each estimation in the estimations list
            estimations.append({'distribution': name, 'sampler': 'hmc', 'estimation': hmc_est})
            estimations.append({'distribution': name, 'sampler': 'lmc', 'estimation': langevin_est})
            estimations.append({'distribution': name, 'sampler': 'stein', 'estimation': stein_est})
            print(f"\tDistribution: {name}, HMC: {hmc_est}, LMC: {langevin_est}, Stein: {stein_est}")
    
    data = pd.DataFrame(estimations)
    data.to_csv("estimations.csv", index=False)
    # Plot the results

    # Set up the matplotlib figure
    plt.figure(figsize=(20, 12))

    # Create a boxplot
    sns.boxplot(x='distribution', y='estimation', hue='sampler', data=data)

    # Add some labels and a title
    plt.xlabel('Distribution')
    plt.ylabel('Estimation Value')
    plt.title('Boxplot of Estimations by Distribution and Sampler')

    # Display the plot
    plt.show()
    



def plot_separate_boxplots() :
    data = pd.read_csv("estimations.csv")
    # Determine the unique distributions
    unique_distributions = data['distribution'].unique()

    # Set the aesthetic style of the plots
    sns.set(style="whitegrid", palette="pastel")

    # Create a separate figure for each distribution
    for i, distribution in enumerate(unique_distributions):
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the figure size as needed

        # Plot a boxplot for the distribution
        sns.boxplot(x='sampler', y='estimation', data=data[data['distribution'] == distribution], ax=ax)
        
        # Set titles and labels
        ax.set_title(f'Distribution: {distribution}', fontsize=14)
        ax.set_xlabel('Sampler', fontsize=12)
        ax.set_ylabel('Estimation Value', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10)

        # Show and save each plot
        plt.tight_layout()
        plt.savefig(f'estimations_{distribution}.png')  # Save each plot as a PNG file
        plt.show()



def exp_compare_dim_Gaussian_2():
    """
    Hard coded results from running the experiment on a multivariate normal distribution with mean 3 and std 5
    """
    dims = [1, 2, 3, 5, 10, 15, 20, 25, 30, 50]
    L = []        
    L.append({'True_moment': 34.0, 'Stein_estimate': 34.27239227294922, 'Langevin_estimate': 40.98512649536133, 'HMC_estimate': 41.045536041259766})
    L.append({'True_moment': 68.0, 'Stein_estimate': 67.44654846191406, 'Langevin_estimate': 70.31332397460938, 'HMC_estimate': 71.7470932006836})
    L.append({'True_moment': 102.0, 'Stein_estimate': 101.07706451416016, 'Langevin_estimate': 89.76448059082031, 'HMC_estimate': 98.81543731689453})
    L.append({'True_moment': 170.0, 'Stein_estimate': 167.57321166992188, 'Langevin_estimate': 160.0421905517578, 'HMC_estimate': 160.23715209960938})
    L.append({'True_moment': 340.0, 'Stein_estimate': 335.82037353515625, 'Langevin_estimate': 346.6443786621094, 'HMC_estimate': 340.8940734863281})
    L.append({'True_moment': 510.0, 'Stein_estimate': 502.8754577636719, 'Langevin_estimate': 487.567138671875, 'HMC_estimate': 526.9076538085938})
    L.append({'True_moment': 680.0, 'Stein_estimate':  671.81005859375, 'Langevin_estimate': 673.3353881835938, 'HMC_estimate': 669.2172241210938})
    L.append({'True_moment': 850.0, 'Stein_estimate':  841.4751586914062, 'Langevin_estimate': 862.7105712890625, 'HMC_estimate': 862.2083740234375})
    L.append({'True_moment': 1020.0, 'Stein_estimate':  1010.1904907226562, 'Langevin_estimate': 1034.386474609375, 'HMC_estimate': 1047.5579833984375})
    L.append({'True_moment': 1700.0, 'Stein_estimate':  1692.1697998046875, 'Langevin_estimate': 1668.9447021484375, 'HMC_estimate': 1686.2840576171875}) #5000 samples and epochs

    data = pd.DataFrame(L)
    # Calculating R squared
    R_squared = {}
    m = data['True_moment'].mean()
    R_squared['Stein_R2'] = 1 - sum([(L[i]['True_moment'] - L[i]['Stein_estimate'])**2 for i in range(len(L))]) / sum([(L[i]['True_moment'] - m)**2 for i in range(len(L))])
    R_squared['Langevin_R2'] = 1 - sum([(L[i]['True_moment'] - L[i]['Langevin_estimate'])**2 for i in range(len(L))]) / sum([(L[i]['True_moment'] - m)**2 for i in range(len(L))])
    R_squared['HMC_R2'] = 1 - sum([(L[i]['True_moment'] - L[i]['HMC_estimate'])**2 for i in range(len(L))]) / sum([(L[i]['True_moment'] - m)**2 for i in range(len(L))])
    print(R_squared)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(dims, data['Stein_estimate'], label=rf"Stein $R^2$={R_squared['Stein_R2']:.5f}", marker='o')
    plt.plot(dims, data['Langevin_estimate'], label=rf"Langevin $R^2$={R_squared['Langevin_R2']:.5f}", marker='x')
    plt.plot(dims, data['HMC_estimate'], label=rf"HMC $R^2$={R_squared['HMC_R2']:.5f}", marker='*')
    plt.plot(dims, data['True_moment'], label='True', marker='s')
    plt.xlabel('Dimension')
    plt.ylabel('Estimated Moment')
    plt.legend(loc='best')
    plt.title(r'Estimated Moment vs. Dimension for $\mathcal{N}(3, 5*I_{d})$')
    plt.savefig('moment_comparison_dim_Gaussian.png')
    plt.show()


def exp_compare_dim_MoG():
    """
    Hard coded results from running the experiment on a mixture of gaussians
    """
    # data points got from running the experiment exp_compare_dim_Gaussian with the MoG distribution
    data_points = [
        {'Dimension': 1, 'True_moment': 0, 'Stein_estimate': 0.0038942224346101284, 'Langevin_estimate': -0.08428305387496948, 'HMC_estimate': -0.15925830602645874},
        {'Dimension': 2, 'True_moment': 0, 'Stein_estimate': -0.010886505246162415, 'Langevin_estimate': -0.1637292206287384, 'HMC_estimate': -0.054844025522470474},
        {'Dimension': 3, 'True_moment': 0, 'Stein_estimate': 0.21192054450511932, 'Langevin_estimate': 0.14667458832263947, 'HMC_estimate': -0.6412695050239563},
        {'Dimension': 5, 'True_moment': 0, 'Stein_estimate': 1.3320016860961914, 'Langevin_estimate': -0.27883660793304443, 'HMC_estimate': 0.42862147092819214},
        {'Dimension': 10, 'True_moment': 0, 'Stein_estimate': 1.393086552619934, 'Langevin_estimate': 3.0766375064849854, 'HMC_estimate': 3.113009214401245},
        {'Dimension': 15, 'True_moment': 0, 'Stein_estimate': 0.9975253343582153, 'Langevin_estimate': 5.610352039337158, 'HMC_estimate': 4.341114521026611},
        {'Dimension': 20, 'True_moment': 0, 'Stein_estimate': 0.20306392014026642, 'Langevin_estimate': 7.227975368499756, 'HMC_estimate': 7.524904251098633},
        {'Dimension': 30, 'True_moment': 0, 'Stein_estimate': 0.27932214736938477, 'Langevin_estimate': 12.126618385314941, 'HMC_estimate': 13.192464828491211},
        {'Dimension': 35, 'True_moment': 0, 'Stein_estimate': 0.05003156140446663, 'Langevin_estimate': 16.504140853881836, 'HMC_estimate': 17.330976486206055},
        {'Dimension': 40, 'True_moment': 0, 'Stein_estimate': -0.21115663647651672, 'Langevin_estimate': 20.183176040649414, 'HMC_estimate': 17.576583862304688},
        {'Dimension': 45, 'True_moment': 0, 'Stein_estimate': 0.48871171474456787, 'Langevin_estimate': 22.879749298095703, 'HMC_estimate': 21.934673309326172}
    ]

    data = pd.DataFrame(data_points)


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data['Dimension'], data['Stein_estimate'], label=f"Stein", marker='o')
    plt.plot(data['Dimension'], data['Langevin_estimate'], label=f"Langevin", marker='x')
    plt.plot(data['Dimension'], data['HMC_estimate'], label=f"HMC", marker='*')
    plt.plot(data['Dimension'], data['True_moment'], label='True', marker='s')
    plt.xlabel('Dimension')
    plt.ylabel('Estimated Moment')
    plt.legend(loc='best')
    plt.title(r'Estimated Moment vs. Dimension MoG')
    plt.savefig('dim_comparison_mog.png')
    plt.show()

#plot_separate_boxplots()
#exp_compare_dim_Gaussian()
#exp_compare_over_multiple_distributions()
exp_compare_dim_MoG()