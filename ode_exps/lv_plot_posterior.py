import torch 
import matplotlib.pyplot as plt

#load posterior data 
all_data = torch.load('all_data_lv.pt')

#unconstrained samples (4000, 8) - 4000 total samples and 8 parameters
post_samples = all_data['X_all']

post_samples_np = post_samples.cpu().detach().numpy()

# plot posterior sample along 2 dimensions
plt.scatter(post_samples_np[:,0], post_samples_np[:, 1], alpha=0.9, label = "Posterior samples along x1, x2")
plt.scatter(post_samples_np[:,2], post_samples_np[:, 3], alpha=0.9, label = "Posterior samples along x3, x4")
plt.scatter(post_samples_np[:,4], post_samples_np[:, 5], alpha=0.9, label = "Posterior samples along x5, x6")
plt.scatter(post_samples_np[:,6], post_samples_np[:, 7], alpha=0.9, label = "Posterior samples along x7, x8")

print("Posterior mean: ", post_samples_np.mean(axis=0))

plt.xlabel('dim 1 (x1, x3, x5, x7)')
plt.ylabel('dim 2 (x2, x4, x6, x8)')
plt.legend()
plt.show()

plt.savefig('./plots/lv_exp/posterior_samples_other.png')