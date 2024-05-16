import torch 
import numpy as np
from cf_utils import * 
from utils import * 
import stein_kernel
import kernel 

# implements control functionals

# parts of code from https://github.com/jz-fun/Meta_Control_Variates/blob/main/CF/sv_CV.py

# the regularization appears to be 1e-3
class Simplied_CF(object):

    def __init__(self, prior_kernel, base_kernel, X_train, Y_train, score_tensor):
        """
        :param prior_kernel: a kernel class, here is a stein kernel class.
        :param base_kernel: a kernel class
        :param X_train:  2d tensor, m * d
        :param Y_train:  2d tensor, m *  1
        :param score_tensor:  2d tensor, m * d -> score function on training data X_train
        """
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.X_train = X_train
        self.Y_train = Y_train
        self.score_tensor = score_tensor


    # Tune kernel hyper-parameters w.r.t. log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_singledat(self.prior_kernel, self.base_kernel,  self.X_train, self.Y_train, self.score_tensor)
        tune_kernelparams_negmllk_obj.do_optimize_logmll(batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr, epochs, verbose)
        optim_base_kernel_parms = torch.Tensor([tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm1, tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm2])
        self.optim_base_kernel_parms =optim_base_kernel_parms.detach()
        return optim_base_kernel_parms.detach()

    # calculate kernel and the estimate for c on the training data
    def do_closed_form_est_for_simpliedCF(self):
        # Simplified CF estimate
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]

        # gram matrix on training data
        k_XX = kernel_obj.cal_stein_base_kernel(self.X_train, self.X_train, self.score_tensor, self.score_tensor)
        m = self.X_train.size()[0]

        k_XX_inv = (k_XX + 0.001 * torch.eye(m)).inverse()

        o  = (torch.ones(1, m )  @ k_XX_inv @ self.Y_train )/( torch.ones(1, m)  @ k_XX_inv @ torch.ones( self.X_train.size()[0], 1 )  )
        return o

    # expectation estimate on test data ( = mean(y_i - predicted_f_i))
    def do_nonsim_CF(self, X_te, Y_te, score_tensor):
        # Simplified CF estimate
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
        
        #for memory debugging
        #print("(Z) X_te.shape: ", X_te.size())
        #print("(X) X_train.shape: ", self.X_train.size())

        
        k_ZX =  kernel_obj.cal_stein_base_kernel(X_te, self.X_train, score_tensor, self.score_tensor)
        print("k_ZX calculated")

        k_XX = kernel_obj.cal_stein_base_kernel(self.X_train, self.X_train, self.score_tensor, self.score_tensor)
        print("k_XX calculated")
        
        n = X_te.size()[0]
        m = self.X_train.size()[0]
        device = X_te.device

        k_XX_inv = (k_XX + 0.001 * torch.eye(m, device = device)).inverse() 
        
        Y_train_dev = self.Y_train.to(device)
        Y_te_dev = Y_te.to(device)

        o  = (torch.ones(1, m, device = device )  @ k_XX_inv @ Y_train_dev )/( torch.ones(1, m, device = device)  @ k_XX_inv @ torch.ones( self.X_train.size()[0], 1, device = device )  )
        fit = k_ZX @  k_XX_inv @ (Y_train_dev.squeeze()-o).squeeze()
        I=  (Y_te_dev.squeeze() - fit.squeeze()).mean()
        return I
    
    # predict using CF on the points X_te 
    def pred_f(self, X_te, score_tensor):
        # Simplified CF estimate
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
        k_ZX =  kernel_obj.cal_stein_base_kernel(X_te, self.X_train, score_tensor, self.score_tensor)
        k_XX = kernel_obj.cal_stein_base_kernel(self.X_train, self.X_train, self.score_tensor, self.score_tensor)
        n = X_te.size()[0]
        m = self.X_train.size()[0]
        device = X_te.device
         
        k_XX_inv = (k_XX + 0.001 * torch.eye(m, device = X_te.device)).inverse() 

        o  = (torch.ones(1, m, device=  device )  @ k_XX_inv @ self.Y_train )/( torch.ones(1, m, device = device)  @ k_XX_inv @ torch.ones( self.X_train.size()[0], 1, device = device )  )
        fit = k_ZX @  k_XX_inv @ (self.Y_train.squeeze()-o).squeeze()
        return fit       

# function for control functionals
def evaluate_cf_expectation(dist, sample_range, n_samples, h, reg = 0., given_sample = None, given_score = None, tune_kernel_params = True, return_learned= False):
    
    # can't give score without samples !
    #assert(not (given_score is None) and (given_sample is None))

    # generate the samples uniformly in range if none given
    if given_sample is None:
        #   Generate and prepare sample data
        sample = dist.generate_points(n_samples, sample_range)
        sample.requires_grad = True
    # use the given samples
    else:
        # copy given samples, and set requires_grad to True
        sample = given_sample.clone().detach().requires_grad_(True)
    
    sample = sample.to(device)

    if given_score is None:
        # calculate the score tensor
        logp_sample = dist.log_prob(sample)
        score_sample = get_grad(logp_sample.sum(), sample).detach()
    else:
        score_sample = given_score.clone().detach().to(device)

    # targets for control functional regression
    h_sample = h(sample).detach()

    cf_obj = Simplied_CF(prior_kernel = stein_kernel.stein_base_kernel, 
                         base_kernel = kernel.rbf_kernel, 
                         X_train  = sample, 
                         Y_train = h_sample, 
                         score_tensor = score_sample)
    
    if tune_kernel_params:
        print("Before tuning params")
        cf_obj.do_tune_kernelparams_negmllk(batch_size_tune = 2, 
                                            flag_if_use_medianheuristic=False, 
                                            beta_cstkernel=0., 
                                            lr=1e-2, 
                                            epochs=5, 
                                            verbose=False)
        print("After tuning params")
    # estimate the moments on the training data / samples

    with torch.no_grad():
        est_moment = cf_obj.do_nonsim_CF(sample, h_sample, score_sample)

    print(f"Est moment CF: {est_moment}")

    if return_learned:
        return est_moment.item(), cf_obj

    return est_moment.item() #-abs(est_moment.mean().item() - dist.second_moment())
