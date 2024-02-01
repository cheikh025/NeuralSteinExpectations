import torch 
import numpy as np 
import math 
import kernel 

# from https://github.com/jz-fun/Meta_Control_Variates/blob/main/CF/stein_operators.py


# the kernel for k_0(x, z) used in control functionals
class stein_base_kernel(object):

    def __init__(self, base_kernel): #, beta=1
        """
        :param distribution: an object from some distribution class
        :param base_kernel: a base kernel class
        :param base_kernel_parm1:
        :param base_kernel_parm2:
        """

        self._base_kernel_parm1 = 1
        self._base_kernel_parm2 = 1
        self.base_kernel = base_kernel


    @property
    def base_kernel_parm1(self):
        return self._base_kernel_parm1
    @base_kernel_parm1.setter
    def base_kernel_parm1(self, x):
        self._base_kernel_parm1 = x


    @property
    def base_kernel_parm2(self):
        return self._base_kernel_parm2
    @base_kernel_parm2.setter
    def base_kernel_parm2(self, x):
        self._base_kernel_parm2 = x


    #requires you to precalculate the score tensor for X and Z
    def cal_stein_base_kernel(self, X, Z, score_tensor_X, score_tensor_Z):
        """
        :param X: 2d tensor, m * p matrix
        :param Z: 2d tensor, n * p matrix
        :param score_tensor_X: 2d tensor, m * d
        :param score_tensor_Z: 2d tensor, n * d
        :return: kernel matrix, k_0(X, Z), m * n
        """
        base_kernel_obj = self.base_kernel()
        base_kernel_obj.kernel_parm1 = self.base_kernel_parm1
        base_kernel_obj.kernel_parm2 = self.base_kernel_parm2


        grad_logpX = score_tensor_X
        grad_logpZ = score_tensor_Z

        # einsum -- https://rockt.github.io/2018/04/30/einsum
        grad_k_X = torch.zeros(X.size()[0], Z.size()[0], X.size()[1])
        grad_k_Z = torch.zeros(X.size()[0], Z.size()[0], X.size()[1])
        gradgrad_k = torch.zeros(X.size()[0], Z.size()[0])
        for i in range(X.size()[0]):
            for j in range(Z.size()[0]):
                _, grad_k_X[i, j, :], grad_k_Z[i, j, :], gradgrad_k[i,j] = base_kernel_obj.deriv_base_kernel(X[i], Z[j])
        
        grad_k_X = grad_k_X.to(device=X.device)
        grad_k_Z = grad_k_Z.to(device=X.device)
        gradgrad_k = gradgrad_k.to(device=X.device)


        a = gradgrad_k
        b = torch.einsum('ik,ijk -> ij' , grad_logpX, grad_k_Z)  # grad_logpx @ grad_k_y.t()
        c = torch.einsum('jk,ijk -> ij' , grad_logpZ, grad_k_X)  # grad_logpy @ grad_k_x.t()
        d = (grad_logpX @ grad_logpZ.t()) * base_kernel_obj.cal_kernel(X,Z)

        # Store grads, as they can used to check if the pytorch autograd is correct in our scenarios
        self.grad_k_X = grad_k_X
        self.grad_k_Z = grad_logpZ
        self.gradgrad_k = gradgrad_k
        self.grad_logpX = grad_logpX
        self.grad_logpZ = grad_logpZ

        value_stein_rbf_kernel = a + b + c + d    # value_stein_rbf_kernel = self.beta + a + b + c + d

        return value_stein_rbf_kernel
