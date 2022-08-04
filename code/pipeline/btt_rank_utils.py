"""
Utilities for Bayesian Rank Selection in Tensor Train Decomposition.

Source Paper: Bayesian Tensorized Neural Networks with Automatic Rank Selection.
Source code for this file is copied over and edited from: https://github.com/vicontek/lrbtnn  
"""
from typing import Union, Iterable

import torch
from torch import nn
from torch import distributions



params_cfg = {
    "a_l": 1, "b_l": 15, 
    "tt_layers_ranks" : [(8,)],
#    "device" : torch.device("cuda"),		#not a good idea, this might cause torch to create another device/driver context
}

def get_lambdas(
    a_l: int = params_cfg["a_l"], 
    b_l: int = params_cfg["b_l"], 
    tt_layers_ranks: Iterable = params_cfg["tt_layers_ranks"], 
    device: Union[str, torch.device] = "cuda"
):
    lambdas = []
    for i in range(len(tt_layers_ranks)):
        lambdas.append(nn.ParameterList([nn.Parameter(torch.distributions.Gamma(a_l, b_l).sample([r])) for r in tt_layers_ranks[i]]).to(device))
    return lambdas

def log_prior(model, lambdas=None, a_l=params_cfg["a_l"], b_l=params_cfg["b_l"]):
    log_prior_sum = 0
    for name, core_tensor in model.named_parameters():
        if 'core' not in name:
            continue
        core_mean = torch.zeros_like(core_tensor)
        
        if lambdas is None:
            core_std = torch.ones_like(core_tensor)
        else:
            layer_idx = int(name.split('.')[0][-1]) - 1
            core_idx = int(name.split('cores.')[-1])
            
            prev_rank = core_tensor.shape[1]
            next_rank = core_tensor.shape[2]
        
            if prev_rank == 1:
                l_next = lambdas[layer_idx][core_idx]
                l_prev = l_next
            elif next_rank  == 1:
                l_prev = lambdas[layer_idx][core_idx - 1]
                l_next = l_prev
            else:
                l_prev = lambdas[layer_idx][core_idx - 1]
                l_next = lambdas[layer_idx][core_idx]
            
#             print(l_prev.shape, l_next.shape)
            core_std = torch.einsum('p,q->pq', l_prev, l_next)
            core_std = core_std.repeat(core_tensor.shape[0], core_tensor.shape[3], 1, 1).permute(0, 2, 3, 1)
            
        log_prior_sum += distributions.Normal(core_mean, core_std).log_prob(core_tensor).sum()
    log_g = log_prior_sum
    log_lambda = 0
    if lambdas is not None:
        for layer_lambdas in lambdas:
            for l in layer_lambdas:
                log_lambda += distributions.Gamma(a_l, b_l).log_prob(l).sum()
        
    return log_g, log_lambda

def our_log_posterior(model, input, gt, lambdas = None, likelihood_coeff = 1., a_l = 1, b_l = 5, klw = 0, kl_d = 0, verbose = False): #copy of log_posterior_2 from ../tensors/lrbtnn/MAP_training*.py
    model_out = model(input)
    log_g = torch.nn.functional.log_softmax(model_out, dim = 1)
    log_g_prior, log_lambda = log_prior(model, lambdas, a_l, b_l)
    logsoftmax = torch.nn.functional.log_softmax(model_out, dim=1)

    if verbose:
        print(f"loss: {likelihood_coeff * torch.nn.functional.nll_loss(logsoftmax, gt.long(), reduction = 'sum')}; likelihood_coeff: {likelihood_coeff}; log_g_prior: {log_g_prior}; log_lambda: {log_lambda}.")
        print(f"final loss: {likelihood_coeff * torch.nn.functional.nll_loss(logsoftmax, gt.long(), reduction = 'sum') + (log_g_prior + log_lambda)}")
    return likelihood_coeff * torch.nn.functional.nll_loss(logsoftmax, gt.long(), reduction = "sum") + (log_g_prior + log_lambda)  + klw * kl_d