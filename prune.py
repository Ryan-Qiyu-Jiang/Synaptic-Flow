from tqdm import tqdm
import torch
import numpy as np
from Utils import load
from Utils import generator
import copy

def prune_loop(model, loss, pruner, dataloader, device,
               sparsity, linear_schedule, scope, epochs, reinitialize=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    model.eval()
    for epoch in tqdm(range(epochs)):
        pruner.apply_mask()
        pruner.score(model, loss, dataloader, device)
        if linear_schedule:
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs) # Linear
        else:
            sparse = sparsity**((epoch + 1) / epochs) # Exponential
        pruner.mask(sparse, scope)
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 1:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rand_prune_loop(unpruned_model, loss, pruner, dataloader, device,
               sparsity, linear_schedule, scope, epochs, args, reinitialize=False, sample_number=None):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    unpruned_model.eval()
    if sample_number is None:
        sample_number = 10

    param_sampled_count= {}
    model = copy.deepcopy(unpruned_model)
    pruner = load.pruner('rand_weighted')(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    for _, p in pruner.masked_parameters:
        param_sampled_count[id(p)] = torch.zeros_like(p)
    
    for _ in tqdm(range(sample_number)):
        model = copy.deepcopy(unpruned_model)
        pruner = load.pruner('rand_weighted')(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
        
        for epoch in range(epochs):
            pruner.apply_mask()
            pruner.score(model, loss, dataloader, device)
            if linear_schedule:
                sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs) # Linear
            else:
                sparse = sparsity**((epoch + 1) / epochs) # Exponential
            if epoch+1 < epochs:
                pruner.mask(sparse, scope)
        
        for mask, p in pruner.masked_parameters:
            param_sampled_count[id(p)] += mask
    
    pruner.scores = param_sampled_count
    pruner.mask(sparsity, scope)
    pruner.apply_mask()

    if reinitialize:
        model._initialize_weights()
    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 1:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))