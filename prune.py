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

def rand_prune_loop(unpruned_model, loss, main_pruner, dataloader, device,
               sparsity, linear_schedule, scope, epochs, args, reinitialize=False, sample_number=None):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    unpruned_model.eval()
    if sample_number is None:
        sample_number = 10

    main_pruner.apply_mask()
    param_sampled_count = [torch.zeros_like(p) for _, p in main_pruner.masked_parameters]
    # zero = torch.tensor([0.]).cuda()
    # one = torch.tensor([1.]).cuda()

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

        for i, (mask, p) in enumerate(pruner.masked_parameters):
            param_sampled_count[i] += pruner.scores[id(p)]
    
    for i, (m, p) in enumerate(main_pruner.masked_parameters):
        print(torch.sum(param_sampled_count[i]), torch.sum(m))
        main_pruner.scores[id(p)] = param_sampled_count[i]
    
    main_pruner.mask(sparsity, scope)
    main_pruner.apply_mask()

    if reinitialize:
        model._initialize_weights()
    # Confirm sparsity level
    remaining_params, total_params = main_pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 1:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))