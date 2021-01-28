from tqdm import tqdm
import torch
import numpy as np
from Utils import load
from Utils import generator
import copy
from train import *
import matplotlib.pyplot as plt

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

def approx_prune_loop(model, loss, pruner, dataloader, device,
               sparsity, linear_schedule, scope, epochs, reinitialize=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    model.eval()
    prev_weights = {}
    cur_weights = {}
    for epoch in tqdm(range(epochs)):
        eval_loss = eval(model, loss, dataloader, device, 0, early_stop=5)[0]
        print("cum_sum:{}, loss: {}".format(pruner.param_sum().item(), eval_loss))

        prev_weights = pruner.get_input_weight()
        pruner.apply_mask()
        cur_weights = pruner.get_input_weight()
        pruner.scale_params(prev_weights, cur_weights)

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
               sparsity, linear_schedule, scope, epochs, args, reinitialize=False, sample_number=None, epsilon=None, jitter=None):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    unpruned_model.eval()
    if sample_number is None:
        sample_number = args.max_samples
    if epsilon is None:
        epsilon = args.epsilon
    if jitter is None:
        jitter = args.jitter
    
    main_pruner.apply_mask()
    # zero = torch.tensor([0.]).cuda()
    # one = torch.tensor([1.]).cuda()

    last_loss = eval(unpruned_model, loss, dataloader, device, 1, early_stop=5)[0]
    sparsity_graph = [1]
    loss_graph = [last_loss]
    n, N = main_pruner.stats()
    k = ticket_size = sparsity*N

    epoch = -1
    while True:
        epoch += 1
        if linear_schedule:
            # assume final sparsity is ticket size
            if n == k:
                break
            n, _ = main_pruner.stats()
            prune_num = np.log(2/sample_number)/(np.log(n-k)-np.log(n))
            sparse = (n-prune_num)/N
            sparsity_graph += [sparse]
            if round(sparse*N) == n:
                #parse = (n-1)/N
                break
            n = round(sparse*N)
            #sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs) # Linear
        else:
            if epoch == epochs:
                break
            sparse = sparsity**((epoch + 1) / epochs) # Exponential

        num_samples = 0
        best_so_far = []
        best_loss_so_far = float('Inf')
        for _ in (range(sample_number)):
            num_samples += 1
            model = copy.deepcopy(unpruned_model)
            pruner = load.pruner(main_pruner.name)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            pruner.apply_mask()
            pruner.jitter = jitter
            pruner.score(model, loss, dataloader, device)
            pruner.mask(sparse, scope)
            remaining_params, total_params = pruner.stats()
            if remaining_params < total_params*sparse-5:
                continue
            pruner.apply_mask()
            
            eval_loss = eval(model, loss, dataloader, device, 0, early_stop=5)[0]
            if (eval_loss/last_loss - 1) < epsilon/epochs:
                last_loss = eval_loss
                loss_graph += [last_loss]
                for i, (mask, p) in enumerate(pruner.masked_parameters):
                    main_pruner.masked_parameters[i][0].copy_(mask)
                main_pruner.apply_mask()
                    # param_sampled_count[i] = pruner.scores[id(p)]
                break
            if num_samples==sample_number:
                last_loss = best_loss_so_far
                loss_graph += [last_loss]
                for i, mask in enumerate(best_so_far):
                    main_pruner.masked_parameters[i][0].copy_(mask)
                main_pruner.apply_mask()
                break

            if eval_loss < best_loss_so_far:
                best_loss_so_far = eval_loss
                best_so_far = [mask for mask, p in pruner.masked_parameters]

        # mse = 0
        # for i, (mask, p) in enumerate(pruner.masked_parameters):
        #     mse += ((param_sampled_count[i]/(sample_iteration+1) - pruner.scores[id(p)])**2).mean()
        #     # param_sampled_count[i] += pruner.scores[id(p)]
        
        # total_mse = (sample_iteration/(sample_iteration+1)) * total_mse + 1/(sample_iteration+1)*mse
        remaining_params, total_params = main_pruner.stats()
        print('sparsity={}, E[n]={}, n={}, num_samples={}, loss={}'.format(round(sparse, 3),sparse*total_params, remaining_params, num_samples, round(last_loss, 7)))

    # for i, (m, p) in enumerate(main_pruner.masked_parameters):
    #     main_pruner.scores[id(p)] = param_sampled_count[i]
    
    # main_pruner.mask(sparsity, scope)
    # main_pruner.apply_mask()

    if reinitialize:
        model._initialize_weights()
    # Confirm sparsity level
    remaining_params, total_params = main_pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 1:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))

    plt.plot(loss_graph)
    return loss_graph


