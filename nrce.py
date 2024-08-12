import torch
import torch.nn.functional as F
import math
import numpy as np

def NRCE(input, target, beta, samples_per_class, k, reduction: str = 'mean'):
    batch_size = input.size(0)
    num_classes = input.size(1)
    labels_one_hot = F.one_hot(target, num_classes).float()
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    weights = torch.tensor(weights, device=input.device).float()
    
    weights = weights.unsqueeze(0)
    weights = weights.repeat(batch_size, 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    
    p = F.softmax(input, dim=-1)  # p: (batch, num_class)
    p = p[torch.arange(p.shape[0]), target]

    sorted_values, _ = torch.sort(p)

    index = int(len(p) * k)
    prob_thresh = sorted_values[index].item()
    tau = 1 / prob_thresh
    boundary_term = math.log(tau) + 1

    loss = torch.empty_like(p)
    clip = p <= prob_thresh

    loss[clip] = -tau * p[clip] + boundary_term
    loss[~clip] = -torch.log(p[~clip])

    loss = weights * loss
    if reduction == 'none':
        return loss
    return torch.mean(loss)