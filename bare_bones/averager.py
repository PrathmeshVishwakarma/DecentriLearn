import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def federated_averaging(main_model: nn.Module, all_model_weights: list):
    global_dict = copy.deepcopy(all_model_weights[0])

    for key in global_dict.keys():

        for i in range(1, len(all_model_weights)):
            global_dict[key] += all_model_weights[i][key]

        global_dict[key] = torch.div(global_dict[key], len(all_model_weights))
    main_model.load_state_dict(global_dict)
    return main_model


def fedamp_averaging(all_models_weights, sigma=1.0):
    num_clients = len(all_models_weights)
    personalized_cloud_models = []
    
    # 1. Flatten the weights for each client to compute distances easily
    flattened_weights = []
    for weights in all_models_weights:
        flat = torch.cat([param.flatten() for param in weights.values()])
        flattened_weights.append(flat)
        
    flattened_weights = torch.stack(flattened_weights) # Shape: (num_clients, total_params)
    
    # 2. Compute pairwise L2 distances and Attention Weights
    for i in range(num_clients):
        distances = torch.norm(flattened_weights - flattened_weights[i], dim=1) ** 2
        # Apply softmax to get attention weights (alpha)
        attention_weights = F.softmax(-sigma * distances, dim=0)
        
        # 3. Create the personalized cloud model U_i, here we are building the state_dict for each client
        U_i = {}
        for key in all_models_weights[0].keys():
            # Weighted sum of tensors for this specific layer
            weighted_layer = sum(
                attention_weights[j] * all_models_weights[j][key] 
                for j in range(num_clients)
            )
            U_i[key] = weighted_layer
            
        personalized_cloud_models.append(U_i)
        
    return personalized_cloud_models # Returns a list of 10 different models!