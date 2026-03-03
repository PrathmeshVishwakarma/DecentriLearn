import copy

import torch
import torch.nn as nn


def federated_averaging(main_model: nn.Module, all_model_weights: list):
    global_dict = copy.deepcopy(all_model_weights[0])

    for key in global_dict.keys():

        for i in range(1, len(all_model_weights)):
            global_dict[key] += all_model_weights[i][key]

        global_dict[key] = torch.div(global_dict[key], len(all_model_weights))
    main_model.load_state_dict(global_dict)
    return main_model
