from typing import List
import torch


def update_model(grads: dict, global_model: dict, learning_rate: float) -> dict:
    for key in global_model.keys():
        if key in grads.keys():
            global_model[key] = _apply_grads(
                global_model[key], grads[key], learning_rate)

    return global_model


@torch.no_grad()
def FederatedAveragingModel(accu_params: list, global_model: dict) -> dict:
    avg_model = {}
    for key in global_model.keys():
        avg_model[key] = torch.stack(
            [accu_params[i][key].float() for i in range(len(accu_params))], 0).mean(0)

    return avg_model


def _apply_grads(param: list, grad: list, lr: float):

    # Convert To Torch Tensors
    grad_ = torch.tensor(grad, dtype=torch.float32)
    param_ = torch.tensor(param, dtype=torch.float32)

    # Apply Gradient Update
    accu_grads = torch.zeros(param_.size())
    accu_grads.add_(grad_, alpha=-lr)
    param_.add_(accu_grads)

    # Convert to List and Return
    return param_.tolist()

# Convert State Dict List to Tensor


def convert_list_to_tensor(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = torch.tensor(params[key], dtype=torch.float32)

    return params_


# Convert State Dict Tensors to List
def convert_tensor_to_list(params: dict) -> dict:
    params_ = {}
    for key in params.keys():
        params_[key] = params[key].tolist()

    return params_
