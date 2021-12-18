import torch
import requests
from os import getpid

debug = False


# Fetch Latest Model Params (StateDict)
def fetch_params(url: str):
    # Send GET request
    r = requests.get(url=url + '/get')

    # Extract data in json format
    data = r.json()

    # Check for Iteration Number (-1 Means, No model params is present on Server)
    if data['iteration'] == -1:
        return {}, False
    else:
        if debug:
            print("Global Iteration", data['iteration'])
        return data['params'], True

# Fetch Latest Model Params (StateDict)


def get_model_lock(url: str) -> bool:
    # Send GET request
    r = requests.get(url=url + '/getLock')

    # Extract data in json format
    data = r.json()

    return data['lock']


# Send Trained Model Gradients (StateDict)
def send_trained_params(url: str, params: dict, train_count: int):
    body = {
        'params': params,
        'pid': getpid(),
        'update_count': train_count
    }

    # Send POST request
    r = requests.post(url=url, json=body)

    # Extract data in json format
    data = r.json()

    return data

# Send Trained Model Gradients (StateDict)


def send_model_params(url: str, params: dict, lr: float):
    body = {
        'model': params,
        'learning_rate': lr,
        'pid': getpid()
    }

    # Send POST request
    r = requests.post(url=url, json=body)

    # Extract data in json format
    data = r.json()

    return data


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
