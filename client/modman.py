import torch
import requests
from os import getpid
import csv
import glob
from pathlib import Path
import re

debug = False
# Fetch Latest Model Params (StateDict)
def fetch_params(url: str):
    body = {
        'pid': getpid()
    }
    # Send GET request
    r = requests.get(url=url, json=body)
    print("Reply", r)
    # Extract data in json format
    data = r.json()

    # Check for Iteration Number (-1 Means, No model params is present on Server)
    if data['iteration'] == -1:
        return {}, data['npush'], False
    else:
        if debug:
            print("Global Iteration", data['iteration'])
        return data['params'], data['npush'], True
# remove send gradient method as we are not dealing with gradients in FL

# Send Trained Model Params (StateDict)

# Get Model Lock
def get_model_lock(url: str) -> bool:
    # Send GET request
    r = requests.get(url=url + 'getLock')

    # Extract data in json format
    data = r.json()
    print("Lock data:->", data['lock'])

    return data['lock'] 

def send_local_update(url: str, params: dict, train_count: int):
    body = {
        'model': params,
        'pid': getpid(),
        'update_count': train_count
    }

    # Send POST request
    r = requests.post(url=url, json=body)

    # Extract data in json format
    data = r.json()
    return (["iteration ", data['iteration'], "No of clients perticipant in the updation", data['n_clients'], data['Message']])


def send_model_params(url: str, params: dict, lr: float):
    body = {
        'model': params,
        'learning_rate': lr,
        'pid': getpid()
    }

    # Send POST request
    r = requests.post(url=url+'set', json=body)

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

def csv_writer(path, data):
    f = open(path, 'a')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    for x in data:
        writer.writerow(x)

    # close the file
    f.close()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path