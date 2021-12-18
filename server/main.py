from flask import Flask, Response, jsonify, request, render_template
import modman
import numpy as np
import torch
import json

# GLOBAL MODEL VARS
CENTRAL_MODEL = {}
ACCUMULATED_PARAMS = []
LEARNING_RATE = 0.001
ITERATION = -1
N_PUSH = 20
N_CLIENTS = 2
UPDATE_COUNT = 0
MODEL_NAME = 'experiment_01'

# LOCK VAR
MODEL_LOCK = False


app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    global CENTRAL_MODEL
    global LEARNING_RATE
    global ITERATION
    global N_PUSH
    global MODEL_NAME
    payload = {
        'params': modman.convert_tensor_to_list(CENTRAL_MODEL),
        'npush': N_PUSH,
        'learning_rate': LEARNING_RATE,
        'iteration': ITERATION,
        'model_name': MODEL_NAME
    }

    return jsonify(payload)


@app.route('/api/model/getLock', methods=['GET'])
def get_model():
    global MODEL_LOCK
    payload = {
        'model_name': MODEL_NAME,
        'lock': MODEL_LOCK
    }

    return jsonify(payload)


@app.route('/api/model/set', methods=['POST'])
def set_model():
    params = request.get_json()

    print(
        f'Got Model Params from Client ID = {params["pid"]} IP Address = {request.remote_addr}')

    global CENTRAL_MODEL
    global ITERATION
    global LEARNING_RATE

    # Check if Iteration -1 or not.
    if ITERATION > -1:
        return jsonify({'iteration': ITERATION, 'Message': 'Error! Model Params Already Set.'})

    # Update ITERATION
    ITERATION += 1

    # Set CENTRAL MODEL params
    set_model = params['model']
    LEARNING_RATE = params['learning_rate']
    if ITERATION <= 0:
        for key, value in set_model.items():
            CENTRAL_MODEL[key] = torch.Tensor(value)

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})


@app.route('/api/model/addParams', methods=['POST'])
def update_model():
    update_params = request.get_json()

    print(
        f'Got Params for Accumulation from Client ID = {update_params["pid"]} IP Address = {request.remote_addr}')

    global CENTRAL_MODEL
    global ACCUMULATED_PARAMS
    global MODEL_LOCK
    global UPDATE_COUNT
    global ITERATION
    global N_PUSH

    # Set Model Lock if Accumulated Params are Empty
    if len(ACCUMULATED_PARAMS) == 0:
        MODEL_LOCK = True

    # Append Params to Accumulated Params
    ACCUMULATED_PARAMS.append(
        modman.convert_list_to_tensor(update_params['params']))

    # Set Global Update Count
    UPDATE_COUNT = update_params['update_count']

    # Execute Federated Averaging if Accumulated Params is full
    if len(ACCUMULATED_PARAMS) == N_PUSH:
        CENTRAL_MODEL = modman.FederatedAveragingModel(ACCUMULATED_PARAMS)

        # Update Iteration Count
        ITERATION += 1

        # Save Model
        json.dump(modman.convert_tensor_to_list(
            CENTRAL_MODEL), f'./models/{MODEL_NAME}.json')

        # Release Model Lock
        MODEL_LOCK = False

    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Updated Model Params.'})


if __name__ == "__main__":
    app.run(debug=True, port=5500)
