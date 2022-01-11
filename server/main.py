from flask import Flask, Response, jsonify, request, render_template
import modman
from datetime import datetime, timedelta
import json
import torch


# GLOBAL VARS
CENTRAL_MODEL = {}
LEARNING_RATE = 0.001
ITERATION = -1
ALL_PARAMS={}
SCORES={}
U_TIME_STAMP = None
WTS=30  # time in seconds to wait if not received the params from all the clients
N_PUSH = 1
N_CLIENTS = 1 
UPDATE_COUNT = 0
MODEL_NAME = 'experiment_01'
Log = True
log_path = './server/logs/'
log_file = 'server_logs.csv'
path = modman.increment_path(path=log_path+log_file,exist_ok=False,mkdir=True)
modman.csv_writer(path=path, data=[['Log for server']])
now = datetime.now
UMT=[] # Updating model time

# Client class to manage updates
class CParamas:
    client_key = None
    # iteration = None
    # steps = None
    epochs = None
    params= None
    # mem_size = None

# LOCK VAR
MODEL_LOCK = False
MODEL_COMPLETE = False 

#Start  Supporting functions
def register(add):
    global N_CLIENTS
    global Log
    print("key:-> ", add, " Registered")
    if Log:
        modman.csv_writer(path=path, data=[["Client with key:-> ", add, " Registered"]])
    SCORES[add] = 1
    N_CLIENTS = len(SCORES)
def add_score(add):
    if add in SCORES:
        return SCORES[add]
    else:
        register(add)
        return 1
def correctness(client_params):
    if SCORES[client_params.client_key] >0 and client_params.params != None:
        return True, [client_params.params, client_params.epochs]
    else:
        return False, []
def collect_params():
    global ALL_PARAMS
    global SCORES
    global N_CLIENTS
    all_params=[]
    keys=[]
    for x in ALL_PARAMS.values():
        valid, params = correctness(x)
        if valid:
            print("valid Params")
            SCORES[x.client_key] +=1
            all_params.append(params)
            keys.append(x.client_key)
        else:
            print("Invalid Params")
            SCORES[x.client_key] -=1
    akeys=SCORES.keys()
    dkeys=akeys-keys
    # print(dkeys)
    for key in dkeys:
        print("key:-> ", key, " Deleted")
        if Log:
            modman.csv_writer(path=path, data=[["Client with key:-> ", key, " Disconnected"]])
        del SCORES[key]
    N_CLIENTS = len(SCORES)
    return all_params
    
def update_model(list_of_params):
    global ITERATION
    global ALL_PARAMS
    
    # Update ITERATION
    ITERATION += 1

    # Apply Gradients and Update CENTRAL MODEL
    F_MODEL = modman.Federated_average(list_of_params)
    # # ALL_PARAMS = {}
    # print("Deleting all params after federated average")
    # dkeys=list(ALL_PARAMS.keys())
    # for key in dkeys:
    #     print("key:-> ", key, " Deleted")
    #     del ALL_PARAMS[key]
    # # PRINT RESPONSE
    # print('iteration :', ITERATION,' Updated Model Params.')
    return F_MODEL
    
#End Supporting functions

app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/model/get', methods=['GET'])
def get_model():
    data = request.get_json()
    global CENTRAL_MODEL
    global LEARNING_RATE
    global ITERATION
    global N_PUSH
    global MODEL_NAME
    print(f'Sending Global model to: Pid = {data["pid"]} : {request.remote_addr}')
    payload = {
        'params': modman.convert_tensor_to_list(CENTRAL_MODEL),
        'npush': N_PUSH,
        'learning_rate': LEARNING_RATE,
        'iteration': ITERATION,
        'model_name': MODEL_NAME
    }

    return jsonify(payload)

@app.route('/api/model/getLock', methods=['GET'])
def get_lock():
    global MODEL_LOCK
    global MODEL_COMPLETE
    payload = {
        'model_name': MODEL_NAME,
        'lock': MODEL_LOCK,
        'complete': MODEL_COMPLETE
    }

    return jsonify(payload)


@app.route('/api/model/setComplete', methods=['POST'])
def set_complete():
    global MODEL_COMPLETE
    MODEL_COMPLETE = True

    params = request.get_json()

    # Save Model
    with open(f'../server/models/{MODEL_NAME}.json', 'w') as f:
        json.dump(params['model'], f)

    return jsonify({'Message': 'Model Set To Complete.'})


@app.route('/api/model/set', methods=['POST'])
def set_model():

    params = request.get_json()

    print(
        f'Got Model Params from Client ID = {params["pid"]} IP Address = {request.remote_addr}')
    global CENTRAL_MODEL
    global ITERATION
    global LEARNING_RATE
    if ITERATION>=0:
        return jsonify({'iteration': ITERATION, 'Message': 'Error Model Already exist.'})

    # Update ITERATION
    ITERATION += 1

    # Set CENTRAL MODEL params
    set_model = params['model']
    MODEL_LOCK=True
    LEARNING_RATE = params['learning_rate']
    if ITERATION <= 0:
        for key, value in set_model.items():
            CENTRAL_MODEL[key] = torch.Tensor(value)
    MODEL_LOCK=False
    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION, 'Message': 'Model Params Set.'})



@app.route('/api/model/post_params', methods=['POST'])
def post_params():
    global ALL_PARAMS
    global CENTRAL_MODEL
    global U_TIME_STAMP #updating time stamp
    global WTS #waiting time stamp
    global N_CLIENTS
    global N_PUSH
    global UPDATE_COUNT
    global ITERATION
    global MODEL_LOCK

    update_params = request.get_json()

    c_key=request.remote_addr+":"+str(update_params["pid"])
    print("\nadd score:->",add_score(c_key))

    c_params = CParamas()
    c_params.client_key=c_key
    c_params.epochs = update_params['update_count']
    c_params.params=update_params['model']
    # c_params.mem_size=update_params['mem_size']
    # c_params.iteration=update_params['iteration']

    print(f'Got Parameters from Client ID = {update_params["pid"]} IP Address = {request.remote_addr}')

    # Storing params
    ALL_PARAMS[c_key]=c_params
    # Set Global Update Count
    UPDATE_COUNT += update_params['update_count']

    if (len(ALL_PARAMS))==1:
        U_TIME_STAMP=datetime.now()+timedelta(seconds=WTS)

    # Execute Federated Averaging if Accumulated Params is full
    
    if len(ALL_PARAMS)==N_CLIENTS or U_TIME_STAMP<datetime.now():   # U_TIME_STAMP<datetime.now() or
        sumt= now() # start time of model updation 
        data=[]
        MODEL_LOCK = True
        print("Model lock...")
        print("Updating global model with clients params: ", len(ALL_PARAMS))
        list_of_params =   collect_params()
        if(len(list_of_params)>0): 
            data.append([f'\nUpdating global parameters with params from {len(list_of_params)} client.'])
            set_model = update_model(list_of_params=list_of_params)
            
            for key, value in set_model.items():
                CENTRAL_MODEL[key] = torch.Tensor(value)
            
            # Empty Accumulated Params
            ALL_PARAMS={}
            print("Cleared All Params: ", len(ALL_PARAMS))
            # Save Model
            with open(f'./server/models/{MODEL_NAME}.json', 'w') as f:
                json.dump(modman.convert_tensor_to_list(CENTRAL_MODEL), f)
            # RETURN RESPONSE
            eumt = now()
            UMT.append(eumt-sumt)
            print('Updation time:->', eumt-sumt)
            data.append([f'Time taken for updation:-> {eumt-sumt}'])
            data.append([f'iteration: {ITERATION} Updated Global Model Params Complete.'])
            if Log:
                modman.csv_writer(path=path, data=data)
            MODEL_LOCK = False
            print("Release lock...")
            return jsonify({'iteration': ITERATION, 'n_clients':len(list_of_params), 'Message': 'Updated Global Model Params.'})
        else: 
            eumt = now()
            UMT.append(eumt-sumt)
            print('Updation time:->', eumt-sumt)
            print('Could not update the model due to receiving invalid params.')
            data.append(f'iteration: {ITERATION} Error! Global Model Params Updation Couldn\'t Complete.')
            if Log:
                modman.csv_writer(path=path, data=data)
            MODEL_LOCK = False
            print("Release lock...")
            return jsonify({'iteration': ITERATION, 'n_clients':len(list_of_params), 'Message': 'Could not update the model due to receiving invalid params.'})
        
    # RETURN RESPONSE
    return jsonify({'iteration': ITERATION,'n_clients':-1, 'Message': 'Collected Model Params.'})


if __name__ == "__main__":
    #app.run(debug=True, port=5500)

    # for listening to any network
    app.run(host="0.0.0.0", debug=False, port=5500)
