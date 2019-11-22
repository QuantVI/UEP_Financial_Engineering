# train 4 MLP with specific architectures to find the best
# We already have decided on neurons and layers
# 3-layers, 64 neurons each
# Now we will use that and train across to categories
    # 1: activation function
    # 2: optimization method

# Our activation function will either be ReLU or Sigmoid
# Our optimization method will either be Adam or mini-batch gradient descent
    # ->for the MBGD optimization, we will use 40 records per mini-batch
    # ! actually, we will stick to the default 32.


# this section is the same as in the run_normd_{something}_MLP.py files
    # comments removed to reduce lines

# !!!! Keras uses mini-batches by default - not just with the vanilla
    # versions of gradient descent.
    # Moreover, it shuffles data by default.
# We have to override at least the batch size for the MLP. (nevermind)

# We should use batches of 1, and NO shuffling, for the LSTM
    # which will be in a different script. (Yes)


import pandas as pd
import os
import numpy as np
import datetime
import inspect
import sys
import time

import tensorflow
sys.modules['keras'] = tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

snpnormfile = "Reuters Data Copy/SandP500/snp_normalized_lagged.csv"
snp_a_data = pd.read_csv(snpnormfile,parse_dates=['Local_Datetime'],
                         index_col=0)

snp_data_chrono = snp_a_data

y = snp_data_chrono['Close']
y_train = y.iloc[:1347-135]
y_validation = y.iloc[1347-135:]
X_train = snp_data_chrono.iloc[:1347-135,2:]
X_validation = snp_data_chrono.iloc[1347-135:,2:]

# we only need one configuration
cross_config = []
flat_config_1000_3_64 = [1000,3,64]
cross_config.append(flat_config_1000_3_64)
# activations
activs_list = ["relu","sigmoid"]
# optimizers
optims_list = ["adam","sgd"]
# combined to make other model settings
ao_settings = [[act,opt] for act in activs_list for opt in optims_list]

# combine settings with configs
full_spec = [(cnfig,setng)
             for cnfig in cross_config
             for setng in ao_settings]
""" full_spec will look like this

[([1000, 3, 64], ['relu', 'adam']),
 ([1000, 3, 64], ['relu', 'sgd']),
 ([1000, 3, 64], ['sigmoid', 'adam']),
 ([1000, 3, 64], ['sigmoid', 'sgd'])]

"""

# Note: deciding on batch size happens in model compiling, not here.
def config_new_model(lyr,nod,activ_fun):
    # expecting activ_fun of "relu or "sigmoid"
    model_info = {"layers":lyr,"layer":{}}
    new_model = Sequential()
    new_model.add(Dense(nod, activation=activ_fun, input_dim=9))
    model_info["layer"][1] = nod
    # define the hidden layers
    if (lyr>1):
        for this_layer in range(1,lyr):
                nod_now = nod
                new_model.add(Dense(nod_now,activation=activ_fun))
                model_info["layer"][this_layer+1] = nod_now

    new_model.add(Dense(1))
    return new_model, model_info

# We want to save/store our trained model(s)
# This is a new section
def train_a_config_return_a_model(trial,modelsettings):
    """train an MLP via adam or sgd"""
    noepoch = trial[0]
    nolayer = trial[1]
    nonodes = trial[2]
    actvfunc = modelsettings[0]
    optmfunc = modelsettings[1]
    
    curr_model, curr_info = config_new_model(nolayer,nonodes,actvfunc)

    if optmfunc == "adam":
        curr_model.compile(optimizer="adam", loss="mse")
        trn_model = curr_model.fit(X_train, y_train,
                                   validation_data=(X_validation,y_validation),
                                   epochs=noepoch, verbose=0)

    elif optmfunc == "sgd":
        curr_model.compile(optimizer="sgd", loss="mse")
        trn_model = curr_model.fit(X_train, y_train,
                                   validation_data=(X_validation,y_validation),
                                   epochs=noepoch, verbose=0)

    else:
        print("Please use adam or sgd")
        return

    ret_dict = {"epochs":trn_model.epoch[-1]+1,
                "config":curr_info,
                "result":trn_model.history}
    return ret_dict, trn_model

def train_all_specs(spec_list,sav_res):
    # info for screen
    listlen = len(spec_list)
    for spec in spec_list:
        # infor for screen
        spec_order = 1+spec_list.index(spec)

        # separate config and model_settings
        trl = spec[0] # trial
        mds = spec[1] # model_settings
        spec_result, spec_model = train_a_config_return_a_model(trl,mds)

        # save training result
        fsav = open(sav_res,'a')
        fsav.write(str(spec_result))
        fsav.write('\n')
        fsav.close()

        # Save trained model:
        # We assume that sav_res > 5 characters,
            # and ends in ".txt"
        mnamebase = sav_res[:-4] # removes the .txt
        spec_num = "{:02}".format(spec_order) #turns 1 into 01
        mname_sub = mnamebase + "_" + spec_num
        mname_trn = mname_sub + "_weights.h5"
        
        spec_model.model.save_weights(mname_trn)

        # Save Keras model config:
        # Required to load the model weights into later,
        # and not need to retrain.
        mname_cfg = mname_sub + "_jsconfig.json"

        mkerascfg = spec_model.model.to_json()
        with open(mname_cfg, "w") as json_file:
            json_file.write(mkerascfg)

        # to print to screen
        epout = spec_result["epochs"]
        szout = spec_result["config"]["layers"]
        to_print_a = "{:02} of {:02}: Epochs; {}, Layers; {}, "
        to_print_b = to_print_a.format(spec_order,listlen,epout,szout)

        tpc = "Activation; {}, Optimizer; {}"
        tpd = tpc.format(mds[0],mds[1])
        
        tp_final = to_print_b + tpd
        
        print(tp_final)


# requires a name for the savefile such as
# MLP_competition_results.txt
if __name__ == "__main__":
    print('\n')
    print(len(full_spec)," : specifications to train\n")
    train_all_specs(full_spec,sys.argv[1])
