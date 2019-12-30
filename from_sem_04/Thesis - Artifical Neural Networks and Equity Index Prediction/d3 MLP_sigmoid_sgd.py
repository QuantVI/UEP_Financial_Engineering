# train a sigmoid+adam MLP, but change batch sizes
# possibly turn off shuffling.
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

snpnormfile = "../Reuters Data Copy/SandP500/snp_normalized_lagged.csv"
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
################
# activations
activs_list = ["sigmoid"]
# optimizers
optims_list = ["sgd"]
# combined to make other model settings
ao_settings = [[act,opt] for act in activs_list for opt in optims_list]
################
#batch sizes
bat = [32,1]
# shuffling
shf = [True,False]
#combined to make compile_settings
compile_settings = [[bt,sh] for bt in bat for sh in shf]

################
# combine settings with configs
full_spec = [(cnfig,setng,cmpst)
             for cnfig in cross_config
             for setng in ao_settings
             for cmpst in compile_settings]
""" full_spec will look like this

[ ([1000, 3, 64], ['sigmoid', 'sgd'], [32, True]),
  ([1000, 3, 64], ['sigmoid', 'sgd'], [32, False]),
  ([1000, 3, 64], ['sigmoid', 'sgd'], [1, True]),
  ([1000, 3, 64], ['sigmoid', 'sgd'], [1, False])]

1. shuffled mini-batch (the norm)
2. unshuffled mini-batch (false moving average)
3. shuffled SGD (pure noisy sgd)
4. unshuffled SGD (ordered iterated sgd learning)
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
def train_a_config_return_a_model(trial,modelsettings,szshuf_setng):
    """train an MLP via sgd. But make a few variants.
    1. with shuffling and mini-batch size 32 (Keras defaultt)
    2. without shuffling, and mini-batch size 32
    3. with shuffling and mini-batch size 1 (true SGD)
    4. wihout shuffling and mini-batch size 1"""
    noepoch = trial[0]
    nolayer = trial[1]
    nonodes = trial[2]
    
    actvfunc = modelsettings[0]
    optmfunc = modelsettings[1]
    
    btchsize = szshuf_setng[0]
    shufBOOL = szshuf_setng[1]

    curr_model, curr_info = config_new_model(nolayer,nonodes,actvfunc)
    curr_model.compile(optimizer=optmfunc, loss="mse")
    # we set batch_size and shuffle
    # defaults are 32 and True
    # we will also try 1 and False
    trn_model = curr_model.fit(X_train, y_train,
                               validation_data=(X_validation,y_validation),
                               epochs=noepoch, verbose=0,
                               batch_size=btchsize,
                               shuffle=shufBOOL)



    ret_dict = {"epochs":trn_model.epoch[-1]+1,
                "config":curr_info,
                "result":trn_model.history,
                "setngs":{"activation":actvfunc,
                          "optimization":optmfunc},
                "compil":{"batchsize":btchsize,
                          "shuffled":shufBOOL}}
    return ret_dict, trn_model

# modified to use compilation settings
def train_all_specs(spec_list,sav_res):
    # info for screen
    listlen = len(spec_list)
    for spec in spec_list:
        # info for screen
        spec_order = 1+spec_list.index(spec)

        # separate config, model_settings, and size+shuffle
        trl = spec[0] # trial
        mds = spec[1] # model_settings
        szsh = spec[2]
        
        spec_result, spec_model = train_a_config_return_a_model(trl,mds,szsh)

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
        to_print_a = "{:02}/{:02}: Eps;{}, Lyr;{}, "
        to_print_b = to_print_a.format(spec_order,listlen,epout,szout)

        tpc = "Actv;{}, Optm;{}, "
        tpd = tpc.format(mds[0],mds[1])

        tpe = "Btch;{}, Shuf;{} : Saved"
        tpf = str(szsh[1])
        tpg = tpe.format(szsh[0],tpf)
        
        tp_final = to_print_b + tpd + tpg
        
        print(tp_final)


# requires a name for the savefile such as
# MLP_competition_results.txt
if __name__ == "__main__":
    print('\n')
    print(len(full_spec)," : specifications to train\n")
    train_all_specs(full_spec,sys.argv[1])
