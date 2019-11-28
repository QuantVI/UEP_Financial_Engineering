# after getting bad results, the fix was to change the input shape
# i've done this, but now, even the first model doesn't finish training
# after 20 minutes. I took 22 minuts for adam optimizer

# 2018-10-16 16b18b test start at 8:56pm
    # using mixed epochs 100 250 500

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
# from keras.layers import Dense, Dropout, Activation
# from keras.layers import Embedding
from keras.layers import LSTM

# i change the fold sructure a bit.
# location of the file is different than in the other scripts
# path includes ../ to go up one level
snpnormfile = "../Reuters Data Copy/SandP500/snp_normalized_lagged.csv"
snp_a_data = pd.read_csv(snpnormfile,parse_dates=['Local_Datetime'],
                         index_col=0)

snp_data_chrono = snp_a_data

y = snp_data_chrono['Close']
y_train = y.iloc[:1347-135]
y_validation = y.iloc[1347-135:]
X_train = snp_data_chrono.iloc[:1347-135,2:]
X_validation = snp_data_chrono.iloc[1347-135:,2:]

# Annoyingly, we need a slightly different shape to our data
# in order to feed it into the LSTM
xtrainlen,xvallen = len(X_train), len(X_validation)
xtrain_array, x_validation_array = np.array(X_train), np.array(X_validation)
# change tp have shape len 9 1, instead of shape len 1 9
xtrain_shape = xtrain_array.reshape(xtrainlen,9,1)
xval_shape = x_validation_array.reshape(xvallen,9,1)

Z_train = xtrain_shape
Z_validation = xval_shape


# epochs layers neurons
cross_config = []

flat_config_1000_3_20 = [1000,3,20]
flat_config_500_3_20 = [500,3,20]
flat_config_250_3_20 = [250,3,20]
flat_config_100_3_20 = [100,3,20]
#flat_config_1000_3_64 = [1000,3,64]

#cross_config.append(flat_config_1000_3_20)
#cross_config = [flat_config_100_3_20
#                ,flat_config_250_3_20
#                ,flat_config_500_3_20
#                #,flat_config_1000_3_20
#                ]

cross_config = [flat_config_1000_3_20]
#cross_config.append(flat_config_1000_3_64)

# optimizers
#optims_list = ["adam","sgd"]
#optims_list = ["sgd"]
optims_list = ["adam"]

################
#batch sizes
# bat = [32,1]
# bat = [32] # long-training time issues with batch=1
#bat = [1]
#bat = [16,8]
#bat = [16]
#bat = [8]
bat = [8,4]

# shuffling
#shf = [True,False]
shf = [False]
#combined to make compile_settings
compile_settings = [[bt,sh] for bt in bat for sh in shf]


# the 64 unit models take too long to train.
# removing them from consideration.
full_spec = [(cnfig,optim,cmpst)
             for cnfig in cross_config
             # compile setting moved to 2nd instead of third
             for cmpst in compile_settings
             for optim in optims_list
             ]

""" full_spec will look like this
[([1000, 3, 20], 'adam', [32, True]), ([1000, 3, 20], 'adam', [32, False]),
 ([1000, 3, 20], 'adam', [ 1, True]), ([1000, 3, 20], 'adam', [1, False]),
 ([1000, 3, 20],  'sgd', [32, True]), ([1000, 3, 20],  'sgd', [32, False]),
 ([1000, 3, 20],  'sgd', [ 1, True]), ([1000, 3, 20],  'sgd', [1, False]),
 
 ([1000, 3, 64], 'adam', [32, True]), ([1000, 3, 64], 'adam', [32, False]),
 ([1000, 3, 64], 'adam', [ 1, True]), ([1000, 3, 64], 'adam', [1, False]),
 ([1000, 3, 64],  'sgd', [32, True]), ([1000, 3, 64],  'sgd', [32, False]),
 ([1000, 3, 64],  'sgd', [ 1, True]), ([1000, 3, 64],  'sgd', [1, False])]
"""
# without the 64 unit per layer model, full_spec will look like this
"""
[([1000, 3, 20], 'adam', [32, True]),
 ([1000, 3, 20], 'adam', [32, False]),
 ([1000, 3, 20], 'adam', [ 1, True]),
 ([1000, 3, 20], 'adam', [ 1, False]),
 ([1000, 3, 20],  'sgd', [32, True]),
 ([1000, 3, 20],  'sgd', [32, False]),
 ([1000, 3, 20],  'sgd', [ 1, True]),
 ([1000, 3, 20],  'sgd', [ 1, False])]
"""

###############
# MAJOR convergence problems when trying to use batch size =1
#   the third model in the list above: Adam 1 True, never completes
#   Even after 80 minutes, it was not done.
#   All 1-step models will be removed, and run separately on Windows.

#   For now, we will get all the mini-batch 32 models to be trained.
# --> we may need to reduce epochs, to see if 1-step batch is useful.
##############
"""
[([1000, 3, 20], 'adam', [32, True]),
 ([1000, 3, 20], 'adam', [32, False]),
 ([1000, 3, 20], 'sgd', [32, True]),
 ([1000, 3, 20], 'sgd', [32, False])]
"""
# since items 1 and 2 finished, we will shrink full spec to items 3 and 4




# Note: disabling shuffling happens in model fitting, not here.
def config_new_model(lyr,nod):
    model_info = {"layers":lyr,"layer":{}}
    new_model = Sequential()
    # supposedly this should be None 9 1
    new_model.add(LSTM(nod, batch_input_shape=(None,9,1),
                       return_sequences=True))
    
    model_info["layer"][1] = nod
    # define the hidden layers
    if (lyr>1):
        for this_layer in range(1,lyr):
                nod_now = nod
                new_model.add(LSTM(nod, return_sequences=True))
                model_info["layer"][this_layer+1] = nod_now

    new_model.add(LSTM(1))
    return new_model, model_info

def train_a_config_return_a_model(trial,optimsetting,szshuf_setng):
    """Train an LSTM via adam or sgd
    1. with shuffling and mini-batch size 32 (Keras default)
    2. without shuffling, and mini-batch size 32
    3. with shuffling and mini-batch size 1 (true SGD)
    4. wihout shuffling and mini-batch size 1"""
    noepoch = trial[0]
    nolayer = trial[1]
    nonodes = trial[2]
    optmfunc = optimsetting

    btchsize = szshuf_setng[0]
    shufBOOL = szshuf_setng[1]

    curr_model, curr_info = config_new_model(nolayer,nonodes)
    # technially, it doesn't matter what the optimization function is.
    # we can just pass it through directly.
    # what needs to change is how we fit the model
    curr_model.compile(optimizer=optmfunc, loss="mse")

    print("Within train_a_config. Prepping to train:")
    print("\t{} {} {}".format(optmfunc,btchsize,shufBOOL))
    # now we fit appropriately.
    # we set batch_size and shuffle
    # defaults are 32 and True
    # we will also try 1 and False
    trn_model = curr_model.fit(Z_train, y_train,
                               validation_data=(Z_validation,y_validation),
                               epochs=noepoch, verbose=0,
                               batch_size=btchsize,
                               shuffle=shufBOOL)
    
    ret_dict = {"epochs":trn_model.epoch[-1]+1,
                "config":curr_info,
                "result":trn_model.history,
                "setngs":{"optimization":optmfunc},
                "compil":{"batchsize":btchsize,
                          "shuffled":shufBOOL}}
    return ret_dict, trn_model


# to train all specifications
# modified to use compilation settings
def train_all_specs(spec_list,sav_res):
    # info for screen
    listlen = len(spec_list)
    for spec in spec_list:
        # infor for screen
        spec_order = 1+spec_list.index(spec)

        # separate config, optimizer, and size+shuffle
        trl = spec[0] # trial
        mds = spec[1] # just the optimizer
        szsh = spec[2] # mini-batch size, shuffle boolean
        
        spec_result, spec_model = train_a_config_return_a_model(trl,mds,szsh)

        # save training result
        fsav = open(sav_res,'a')
        fsav.write(str(spec_result))
        fsav.write('\n')
        fsav.close()

        # Save weights of the trained model:
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

        tpc = "Optm;{} "
        tpd = tpc.format(mds)

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

