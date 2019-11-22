# To save on memory. this can be run, with browser closed
# Uses the normalized chronological data (starts before dec24)

# Goal in version 01:
# Better randomization of neuron count in hidden layers
#   no changes yet

# regular imports
import pandas as pd
import os
import numpy as np
import datetime

# other imports
import inspect
import sys
import time

# the file
snpnormfile = "Reuters Data Copy/SandP500/snp_normalized_lagged.csv"
snp_a_data = pd.read_csv(snpnormfile,parse_dates=['Local_Datetime'],
                         index_col=0)

# just to keep from changing to many variable names below
snp_data_chrono = snp_a_data

# define MLP dataset
# we need to leave a validation set, which we can use for test later
y = snp_data_chrono['Close']
y_train = y.iloc[:1347-135]
y_validation = y.iloc[1347-135:]
X_train = snp_data_chrono.iloc[:1347-135,2:]
X_validation = snp_data_chrono.iloc[1347-135:,2:]



# for NN first imports
# for tensorflow with keras
import tensorflow
# due to a problem on Windows, force the aliasing
sys.modules['keras'] = tensorflow.keras

# for NN second imports
from keras.models import Sequential
from keras.layers import Dense


# here are our bases for configurations
cross_config = []
epics = [500,1000,2500,5000]
lyrz = [1,2,3]
nodz = [32,64,128]

for epic in epics:
    for layer in lyrz:
        for node in nodz:
            cross_config.append([epic,layer,node])


# make a model based on the config

def config_model(lyr,nod):
    LO,HI = 0,99
    model_info = {"layers":lyr,
                  "layer":{}
                 }
    
    new_model = Sequential()

    # np.random.randint(low=0,high=100) #excludes upper bound
    # define model
    new_model.add(Dense(nod, activation="relu", input_dim=9))
    model_info["layer"][1] = nod
    
    # define the hidden layers
    if (lyr>1):
        for this_layer in range(lyr-1):
            # number of nodes in the hidden layer is +/- 25% of the nod, or equal to it
            node_randint = np.random.randint(LO,HI)
            node_div = node_randint//33

            if not(node_div):
                # this is zero, we reduce the nodes (ints 0 to 32)
                nod_now = int(nod*0.75)
                new_model.add(Dense(nod_now,activation="relu"))
                
                model_info["layer"][this_layer+2] = nod_now

            elif ((node_div % 2) == 0):
                # (ints 66 to 98). we increase the nodes
                nod_now = int(nod*1.25)
                new_model.add(Dense(nod_now,activation="relu"))
                
                model_info["layer"][this_layer+2] = nod_now

            else:
                # we were from 33 to 65. 33//33 ==1
                # we use same number of nodes
                new_model.add(Dense(nod,activation="relu"))
                
                model_info["layer"][this_layer+2] = nod
                
    # the loop does this for every extra hidden layer
    # finally, we add the out layer
    
    new_model.add(Dense(1))

    return new_model, model_info

# notice it returns abbreviated config info
    # e.g. {'layer': {1: 64, 2: 48, 3: 48, 4: 48}, 'layers': 4}

# we still have to compilte afterward, then fit

def train_a_config(trial):
    noepoch = trial[0]
    nolayer = trial[1]
    nonodes = trial[2]
    curr_model, curr_info = config_model(nolayer,nonodes)
    
    curr_model.compile(optimizer="adam", loss="mse")
    
    trn_model = curr_model.fit(X_train, y_train,
                               validation_data=(X_validation,y_validation),
                               epochs=noepoch, verbose=0)

    ret_dict = {"epochs":trn_model.epoch[-1]+1,
                "config":curr_info,
                "result":trn_model.history}
    return ret_dict

# where to save results
sav_res = "normd_MLP_results.txt"
tempf = open(sav_res,'w')
tempf.close()
time.sleep(3)

# cross_config is our test_list
def train_all_mlps(test_list):
    for test in test_list:
        test_result = train_a_config(test)
        fsav = open(sav_res,'a')
        fsav.write(str(test_result))
        fsav.write('\n')
        fsav.close()
        time.sleep(5)

if __name__ == "__main__":
    train_all_mlps(cross_config)
