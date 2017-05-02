import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Masking, Merge
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from custom_layers import uRNN
from custom_optimizers import RMSprop_and_natGrad
from audio_dataset import AudioDataset
import util

import yaml
import hashlib
import json


config={}

config['taskfile_x_train']='/data1/swisdom/chime2/chime2-wsj0/isolated/si_tr_s/taskfile.txt'
config['taskfile_y_train']='/data1/swisdom/chime2/chime2-wsj0/scaled/si_tr_s/taskfile.txt'
config['taskfile_x_valid']='/data1/swisdom/chime2/chime2-wsj0/isolated/si_dt_05/taskfile.txt'
config['taskfile_y_valid']='/data1/swisdom/chime2/chime2-wsj0/scaled/si_dt_05/taskfile.txt'
config['taskfile_x_test']='/data1/swisdom/chime2/chime2-wsj0/isolated/si_et_05/taskfile.txt'
config['taskfile_y_test']='/data1/swisdom/chime2/chime2-wsj0/scaled/si_et_05/taskfile.txt'

config['datafile_train']='chime2_si_tr_s.hdf5'
config['datafile_valid']='chime2_si_dt_05.hdf5'
config['datafile_test'] ='chime2_si_dt_05.hdf5'

config['transform_x']='mag'
config['transform_y']='mag'
config['loss']='mse_of_masked'

config['K_layers']=1
config['hidden_dim']=512
config['nb_epoch']=200
config['learning_rate']=1e-4
config['batch_size']=32
config['clipnorm']=1.
config['optimizer']='rmsprop'
config['patience']=10

model_type = 'LSTM'

config['params_stft']={'N': 512, 'hop': 128, 'nch': 1}

if 'num_fast_layers' not in config:
    config['num_fast_layers']=1
if 'flag_untie_fast_layers' not in config:
    config['flag_untie_fast_layers']=False

if config['transform_x']=='mag':
    transform_x = (lambda x: np.sqrt(x[:x.shape[0]/2,:]**2 + x[x.shape[0]/2:,:]**2))
    mask_value = -1.
else:
    transform_x = (lambda x : x)
    mask_value = 0.

if config['transform_y']=='mag':
    transform_y = (lambda y: np.sqrt(y[:y.shape[0]/2,:]**2 + y[y.shape[0]/2:,:]**2))
    mask_value = -1.
else:
    transform_y = (lambda y : y)
    mask_value = 0.

# load the data
####################
maxlen=None
maxlen=500

print "Loading data..."

# development data
D_valid=AudioDataset(config['taskfile_x_valid'], config['taskfile_y_valid'], datafile=config['datafile_valid'], params_stft=config['params_stft'])

#print "  Loading validation data..."
#x_valid, y_valid, mask_valid = D_valid.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=maxlen)

for i in range(10):
    x = util.wavread(D_valid.x_wavfiles[i])[0:1,:]
    xr = D_valid.reconstruct_x(i)[0:1,:]
    if xr.shape[1] > x.shape[1]:
        xr = xr[:, :x.shape[1]]
    print "For file %d, NMSE between original x and reconstructed x is %e" % (i, np.mean( (x-xr)**2)/np.mean(x**2))
    
    y = util.wavread(D_valid.y_wavfiles[i])[0:1,:]
    yr = D_valid.reconstruct_y(i)
    if yr.shape[1] > y.shape[1]:
        yr = yr[:, :y.shape[1]]
    print "For file %d, NMSE between original y and reconstructed y is %e" % (i, np.mean( (y-yr)**2)/np.mean(y**2))

D_valid.reconstruct_audio(description="test_reconstruction_audio", idx=range(10), test=True )
