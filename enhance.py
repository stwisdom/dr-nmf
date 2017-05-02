import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Masking, Merge, Lambda
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
from custom_layers import uRNN
from custom_optimizers import RMSprop_and_natGrad
from audio_dataset import AudioDataset
import util
from util import pad_axis_toN_with_constant 

import sys
import os
import yaml
import hashlib
import json
import getopt
from pprint import pprint


def get_mask_value(config):
    if config['transform_x']=='mag':
        return -1.
    elif config['transform_y']=='logmag':
        return -1.
    else:
        return 0.


def load_data(config, dataset='train_valid'):
    
    if 'downsample' in config:
        downsample = config['downsample']
    else:
        downsample = 1

    if config['transform_x']=='mag':
        transform_x = (lambda x: np.sqrt(x[:x.shape[0]/2,:]**2 + x[x.shape[0]/2:,:]**2))
    elif config['transform_y']=='logmag':
        transform_x = (lambda x: np.log(np.float32(1.) + np.sqrt(x[:x.shape[0]/2,:]**2 + x[x.shape[0]/2:,:]**2)))
    else:
        transform_x = (lambda x : x)
    mask_value = get_mask_value(config)

    metrics=[]
    if config['transform_y']=='mag':
        transform_y = (lambda y: np.sqrt(y[:y.shape[0]/2,:]**2 + y[y.shape[0]/2:,:]**2))
    elif config['transform_y']=='logmag':
        transform_y = (lambda y: np.log(np.float32(1.) + np.sqrt(y[:y.shape[0]/2,:]**2 + y[y.shape[0]/2:,:]**2)))
        def mse_of_mag(y_true, y_pred):
            mask = np.float32(y_true >= 0.)
            mask_inverse_proportion = np.float32(mask.size)/np.sum(mask)
            return K.mean( mask*( (K.exp(y_true) - K.exp(y_pred))**2 ) )*mask_inverse_proportion
        metrics.append(mse_of_mag)
    else:
        transform_y = (lambda y : y)
    mask_value = get_mask_value(config)
        
    if (dataset=='test'):
        D_test=AudioDataset(config['taskfile_x_test'], config['taskfile_y_test'], datafile=config['datafile_test'], params_stft=config['params_stft'], downsample=downsample)
        x_test, y_test, mask_test = D_test.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=None)

    elif (dataset=='valid'):
        # development data
        D_valid=AudioDataset(config['taskfile_x_valid'], config['taskfile_y_valid'], datafile=config['datafile_valid'], params_stft=config['params_stft'], downsample=downsample)

        print "  Loading validation data..."
        x_valid, y_valid, mask_valid = D_valid.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=config['maxlen'])

        print "  Padding data to ensure equal sequence lengths..."
        maxseq=x_valid.shape[1]
        x_valid = pad_axis_toN_with_constant(x_valid, 1, maxseq, mask_value)
        y_valid = pad_axis_toN_with_constant(y_valid, 1, maxseq, mask_value)
        mask_valid = pad_axis_toN_with_constant(mask_valid, 1, maxseq, 0.)

        return x_valid, y_valid, mask_valid

    elif (dataset=='train'):
        # training data
        D_train=AudioDataset(config['taskfile_x_train'], config['taskfile_y_train'], datafile=config['datafile_train'], params_stft=config['params_stft'], downsample=downsample)
        x_train, y_train, mask_train = D_train.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=config['maxlen'])

        print "  Padding data to ensure equal sequence lengths..."
        maxseq=x_train.shape[1]
        x_train = pad_axis_toN_with_constant(x_train, 1, maxseq, mask_value)
        y_train = pad_axis_toN_with_constant(y_train, 1, maxseq, mask_value)
        mask_train = pad_axis_toN_with_constant(mask_train, 1, maxseq, 0.)

        return x_train, y_train, mask_train

    elif (dataset=='train_valid'):
        # training data
        D_train=AudioDataset(config['taskfile_x_train'], config['taskfile_y_train'], datafile=config['datafile_train'], params_stft=config['params_stft'], downsample=downsample)

        # development data
        D_valid=AudioDataset(config['taskfile_x_valid'], config['taskfile_y_valid'], datafile=config['datafile_valid'], params_stft=config['params_stft'], downsample=downsample)

        print "  Loading training data..."
        x_train, y_train, mask_train = D_train.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=config['maxlen'])
        print "  Loading validation data..."
        x_valid, y_valid, mask_valid = D_valid.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=config['maxlen'])

        print "  Padding data to ensure equal sequence lengths..."
        maxseq=max(x_train.shape[1],x_valid.shape[1])
        if x_train.shape[1]<maxseq:
            x_train = pad_axis_toN_with_constant(x_train, 1, maxseq, mask_value)
            y_train = pad_axis_toN_with_constant(y_train, 1, maxseq, mask_value)
            mask_train = pad_axis_toN_with_constant(mask_train, 1, maxseq, 0.)
        else:
            x_valid = pad_axis_toN_with_constant(x_valid, 1, maxseq, mask_value)
            y_valid = pad_axis_toN_with_constant(y_valid, 1, maxseq, mask_value)
            mask_valid = pad_axis_toN_with_constant(mask_valid, 1, maxseq, 0.)

        return x_train, y_train, mask_train, x_valid, y_valid, mask_valid
    else:
        ValueError("Unsupported dataset '%s'" % dataset)


def main(argv):
    configfile = ''
    helpstring = 'enhance.py -c <config YAML file>'
    try:
        opts, args = getopt.getopt(argv,"hc:",["config="])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpstring)
            yamlstring = yaml.dump(config,default_flow_style=False,explicit_start=True)
            print("YAML configuration file format:")
            print("")
            print("%YAML 1.2")
            print(yamlstring)
            sys.exit()
        elif opt in ("-c","--config"):
            configfile=arg
    print("Config file is %s" % configfile)
    if os.path.exists(configfile):
        f = open(configfile)
        config = yaml.load(f.read())
    else:

        config={}

        config['taskfile_x_train']='/data1/swisdom/chime2/chime2-wsj0/isolated/si_tr_s/taskfile.txt'
        config['taskfile_y_train']='/data1/swisdom/chime2/chime2-wsj0/scaled/si_tr_s/taskfile.txt'
        config['taskfile_x_valid']='/data1/swisdom/chime2/chime2-wsj0/isolated/si_dt_05/taskfile.txt'
        config['taskfile_y_valid']='/data1/swisdom/chime2/chime2-wsj0/scaled/si_dt_05/taskfile.txt'
        config['taskfile_x_test']='/data1/swisdom/chime2/chime2-wsj0/isolated/si_et_05/taskfile.txt'
        config['taskfile_y_test']='/data1/swisdom/chime2/chime2-wsj0/scaled/si_et_05/taskfile.txt'

        config['datafile_train']='chime2_si_tr_s.hdf5'
        config['datafile_valid']='chime2_si_dt_05.hdf5'
        config['datafile_test'] ='chime2_si_et_05.hdf5'


        config['transform_x']='mag'
        config['transform_y']='mag'
        config['loss']='mse_of_masked'

        #config['transform_x']='logmag'
        #config['transform_y']='logmag'
        #config['loss']='mse_of_logmasked'
        
        #config['transform_x']='logmag'
        #config['transform_y']='logmag'
        #config['loss']='mse_of_logmag'

        #maxlen=None
        maxlen=500

        if maxlen is not None:
            config['maxlen']=maxlen

        #model_type = 'LSTM'
        
        #model_type = 'uRNN'; config['clipnorm']=0.; config['optimizer']='rmsprop_and_natgrad'; config['lr_natGrad']=1e-4; #config['num_fast_layers']=3; config['flag_untie_fast_layers']=True
        #config['unitary_impl']='full_natGradRMS'
        #config['hidden_dim']=890
        
        model_type = 'SNMF'
        
        
        if not (model_type=='SNMF'):
            config['K_layers']=1
            config['hidden_dim']=512
            config['nb_epoch']=200
            config['learning_rate']=1e-4
            config['batch_size']=32
            config['clipnorm']=1.
            config['optimizer']='rmsprop'
            config['patience']=10


        if not (model_type=='LSTM'):
            # only store model_type in config if it's not equal to LSTM
            # (maintains consistency with old hashes)
            config['model_type']=model_type

        config['params_stft']={'N': 512, 'hop': 128, 'nch': 1}

    if config['datafile_test']==config['datafile_valid']:
        print "Discovered an experiment where datafile_test==datafile_valid. Correcting config and save files..."
        config_fixed = dict(config)
        config_fixed['datafile_test']='chime2_si_et_05.hdf5'
        hash_config = hashlib.md5(json.dumps(config, sort_keys=True)).hexdigest()
        hash_config_fixed = hashlib.md5(json.dumps(config_fixed, sort_keys=True)).hexdigest()
        print "Old hash: %s, new hash: %s" % (hash_config, hash_config_fixed)
        os.rename('savefile_' + hash_config, 'savefile_' + hash_config_fixed)
        config = config_fixed

    hash_config = hashlib.md5(json.dumps(config, sort_keys=True)).hexdigest()
    print "Hash is %s" % hash_config
    yaml.dump(config, open('config_enhance_' + hash_config + '.yaml', 'w'), default_flow_style=True)
    config['savefile']='savefile_' + hash_config

    if 'model_type' not in config:
        config['model_type'] = 'LSTM'

    if 'maxlen' not in config:
        config['maxlen']=None

    if not (model_type=='SNMF'):
        if 'num_fast_layers' not in config:
            config['num_fast_layers']=1
        if 'flag_untie_fast_layers' not in config:
            config['flag_untie_fast_layers']=False


    print "Printing configuration:"
    pprint(config)


    mask_value = get_mask_value(config)
    if os.path.exists(config['savefile']):
        input_dim = config['params_stft']['N']/2+1
        output_dim = input_dim
    else:
        # load the data
        ####################
        print "Loading data..."

        x_train, y_train, mask_train, x_valid, y_valid, mask_train = load_data(config)

        input_dim  = x_train.shape[2]
        output_dim = y_train.shape[2]


    # build the network
    #####################
    def build_model(config, mask_value, maxseq, input_dim, output_dim):
        model = Sequential()
        model.add(Masking(mask_value=mask_value, input_shape=(maxseq, input_dim)))
        for k_layer in range(config['K_layers']):
            if config['model_type']=='LSTM':
                model.add(LSTM(config['hidden_dim'],
                               return_sequences=True,
                               input_shape=(maxseq, input_dim)))
            elif config['model_type']=='uRNN':
                if 'unitary_impl' in config:
                    unitary_impl = config['unitary_impl']
                else:
                    unitary_impl = 'full_natGrad'
                model.add(uRNN(config['hidden_dim'],
                               return_sequences=True,
                               input_shape=(maxseq, input_dim),
                               unitary_impl=unitary_impl,
                               num_fast_layers=config['num_fast_layers'],
                               flag_untie_fast_layers=config['flag_untie_fast_layers']))
            else:
                ValueError("Unknown model_type of '%s'" % config['model_type'])
        model.add(TimeDistributed(Dense(output_dim)))
        
        if config['loss']=='mse_of_masked':
            #use sigmoid output layer
            model.add(TimeDistributed(Activation('sigmoid')))
            model_input = Model(input=model.input, output=model.layers[0].output)
            model_copy  = Model(input=model.input, output=model.output)
            model = Sequential()
            model.add(Merge([model_input, model_copy], mode='mul'))
            loss='mse'
        elif config['loss']=='mse_of_logmasked':
            #use sigmoid output layer
            model.add(TimeDistributed(Activation('sigmoid')))
            model_input = Model(input=model.input, output=model.layers[0].output)
            model_copy  = Model(input=model.input, output=model.output)
            model = Sequential()
            model.add(Merge([model_input, model_copy], mode='mul'))
            model.add(Lambda(lambda x: K.log(np.float32(1.) + x)))
            loss='mse'
        elif config['loss']=='mse_of_logmag':
            #use linear output layer
            loss='mse'
        else:
            ValueError("Unrecognized loss '%s'" % config['loss'])

        if config['optimizer']=='rmsprop':
            optimizer = RMSprop(lr=config['learning_rate'], clipnorm=config['clipnorm'])
        elif config['optimizer']=='rmsprop_and_natgrad':
            lr_natGrad=None
            if 'lr_natGrad' in config:
                lr_natGrad=config['lr_natGrad']
            optimizer = RMSprop_and_natGrad(lr=config['learning_rate'], lr_natGrad=lr_natGrad, clipnorm=config['clipnorm'])
        else:
            ValueError("Unrecognized optimizer '%s'" % config['optimizer'])
        model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')#, metrics=metrics)

        return model

    if os.path.exists(config['savefile']):
        """
        model.load_model(config['savefile'])
        loss_train = model.test_on_batch(x_train, y_train, sample_weight=mask_train)
        loss_valid = model.test_on_batch(x_valid, y_valid, sample_weight=mask_valid)
        print "Train loss is %f" % loss_train
        print "Validation loss is %f" % loss_valid
        """
        print "Saved weights exist in file '%s', continuing to testing..." % config['savefile']

    else:

        print "Building and compiling model..."

        model=build_model(config, mask_value, maxseq, input_dim, output_dim)

        checkpointer = ModelCheckpoint(filepath=config['savefile'], verbose=1, save_best_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1, mode='auto')

        # train the network
        #####################
        #print "Mean of squares of train is %e" % np.mean(y_train**2)
        #print "Mean of squares of valid is %e" % np.mean(y_valid**2)
        mask_train=np.squeeze(mask_train)
        mask_valid=np.squeeze(mask_valid)
        model.fit(x_train, y_train,
                  sample_weight=mask_train,
                  batch_size=config['batch_size'],
                  nb_epoch=config['nb_epoch'],
                  verbose=1,
                  validation_data=(x_valid, y_valid, mask_valid),
                  callbacks=[checkpointer, earlystopping])


    # test the trained network on the evaluation set
    ######################
    print "  Loading evaluation data..."
    x_test, y_test, mask_test = load_data(config, dataset='test')

    model_test = build_model(config, mask_value, x_test.shape[1], input_dim, output_dim)
    model_test.load_weights(config['savefile'])

    irm_output = model_test.layers[0].layers[1].layers[-1].output
    model_irm_test = Model(input=model_test.input, output=irm_output)

    mask_test=np.squeeze(mask_test)
    loss_test=0.
    snr_test=0.
    snr_test_db=0.
    idx=0
    nparts=10
    for i in range(nparts):
        print "Computing test prediction part %d of %d..." % (i+1, nparts)
        loss_test_cur = model_test.test_on_batch(x_test[idx:idx+x_test.shape[0]/nparts,:,:], y_test[idx:idx+y_test.shape[0]/nparts,:,:], sample_weight=mask_test[idx:idx+mask_test.shape[0]/nparts,:])
        loss_test += loss_test_cur*np.float32(np.sum(mask_test[idx:idx+mask_test.shape[0]/nparts,:]==1.))
        
        irm_test = model_irm_test.predict_on_batch(x_test[idx:idx+x_test.shape[0]/nparts,:,:])
        j_abs=0
        for j in range(idx, idx+x_test.shape[0]/nparts):
            yest = D_test.reconstruct_x(j, mask=irm_test[j_abs,:np.sum(mask_test[j,:]),:].T)
            y = D_test.reconstruct_y(j)
            wavfile_enhanced = D_test.y_wavfiles[j].replace('scaled', 'enhanced_%s' % hash_config)
            if not os.path.exists(os.path.dirname(wavfile_enhanced)):
                os.makedirs(os.path.dirname(wavfile_enhanced))
            util.wavwrite(wavfile_enhanced, 16e3, yest)

            snr_test_cur = np.sum(y**2)/np.sum((y-yest)**2)
            snr_test += snr_test_cur/x_test.shape[0]
            snr_test_db += 10.*np.log10(snr_test_cur)/x_test.shape[0]

            j_abs += 1

        idx += x_test.shape[0]/nparts
    print "Test loss is %f" % (loss_test/np.float32(np.sum(mask_test==1.)))
    print "Mean linear SNR in dB is %f" % (10.*np.log10(snr_test))
    print "Mean SNR in dB is %f" % snr_test_db

if __name__ == "__main__":
    main(sys.argv[1:])

