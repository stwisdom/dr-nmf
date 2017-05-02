import os, sys, copy
from pprint import pprint

import getopt
import yaml
import numpy as np
np.random.seed(7654)
import cPickle
import hickle
import h5py
import hashlib
import json
import re
import theano
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Masking, multiply, add, Lambda
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import Constraint
import keras.backend as K
from custom_callbacks import LossHistory
from custom_layers import SimpleDeepRNN

import util
from snmf import sparse_nmf_matlab
from audio_dataset import AudioDataset, load_data



class DenseNonNegW(Dense):
    """Equivalent to a Dense layer with a differential elementwise 
       nonnegative constraint on the kernel by using K.exp(kernel)
       during forward pass.
       Thus, to initialize the kernel to a known nonnegative matrix
       A, the kernel should be initialized with log(eps + A), where
       eps is a small value like 1e-7 to prevent NaNs.
    """
    def call(self, inputs):
        output = K.dot(inputs, K.exp(self.kernel))
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output 



class DivideAbyAplusB(_Merge):
    """Layer that divides (element-wise) the first input by the 
    elementwise sum of the first input and second input.
    It takes as input a list of tensors of len 2, all of the 
    same shape, and returns a single tensor (also of the same 
    shape).
    """

    def _merge_function(self, inputs):
        A = inputs[0]
        B = inputs[1]
        output = K.exp( K.log(1e-7 + A) - K.log(1e-7 + A + B) )
        return output

def divide_A_by_AplusB(inputs, **kwargs):
    """Functional interface to the `DivideAbyAplusB` layer.
    # Arguments
        inputs: A list of input tensors (length exactly 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, equal to A/(A+B).
    """
    return DivideAbyAplusB(**kwargs)(inputs)



def masked_seqs_to_frames(x, mask):
    (n_examples, time_steps, n_feature) = x.shape
    x=x.transpose((2,0,1)) #shape (n_feature,n_examples,time_steps)
    x_reshape = np.reshape(x, (n_feature, n_examples*time_steps))
    mask = mask.transpose((2,0,1)) #shape (1,n_examples,time_steps)
    mask_reshape = np.reshape(mask, (n_examples*time_steps,))
    idx_of_mask = np.where(mask_reshape==mask_reshape[0])[0]
    x_reshape = x_reshape[:, idx_of_mask]
    return x_reshape



def load_snmf(savefile_W_pickle, savefile_W_hickle, save_H=True):

    if os.path.exists(savefile_W_pickle) and (not os.path.exists(savefile_W_hickle)):
        print("cPickle file %s exists, but hickle file %s doesn't exist. Loading cPickle file and dumping its contents to the hickle file..." % (savefile_W_pickle, savefile_W_hickle))

        # first, load the data from the cPickle file
        with open(savefile_W_pickle, 'rb') as f:
            L = cPickle.load(f)
            W = L['W']
            if 'H' in L:
                H = L['H']
            else:
                H = None
            obj_snmf = L['obj_snmf']
            params_snmf = L['params_snmf']

        # then dump the data to a hickle file
        hickle.dump({'W': W, 'H': H, 'obj_snmf': obj_snmf, 'params_snmf': params_snmf}, savefile_W_hickle, mode='w')

    # always load from the hickle file
    print("Loading SNMF parameters from savefile %s..." % (savefile_W_hickle))
    W = hickle.load(savefile_W_hickle, path='/data_0/W')
    if save_H:
        H = hickle.load(savefile_W_hickle, path='/data_0/H')
    else:
        H = None
    obj_snmf = hickle.load(savefile_W_hickle, path='/data_0/obj_snmf')

    return W, H, obj_snmf


class MyEncoder(json.JSONEncoder):
    """From http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_snmf_savefile(params_snmf, path_dicts=''):
    sparsity = params_snmf['sparsity']
    hash_W = hashlib.md5(json.dumps(params_snmf, sort_keys=True, cls=MyEncoder)).hexdigest()
    savefile_W_hickle = path_dicts + 'W_noisy_' + hash_W + ('_sparsity%.3f.hkl' % sparsity)
    return savefile_W_hickle


def train_snmf(clean_frames, noisy_frames, params_snmf, gpuIndex=1, verbose=True, flag_recompute=False, path_dicts='./', save_H=True):

    sparsity = params_snmf['sparsity']

    # train SNMF on clean speech
    print "Training SNMF with sparsity of %.3f on clean speech..." % sparsity
    savefile_W_hickle = get_snmf_savefile(params_snmf, path_dicts=path_dicts).replace('noisy', 'clean')   
    savefile_W_pickle = savefile_W_hickle.replace('.hkl', '.p')

    if ( os.path.exists(savefile_W_pickle) or os.path.exists(savefile_W_hickle) ) and (not flag_recompute):

        print "Loading SNMF dictionary W of clean..."

        W, H, obj_snmf = load_snmf(savefile_W_pickle, savefile_W_hickle, save_H=save_H)

    else:
    
        W, H, obj_snmf = sparse_nmf_matlab(clean_frames, params_snmf, verbose=verbose, gpuIndex=gpuIndex, save_H=save_H)

        dict_dump = {'W':W,'H':H,'obj_snmf':obj_snmf,'params_snmf':params_snmf}
        if not save_H:
            # remove H from the dictionary we are dumping
            dict_dump['H'] = None
        hickle.dump(dict_dump, savefile_W_hickle, mode='w')


    # train SNMF on noisy speech, fixing the clean speech dictionary
    print "Training SNMF with sparsity of %.3f on noisy speech..." % sparsity

    W_init = np.concatenate((W, np.random.rand(*W.shape).astype(np.float32)), axis=1)
    idx_update = np.concatenate( (np.zeros(int(params_snmf['r']),dtype=bool), np.ones(int(params_snmf['r']),dtype=bool)) )

    params_snmf_noisy=copy.deepcopy(params_snmf)
    params_snmf_noisy.update({'r': 2*params_snmf['r'],
                              'init_w' : W_init,
                              'w_update_ind' : idx_update})
    
    savefile_W_hickle = get_snmf_savefile(params_snmf, path_dicts=path_dicts)    
    savefile_W_pickle = savefile_W_hickle.replace('.hkl', '.p')
    

    if ( os.path.exists(savefile_W_pickle) or os.path.exists(savefile_W_hickle) ) and (not flag_recompute):

        W_noisy, H_noisy, obj_snmf_noisy = load_snmf(savefile_W_pickle, savefile_W_hickle, save_H=save_H)

    else:

        W_noisy, H_noisy, obj_snmf_noisy = sparse_nmf_matlab(noisy_frames, params_snmf_noisy, verbose=verbose, gpuIndex=gpuIndex, save_H=save_H)

        hickle.dump({'W':W_noisy,'H':H_noisy,'obj_snmf':obj_snmf_noisy,'params_snmf':params_snmf_noisy}, savefile_W_hickle, mode='w')

    obj_snmf_noisy['cost'] = np.squeeze(obj_snmf_noisy['cost'])
    obj_snmf_noisy['div'] = np.squeeze(obj_snmf_noisy['div'])

    return W_noisy, H_noisy, obj_snmf_noisy



def build_alt(output_dim, K_layers, params, params_untied=[]):
    # assign maps from alternate parameters
    # note that A is identity here, so A^TA=I
    maps_from_alt={}    

    # same as SISTA-RNN, except that we drop the lam2 term, and simply initialize ISTA optimization for h_t with the solution from h_{t-1}

    # alternate params
    alt_params={'log_D':np.log(1e-7+params['W']), 'log_U1':np.log(np.float32(1e-7)+params['U1']), 'log_Uk':np.log(np.float32(1e-7)+params['Uk']), 'log_alph':np.log(np.float32(1e-7)+params['alph']), 'log_lam1':np.log(np.float32(1e-7)+params['lam1'])}

    # for each alt param, create a list of length K_layers that contains labels for the kth layer's param
    labels_per_k = {}
    for param_name in ['log_D', 'log_alph', 'log_lam1']:
        if param_name in params_untied:
            labels_per_k[param_name] = [param_name + ('_%d' % k) for k in range(K_layers)]
            param_to_untie = alt_params[param_name]
            del alt_params[param_name]
            for k in range(K_layers):
                alt_params[param_name + ('_%d' % k)] = param_to_untie
        else:
            labels_per_k[param_name] = [param_name]*K_layers

    # map from alt to recurrence matrix U_1
    maps_from_alt['U']=[]
    maps_from_alt['U'].append( lambda a: K.exp(a['log_U1']).transpose() )
    # map from alt to recurrence matrix U_k, k>1
    Uk_gt1 = lambda a: K.exp(a['log_Uk']).transpose()
    for k in range(K_layers-1):
        maps_from_alt['U'].append(Uk_gt1)

    # map from alt to S_k-1_to_k
    maps_from_alt['S']=[]
    I_NxN=np.eye(output_dim).astype(np.float32)
    for k in range(1, K_layers):
        # Sk = [ I - (D^T D) / alph ]^T
        Sk = lambda a, labels={'log_D':labels_per_k['log_D'][k],
                               'log_alph':labels_per_k['log_alph'][k]}: \
                        (I_NxN \
                        - K.dot(( (K.exp(alt_params[labels['log_D']])/K.sqrt(K.sum(K.square(K.exp(alt_params[labels['log_D']])), axis=0, keepdims=True)))/K.exp(a[labels['log_alph']]) ).transpose(), \
                                (K.exp(alt_params[labels['log_D']])/K.sqrt(K.sum(K.square(K.exp(alt_params[labels['log_D']])), axis=0, keepdims=True))) \
                               ) \
                       ).transpose()
        maps_from_alt['S'].append(Sk)

    # map from alt to input transform W_k
    maps_from_alt['W']=[]
    for k in range(K_layers):
        # Wk = [ D^T / alph ]^T
        Wk = lambda a, labels={'log_D':labels_per_k['log_D'][k],
                               'log_alph':labels_per_k['log_alph'][k]}: \
                        ((
                          (K.exp(a[labels['log_D']]) \
                           /K.sqrt(K.sum(K.square(K.exp(a[labels['log_D']])), axis=0, keepdims=True)) \
                          )/K.exp(a[labels['log_alph']])
                         ).transpose()
                        ).transpose()
        maps_from_alt['W'].append(Wk)

    # map from alt to biases b_k
    maps_from_alt['b']=[]
    for k in range(K_layers):
        # bk = -lam1 / alph
        bk = lambda a, labels={'log_alph':labels_per_k['log_alph'][k],
                               'log_lam1':labels_per_k['log_lam1'][k]}: \
                        -K.ones((output_dim,),dtype='float32')*K.exp(a[labels['log_lam1']])/K.exp(a[labels['log_alph']])
        maps_from_alt['b'].append(bk)

    return alt_params,maps_from_alt


def build_unfolded_snmf(params_unfolded_snmf):
    input_dim = params_unfolded_snmf['input_dim']
    hidden_dim = params_unfolded_snmf['hidden_dim']
    output_dim = params_unfolded_snmf['output_dim']
    mask_value = params_unfolded_snmf['mask_value']
    maxseq = params_unfolded_snmf['maxseq']
    K_layers = params_unfolded_snmf['K_layers']
    W_noisy = params_unfolded_snmf['W']
 
    # initializations for various parameters of the SNMF model
    params_const={'W':np.float32(W_noisy),
                  'U1':np.eye(hidden_dim).astype(np.float32),
                  'Uk':np.zeros((hidden_dim,hidden_dim)).astype(np.float32),
                  'alph':np.float32(params_unfolded_snmf['alph']),
                  'lam1':np.float32(params_unfolded_snmf['lam1'])} 

    if 'untie_alph' in params_unfolded_snmf and params_unfolded_snmf['untie_alph']:
        params_const['alph'] = params_const['alph']*np.ones((hidden_dim,), dtype=np.float32)

    # get list of untied parameters
    if 'params_untied' in params_unfolded_snmf:
        params_untied = params_unfolded_snmf['params_untied']
    else:
        params_untied = []

    # build maps from SNMF parameters to RNN parameters
    alt_params,maps_from_alt = build_alt(hidden_dim, K_layers, params_const, params_untied=params_untied)

    # assign which parameters are trainable, and reconcile these
    # with the untied parameters
    if 'params_trainable' in params_unfolded_snmf:
        keys_trainable = params_unfolded_snmf['params_trainable']
        if len(params_untied) > 0:
            keys_trainable_new = []
            for param in keys_trainable:
                if param in params_untied:
                    keys_trainable_new = keys_trainable_new + [(param + ('_%d' %k)) for k in range(K_layers)]
                else:
                    keys_trainable_new = keys_trainable_new + [param]
            keys_trainable = keys_trainable_new

    model = Sequential()
    
    # input masking layer to deal with different len. seq.s
    model.add(Masking(mask_value=mask_value,
                      input_shape=(maxseq, input_dim)))

    # SISTA-RNN layer contains K deep layers at each time step
    model.add(SimpleDeepRNN(hidden_dim,
            input_shape=(maxseq, input_dim),
            return_sequences=True,
            activation='relu',
            K_layers=K_layers,
            alt_params=alt_params,
            keys_trainable=keys_trainable,
            maps_from_alt=maps_from_alt,
            flag_connect_input_to_layers=True,
            flag_nonnegative=True))

    # add output layer for reconstruction            
    def index_first_n_dim2(x, idx=-1):
        return x[:,:,:idx]
    def index_last_n_dim2(x, idx=0):
        return x[:,:,idx:]
    r = hidden_dim/2
    index_output_shape = (maxseq, r)

    # H_clean consists of the first r elements of h at each time step
    H_clean = Lambda(index_first_n_dim2,
                     arguments={'idx':r},
                     output_shape=index_output_shape, name='H_clean')(model.output)

    # multiply by W_clean to create the reconstructed spectrogram of clean
    log_W_clean = K.log(1e-7 + W_noisy[:,:r]).eval()
    clean_est = TimeDistributed(DenseNonNegW(output_dim, use_bias=False, weights=[log_W_clean.T]), name='clean_est')(H_clean)
    
    # H_noise consists of the last r elements of h at each time step
    H_noise = Lambda(index_last_n_dim2, 
                     arguments={'idx':r},
                     output_shape=index_output_shape, name='H_noise')(model.output)

    # multiply by W_noise to create the reconstructed spectrogram of noise
    log_W_noise = K.log(1e-7 + W_noisy[:,r:]).eval()
    noise_est = TimeDistributed(DenseNonNegW(output_dim, use_bias=False, weights=[log_W_noise.T]), name='noise_est')(H_noise)

    if 'transform_before_irm' in params_unfolded_snmf:
        print("Using transform of '%s' before computing IRM" % (params_unfolded_snmf['transform_before_irm']))
        if params_unfolded_snmf['transform_before_irm'] == 'square':
            # ideal ratio mask is clean_est^2 / (clean_est^2 + noise_est^2):
            clean_est_xformed = Lambda( lambda x: K.square(x), output_shape=(maxseq, input_dim), name='clean_est_xformed')(clean_est)
            noise_est_xformed = Lambda( lambda x: K.square(x), output_shape=(maxseq, input_dim), name='noise_est_xformed')(noise_est)
            irm_predicted = divide_A_by_AplusB([clean_est_xformed, noise_est_xformed])
        else:
            ValueError("Unknown 'transform_before_irm' of '%s'" % (params_unfolded_snmf['transform_before_irm']))
    else:
        # ideal ratio mask is clean_est / (clean_est + noise_est):
        irm_predicted = divide_A_by_AplusB([clean_est, noise_est])

    model = Model(inputs=model.input, outputs=irm_predicted)

    # make sure the weights of the reconstruction layer are set:
    if 'transform_before_irm' in params_unfolded_snmf:
        model.layers[-5].set_weights([log_W_clean.T])
        model.layers[-4].set_weights([log_W_noise.T])
    else:
        model.layers[-3].set_weights([log_W_clean.T])
        model.layers[-2].set_weights([log_W_noise.T])

    return model



def build_lstm(params_lstm):
    mask_value=params_lstm['mask_value']
    maxseq=params_lstm['maxseq']
    input_dim=params_lstm['input_dim']
    output_dim=params_lstm['output_dim']

    model = Sequential()

    # input masking layer to deal with different len. seq.s
    model.add(Masking(mask_value=mask_value,
                      input_shape=(maxseq, input_dim)))

    # add K layers of LSTMS
    for k_layer in range(params_lstm['K_layers']):
        model.add(LSTM(params_lstm['hidden_dim'],
                       return_sequences=True,
                       input_shape=(maxseq, input_dim)))

    # linear output layer
    model.add(TimeDistributed(Dense(output_dim)))

    # sigmoid activation function
    model.add(TimeDistributed(Activation('sigmoid')))

    return model


def scoresMat_to_arrayAndLabels(scores):
    # reads scores from the score_audio.m Matlab function and converts them
    # to a numpy array and labels
    labels = [l[0] for l in scores['labels'][0]]
    S = scores['S']
    return S, labels


def print_scores(scores, labels, prefix=""):
    (nfiles, nscores) = scores.shape
    for iscore,label in enumerate(labels):
        score_mean = np.mean(scores[:, iscore])
        print("%sMean %s %.3f" % (prefix, label, score_mean))


def load_data_tensors(params_data, datafile, dataset, maxlen, downsample=1):
    if os.path.exists(datafile):
        print "Loading %s data from datafile '%s'..." % (dataset, datafile)
        f = h5py.File(datafile, "r")
        x = f['x_' + dataset][:]
        y = f['y_' + dataset][:]
        mask = f['mask_' + dataset][:]
        f.close()
    else:
        params_data_load = copy.deepcopy(params_data)
        params_data_load.update({'maxlen': maxlen})
        x, y, mask = load_data(params_data_load, dataset=dataset, downsample=downsample)
        print "Saving %s data to datafile '%s'" % (dataset, datafile)
        f = h5py.File(datafile, "w")
        f.create_dataset('x_' + dataset, data=x)
        f.create_dataset('y_' + dataset, data=y)
        f.create_dataset('mask_' + dataset, data=mask)
        f.close()

    return x, y, mask


def kl_div(x, y):
    log_x = np.log(1e-9 + x)
    log_y = np.log(1e-9 + y)
    return x*log_x - x*log_y - x + y

def beta_div(x, y, beta):
    if beta==1.:
        return kl_div(x, y)
    elif beta==0.:
        return (x/y) - np.log(1e-9 + x) + np.log(1e-9 + y) - 1
    else:
        return (1./(beta*(beta-1.)))*( (x**beta) \
                                       + (beta-1)*(y**beta) \
                                       - beta*x*(y**(beta-1)) \
                                     )

def ista_ed(x, W, H, lam1, alph, K, verbose=True):

    soft = lambda x: np.maximum(0, x)

    xest = np.dot(W, H)
    if verbose:
        div = np.sum( 0.5*((x-xest)**2) )
        cost = div + lam1*np.sum(H)
        print "ISTA with ED: k=%d, div %e cost %e" % (0, div, cost)
    for k in range(K):
        H = soft( -lam1/alph + H + (1./alph)*np.dot( W.T, x - xest ) )
        xest = np.dot(W, H)
        if verbose:
            div = np.sum( 0.5*((x-xest)**2) )
            cost = div + lam1*np.sum(H)
            print "ISTA with ED: k=%d, div %e cost %e" % (k+1, div, cost)
    return H


def ista_kl(x, W, H, lam1, alph, K, verbose=True):

    soft = lambda x: np.maximum(0, x)

    xest = np.dot(W, H)
    if verbose:
        div = np.sum(kl_div(x, xest))
        cost = div + lam1*np.sum(H)
        print "ISTA with KL-div: k=%d, div %e cost %e" % (0, div, cost)
    for k in range(K):
        H = soft( -lam1/alph + H + (1./alph)*np.dot( W.T, x/xest - 1 ) )
        xest = np.dot(W, H)
        if verbose:
            div = np.sum(kl_div(x, xest))
            cost = div + lam1*np.sum(H)
            print "ISTA with KL-div: k=%d, div %e cost %e" % (k+1, div, cost)
    return H


def ista_beta(x, W, H, lam1, alph, K, beta, verbose=True):

    soft = lambda x: np.maximum(0, x)

    xest = np.dot(W, H)
    if verbose:
        div = np.sum(beta_div(x, xest, beta))
        cost = div + lam1*np.sum(H)
        print "ISTA with beta-div (beta=%.2f): k=%d, div %e cost %e" % (beta, 0, div, cost)
    for k in range(K):
        H = soft( -lam1/alph + H + (1./alph)*np.dot( W.T, x*(xest**(beta-2.)) - (xest**(beta-1.)) ) )
        xest = np.dot(W, H)
        if verbose:
            div = np.sum(beta_div(x, xest, beta))
            cost = div + lam1*np.sum(H)
            print "ISTA with beta-div (beta=%.2f): k=%d, div %e cost %e" % (beta, k+1, div, cost)
    return H


def main(argv):
    configfile = ''
    configfile_data = ''
    helpstring = 'enhance_snmf.py -c <config YAML file> -d <data setup YAML file>'
    try:
        opts, args = getopt.getopt(argv,"hc:d:")
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpstring)
            sys.exit()
        elif opt in ("-c","--config"):
            configfile=arg
        elif opt in ("-d","--data"):
            configfile_data=arg

    print("Config file is %s" % configfile)
    print("Data setup config file is %s" % configfile_data)

    if os.path.exists(configfile):

        with open(configfile) as f:
            config_loaded = yaml.load(f.read())

    if os.path.exists(configfile_data):

        with open(configfile_data) as f:
            config_data_loaded = yaml.load(f.read())


    config={}

    if not (configfile_data == ''):
        config['params_data'] = config_data_loaded
    else:
        config['params_data'] = {}
        # paths for taskfiles used to load training/validation/test data
        config['params_data']['taskfile_x_train']='taskfile_chime2_train_noisy.txt'
        config['params_data']['taskfile_y_train']='taskfile_chime2_train_clean.txt'
        config['params_data']['taskfile_x_valid']='taskfile_chime2_valid_noisy.txt'
        config['params_data']['taskfile_y_valid']='taskfile_chime2_valid_clean.txt'
        config['params_data']['taskfile_x_test']='taskfile_chime2_test_noisy.txt'
        config['params_data']['taskfile_y_test']='taskfile_chime2_test_clean.txt'

        # transformations on complex-valued STFTs for noisy x and reference y
        config['params_data']['transform_x']='mag'
        config['params_data']['transform_y']='mag'

        # STFT parameters
        config['params_data']['params_stft']={'N': 512, 'hop': 128, 'nch': 1}

        # maximum length of training sequences
        config['params_data']['maxlen']=500 

        # downsampling factor for training data
        #config['params_data']['downsample'] = 10
        config['params_data']['downsample'] = 1
        
        config['params_data']['datafile_train']='chime2_si_tr_s_ds%d.hdf5' % (config['params_data']['downsample'])
        #config['datafile_valid']='chime2_si_dt_05_ds%d.hdf5' % (config['downsample'])
        config['params_data']['datafile_valid']='chime2_si_dt_05.hdf5'
        config['params_data']['datafile_test'] ='chime2_si_et_05.hdf5'


    params_loaded = None
    if not (configfile==''):
        if 'unfolded_snmf' in configfile:
            config['model'] = 'unfolded_snmf'
            params_loaded = config_loaded
        elif 'snmf' in configfile:
            config['model'] = 'snmf'
            params_loaded = config_loaded
        elif 'lstm' in configfile:
            config['model'] = 'lstm'
            params_loaded = config_loaded

    else:
        # choose a model, options are 'snmf', 'unfolded_snmf', 'lstm':
        config['model'] = 'snmf'
        #config['model'] = 'unfolded_snmf'
        #config['model'] = 'lstm'

    input_dim = config['params_data']['params_stft']['N']/2 + 1
    output_dim = input_dim

    if config['model'] == 'snmf':

        # number of basis vectors for speech and noise 
        # so total number of basis vectors is 2r
        #config['r'] = 100
        config['r'] = 1000

        # \lambda_1, the regularization weight on sparsity
        #config['sparsity'] = [0.01, 0.03, 0.1, 0.3, 1., 3., 5., 10.]
        #config['sparsity'] = [1., 3., 5., 10.]
        config['sparsity'] = [1.]

        # parameters for well-done sparse NMF multiplicative updates
        if not (params_loaded is None):
            config['params_snmf'] = params_loaded
            config['r'] = params_loaded['r']
            if len(config['sparsity']) == 1:
                config['sparsity'] = [params_loaded['sparsity']]
        else:
            config['params_snmf']= {'cf': 'ed',
                                    #'cf': 'kl',
                                    #'cf': 'is',
                                    'sparsity': config['sparsity'][0],
                                    'max_iter': 1000.,
                                    #'max_iter': 100.,
                                    'conv_eps': 1e-4,
                                    'display': 0.,
                                    'random_seed': 2016.,
                                    'r': config['r']}

        snmf_gpuIndex = int(re.findall("gpu(\d+)", theano.config.device)[0])+1 #note that Matlab uses 1-indexing for GPUs

    elif config['model'] == 'unfolded_snmf':

        # number of basis vectors for speech and noise 
        # so total number of basis vectors is 2r
        #config['r'] = 100
        #config['r'] = 500
        config['r'] = 1000

        # parameters for well-done sparse NMF multiplicative updates
        config['params_snmf']= {'cf': 'ed',
                                'sparsity': 1.,
                                'max_iter': 1000.,
                                'conv_eps': 1e-4,
                                'display': 0.,
                                'random_seed': 2016.,
                                'r': config['r']}

        snmf_gpuIndex = int(re.findall("gpu(\d+)", theano.config.device)[0])+1 #note that Matlab uses 1-indexing for GPUs

        # parameters for unfolded SNMF
        if not (params_loaded is None):
            # we have loaded parameters for unfolded SNMF from configfile
            config['params_unfolded_snmf'] = params_loaded
            config['r'] = params_loaded['r']
            config['params_snmf']['r'] = params_loaded['r']
            config['params_snmf']['sparsity'] = params_loaded['lam1']
        else:
            alph = 50
            if config['r']==100:
                alph = 50.
            elif config['r']==500:
                alph = 200.
            elif config['r']==1000:
                alph = 400.

            config['params_unfolded_snmf']= {'K_layers': 2,
                                             'loss' : 'mse_of_masked',
                                             #'epochs': 1200,
                                             'epochs': 0,
                                             'batch_size': 32,
                                             'learning_rate': 1e-3,
                                             'clipnorm': 0.,
                                             'optimizer': 'adam',
                                             'patience': 50,
                                             #'decay': 2e-4,
                                             'r': int(config['params_snmf']['r']),
                                             'lam1' : config['params_snmf']['sparsity'],
                                             'alph' : alph,
                                             #'untie_alph' : True,
                                             #'params_untied': ['log_D', 'log_alph'],
                                             #'params_trainable': ['log_D', 'log_alph']
                                             #'params_untied': ['log_D', 'log_alph'],
                                             'params_untied': ['log_D', 'log_alph'],
                                             'params_trainable': ['log_D', 'log_alph']
                                             #'transform_before_irm': 'square'
                                             #'savefile_init': 'data_setup_db3355248efc7ce949ff0bc5206f0a81/models/model_unfolded_snmf_2767a99603c236107da44c5c2083df31.hdf5'
                                             #'pretrain_with_snmf_cost': True
                                            }

    elif config['model'] == 'lstm':

        # parameters for LSTM network
        if not (params_loaded is None):
            # we have loaded parameters for LSTM from configfile
            config['params_lstm'] = params_loaded
        else:
            config['params_lstm'] = {'K_layers': 5,
                                     'hidden_dim': 250,
                                     'loss' : 'mse_of_masked',
                                     'epochs': 400,
                                     'batch_size': 32,
                                     'learning_rate': 1e-4,
                                     'clipnorm': 1.,
                                     'optimizer': 'adam',
                                     'patience': 50}

    else:
        ValueError("Unknown 'model' of '%s'" % config['model'])


    # some global flags
    #---------------------
    path_dicts = '/data1/snmf/'
    path_data = '/data1/'
    verbose = True 
    save_H = False # controls if we save H for SNMF training
    
    # if flag_recompute is True, reruns training of models, 
    # regardless of the presence of an existing savefile:
    flag_recompute = False
    
    flag_rescore = False    # force rescoring, even if .mat files exist
    flag_score_valid = True # reconstruct and score validaiton data if True
    flag_score_test = True  # reconstruct and score test data if True
    #---------------------
    # end global flags


    # check directory names, create directories, and save off
    # configurations
    #---------------------
    # make sure the 'experiments' directory exists
    if not os.path.exists('experiments'):
        os.makedirs('experiments')        

    # create a hash and a folder for the current data setup
    params_data = config['params_data']
    params_data_to_hash = copy.deepcopy(params_data)
    # remove system-specific fields from the data parameter dictionary:
    del params_data_to_hash['datafile_train']
    del params_data_to_hash['datafile_valid']
    del params_data_to_hash['datafile_test']
    hash_params_data = hashlib.md5(json.dumps(params_data_to_hash, sort_keys=True)).hexdigest()
    folder_exp = 'data_setup_%s' % hash_params_data
    if not os.path.exists(folder_exp):
        os.makedirs(folder_exp)
    # write the data configuration to a YAML file to remember it
    configfile_data = folder_exp + '/params_data.yaml'
    with open(configfile_data, 'wb') as f:
        yaml.dump(params_data, f)

    # create directories to store trained SNMF dictionaries
    path_dicts = path_dicts + '/' + folder_exp + '/'
    if not os.path.exists(path_dicts):
        os.makedirs(path_dicts)

    # create directories to store results 
    folders = [folder_exp + '/' + f for f in ['configs', 'history', 'models', 'scores']]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    #--------------------
    #end checking and creating directories and saving configurations


    # specify parameters for load the data
    #--------------------
    if (config['model'] == 'snmf'):
        """
        # datafile used to store reshaped data for quick loading
        datafile = path_data + 'data_chime2_tr_dt_ds%d.hdf5' % (params_data['downsample'])
        datafile_test = path_data + 'data_chime2_et.hdf5'
        maxlen_valid = None
        """
        snmf_gpuIndex = int(re.findall("gpu(\d+)", theano.config.device)[0])+1 #note that Matlab uses 1-indexing for GPUs
    """
    else:
    """
    datafile_train = path_data + 'data_chime2_tr_ds%d_maxlen%d.hdf5' % (params_data['downsample'], params_data['maxlen'])
    # validation data needs to have same maxlen as train
    datafile_valid = path_data + 'data_chime2_dt_maxlen%d.hdf5' % (params_data['maxlen'])
    datafile_valid_no_maxlen = path_data + 'data_chime2_dt.hdf5'
    datafile_test = path_data + 'data_chime2_et.hdf5'
    maxlen_valid = params_data['maxlen']

    # build AudioDataset objects, which will be used for reconstruction
    # and scoring   
    D_train=AudioDataset(params_data['taskfile_x_train'], params_data['taskfile_y_train'], datafile=params_data['datafile_train'], params_stft=params_data['params_stft'], downsample=params_data['downsample'])
    D_valid=AudioDataset(params_data['taskfile_x_valid'], params_data['taskfile_y_valid'], datafile=params_data['datafile_valid'], params_stft=params_data['params_stft'])
    if flag_score_test:
        D_test=AudioDataset(params_data['taskfile_x_test'], params_data['taskfile_y_test'], datafile=params_data['datafile_test'], params_stft=params_data['params_stft'])
    #--------------------
    # end specifying parameters for loading data


    # now let's get to building, training, and evaluating the models...
    #----------------------
    if config['model'] == 'snmf':

        params_snmf = config['params_snmf']

        if 'spectrogram_power' in params_snmf:
            spectrogram_power = params_snmf['spectrogram_power']
        else:
            spectrogram_power = np.float32(1.)


        # load training data tensors, if we need them
        train_data_loaded = False
        params_snmf_cur = copy.deepcopy(params_snmf)
        for sparsity in config['sparsity']:
            # iterate through all the dictionaries we'll need to load, and
            # only load the training data if at least one of the savefiles
            # doesn't exist
            params_snmf_cur.update({'sparsity':sparsity})
            savefile_W_hickle = get_snmf_savefile(params_snmf_cur, path_dicts=path_dicts)
            if os.path.exists(savefile_W_hickle) and (not flag_recompute):
                # we don't need to train
                y_train_frames = None
                x_train_frames = None
            else:
                # load the training data
                x_train, y_train, mask_train = load_data_tensors(params_data, datafile_train, 'train', params_data['maxlen'], downsample=params_data['downsample'])

                x_train = x_train**spectrogram_power
                y_train = y_train**spectrogram_power

                x_train_frames = masked_seqs_to_frames(x_train, mask_train)
                y_train_frames = masked_seqs_to_frames(y_train, mask_train)
                break
                train_data_loaded = True                

        # load validation data tensors, if we need them
        if flag_score_valid:
            # convert the masked sequence tensors to matrices of shape (n_freq, n_frames)
            x_valid, y_valid, mask_valid = load_data_tensors(params_data, datafile_valid, 'valid', maxlen_valid)

            x_valid = x_valid**spectrogram_power
            y_valid = y_valid**spectrogram_power

            x_valid_frames = masked_seqs_to_frames(x_valid, mask_valid)
            y_valid_frames = masked_seqs_to_frames(y_valid, mask_valid)

        # load test data tensors, if we need them
        if flag_score_test:
            # convert the masked sequence tensors to matrices of shape (n_freq, n_frames)
            x_test, y_test, mask_test = load_data_tensors(params_data, datafile_test, 'test', None)

            x_valid = x_valid**spectrogram_power
            y_valid = y_valid**spectrogram_power

            x_test_frames = masked_seqs_to_frames(x_test, mask_test)

        for sparsity in config['sparsity']:

            params_snmf = config['params_snmf']
            
            params_snmf['sparsity'] = sparsity
            
            # train the dictionary W_noisy=[W_clean, W_noise]
            W_noisy, H_noisy, obj_snmf_noisy = train_snmf(y_train_frames, x_train_frames, params_snmf, gpuIndex=snmf_gpuIndex, verbose=verbose, flag_recompute=flag_recompute, path_dicts=path_dicts, save_H=save_H)

            if (sparsity == config['sparsity'][-1]) and train_data_loaded:
                # clear the memory of large training tensors:
                del x_train_frames
                del x_train
                del y_train_frames
                del y_train

            # extract subdictionaries for clean speech and noise
            r = int(params_snmf['r'])
            W_clean=W_noisy[:, :r]
            W_noise=W_noisy[:, r:]
            #description_enh = "tune_snmf_sparsity%.3f" % sparsity
            hash_params_snmf = hashlib.md5(json.dumps(params_snmf, sort_keys=True)).hexdigest()
            description_enh = "snmf_%s" % hash_params_snmf
            paramsfile = folder_exp + ("/configs/params_snmf_%s.yaml" % hash_params_snmf)
            with open(paramsfile, 'wb') as f:
                yaml.dump(params_snmf, f)
            print("paramsfile is %s" % paramsfile)

            histfile = folder_exp + "/history/history_snmf_%s" % hash_params_snmf

            if flag_score_valid:
                # infer H on validation set.
                params_snmf_infer = copy.deepcopy(params_snmf)
                params_snmf_infer['r'] = 2*params_snmf['r']
                idx_update = np.zeros(2*int(params_snmf['r']), dtype=bool)
                params_snmf_infer.update({'init_w': W_noisy,'w_update_ind' : idx_update, 'conv_eps' : 0., 
                'max_iter' : 200.
                #'max_iter' : 25.
                })
                _, H_valid, obj_snmf_valid = sparse_nmf_matlab(x_valid_frames, params_snmf_infer, verbose=verbose, gpuIndex=snmf_gpuIndex)

                # compute the ideal ratio mask
                H_clean=H_valid[:r, :]
                H_noise=H_valid[r:, :]
                clean_est = np.dot(W_clean, H_clean)
                noise_est = np.dot(W_noise, H_noise)
                irm = clean_est/(1e-9+clean_est+noise_est)

                # compute and save off validation loss
                val_loss = np.mean( (irm*x_valid_frames - y_valid_frames)**2 )
                print "Signal approximation loss for SNMF on validation set is %.4f, saving to histfile '%s'" % (val_loss, histfile)
                with open(histfile, 'wb') as f:
                    cPickle.dump({'on_epoch_end': {'val_loss': [val_loss]}}, f)

                # reconstruct the enhanced audio
                n_wavfiles = len(D_valid.x_wavfiles)
                description_enh_valid = description_enh + '_valid'
                for j in range(n_wavfiles):
                    D_valid.reconstruct_audio(description_enh_valid, idx=j,irm=irm[:,D_valid.fidx[j,0]:D_valid.fidx[j,1]])

                # score enhanced validation data
                snrs = ["m6dB","m3dB","0dB","3dB","6dB","9dB"]
                for snr in snrs:
                    print("")
                    print("  Scoring data for SNR of %s" % snr)
                    scores_mat = D_valid.score_audio(description=description_enh_valid, snr=snr, verbose=False, datadir=(folder_exp+"/"), flag_rescore=flag_rescore)
                    scores, labels = scoresMat_to_arrayAndLabels(scores_mat)
                    print_scores(scores, labels, prefix="  ")
                    if snr==snrs[0]:
                        scores_mean = np.sum(scores, axis=0, keepdims=True)
                    else:
                        scores_mean = scores_mean + np.sum(scores, axis=0, keepdims=True)

                print("")
                print("  Overall scores (validation):")
                scores_mean = scores_mean/n_wavfiles
                print_scores(scores_mean, labels, prefix="  ")

            if flag_score_test:
                # infer H on validation set.
                params_snmf_infer = copy.deepcopy(params_snmf)
                params_snmf_infer['r'] = 2*params_snmf['r']
                idx_update = np.zeros(2*int(params_snmf['r']), dtype=bool)
                params_snmf_infer.update({'init_w': W_noisy,'w_update_ind' : idx_update, 'conv_eps' : 0., 
                'max_iter' : 200.
                #'max_iter' : 25.
                })
                _, H_test, obj_snmf_test = sparse_nmf_matlab(x_test_frames, params_snmf_infer, verbose=verbose, gpuIndex=snmf_gpuIndex)

                # compute the ideal ratio mask
                H_clean=H_test[:r, :]
                H_noise=H_test[r:, :]
                clean_est = np.dot(W_clean, H_clean)
                noise_est = np.dot(W_noise, H_noise)
                irm = clean_est/(1e-9+clean_est+noise_est)

                # reconstruct the enhanced audio
                n_wavfiles = len(D_test.x_wavfiles)
                description_enh_test = description_enh + '_test'
                for j in range(n_wavfiles):
                    D_test.reconstruct_audio(description_enh_test, idx=j,irm=irm[:,D_test.fidx[j,0]:D_test.fidx[j,1]])

                # score enhanced validation data
                snrs = ["m6dB","m3dB","0dB","3dB","6dB","9dB"]
                for snr in snrs:
                    print("")
                    print("  Scoring data for SNR of %s" % snr)
                    scores_mat = D_test.score_audio(description=description_enh_test, snr=snr, verbose=False, datadir=(folder_exp+"/"), flag_rescore=flag_rescore)
                    scores, labels = scoresMat_to_arrayAndLabels(scores_mat)
                    print_scores(scores, labels, prefix="  ")
                    if snr==snrs[0]:
                        scores_mean = np.sum(scores, axis=0, keepdims=True)
                    else:
                        scores_mean = scores_mean + np.sum(scores, axis=0, keepdims=True)

                print("")
                print("  Overall scores (test):")
                scores_mean = scores_mean/n_wavfiles
                print_scores(scores_mean, labels, prefix="  ")
            
            # end of loop over different sparsity levels

        return
        #------------------------------ 
        # end of SNMF model section


    elif config['model'] == 'unfolded_snmf':

        params_snmf = config['params_snmf']

        r = int(params_snmf['r'])
        #input_dim = x_train.shape[2]
       
        train_data_loaded = False
        valid_data_loaded = False

        # load the training data if we need to train SNMF
        #-------------------
        # save off the SNMF parameters
        hash_params_snmf = hashlib.md5(json.dumps(params_snmf, sort_keys=True)).hexdigest()
        paramsfile_snmf = folder_exp + ("/configs/params_snmf_%s.yaml" % hash_params_snmf)
        with open(paramsfile_snmf, 'wb') as f:
            yaml.dump(params_snmf, f)
        print("paramsfile for SNMF is %s" % paramsfile_snmf)

        # get the name of the savefile for SNMF
        savefile_W_hickle = get_snmf_savefile(params_snmf, path_dicts=path_dicts)
        if os.path.exists(savefile_W_hickle) and (not flag_recompute):
            # the savefile exists and we're not recomputing, so assign
            # dummy values for the training data so we don't take up
            # a lot of memory
            y_train_frames = None
            x_train_frames = None
            train_data_loaded = False
        else:
            # the savefile doesn't exist or we're recomputing, so load up 
            # the training data tensors
            x_train, y_train, mask_train = load_data_tensors(params_data, datafile_train, 'train', params_data['maxlen'], downsample=params_data['downsample'])

            # convert the masked sequence tensors to matrices of shape (n_freq, n_frames)
            x_train_frames = masked_seqs_to_frames(x_train, mask_train)
            y_train_frames = masked_seqs_to_frames(y_train, mask_train)
            train_data_loaded = True

        # train SNMF build the unfolded SNMF model
        #-------------------
        print("")
        print("Building the unfolded SNMF model...")
        
        # train (or load) the dictionary W_noisy=[W_clean, W_noise]
        print("Training SNMF model...")
        W_noisy, H_noisy, obj_snmf_noisy = train_snmf(y_train_frames, x_train_frames, params_snmf, gpuIndex=snmf_gpuIndex, verbose=verbose, flag_recompute=flag_recompute, path_dicts=path_dicts, save_H=save_H)
        
        # clear the memory of large training tensors:
        del x_train_frames
        del y_train_frames

        print("SNMF cost %e, SNMF div %e" % (float(obj_snmf_noisy['cost'][-1]), float(obj_snmf_noisy['div'][-1])))
        #print("SNMF mean cost %e, SNMF mean div %e" % (float(obj_snmf_noisy['cost'][-1])/(np.sum(mask_train)*input_dim), float(obj_snmf_noisy['div'][-1])/(np.sum(mask_train)*input_dim)))

        # build the unfolded SNMF model
        params_unfolded_snmf = config['params_unfolded_snmf']
        params_unfolded_snmf_build = copy.deepcopy(params_unfolded_snmf)
        params_unfolded_snmf_build['input_dim'] = input_dim
        params_unfolded_snmf_build['hidden_dim'] = 2*r
        params_unfolded_snmf_build['output_dim'] = output_dim
        params_unfolded_snmf_build['mask_value'] = -1. # when transform_x, transform_y are 'mag'
        params_unfolded_snmf_build['maxseq'] = params_data['maxlen']
        params_unfolded_snmf_build['K_layers'] = params_unfolded_snmf['K_layers']
        params_unfolded_snmf_build['W'] = W_noisy
        
        print("")
        print("Using a DR-NMF model with parameters:")
        pprint(params_unfolded_snmf_build)
    

        model = build_unfolded_snmf(params_unfolded_snmf_build)

        # optional pretraining with SNMF cost function:
        if 'pretrain_with_snmf_cost' in params_unfolded_snmf and params_unfolded_snmf['pretrain_with_snmf_cost']:
            h_estimated = model.layers[-6].output; h_estimated.name='h_estimated'
            x_recon = add([model.layers[-3].output, model.layers[-2].output]); x_recon.name='x_recon'
            model_pretrain = Model(inputs=model.input, outputs=[x_recon, h_estimated])
            def l1_of_output(y_true, y_pred):
                return K.mean(K.abs(y_pred), axis=-1)

            loss_pretrain = ['mse', l1_of_output]
            lam1 = params_unfolded_snmf['lam1']
            # set loss weights for SNMF code. Note the adjustment to lam1,
            # which accounts for the fact that keras takes the mean of the
            # losses
            loss_weights_pretrain = [0.5, lam1*np.float32(2*r)/input_dim]


        # set up the model for the desired training loss function
        if params_unfolded_snmf['loss'] == 'mse_of_masked':
 
            output_masked = multiply([model.layers[0].output, model.output])
            model = Model(inputs=model.input, outputs=output_masked)
            
            loss = 'mse'

        else:
            ValueError("Unknown 'loss' of '%s'" % params_unfolded_snmf['loss'])

        
        # set up the optimizer for training
        if params_unfolded_snmf['optimizer'] == 'adam':
            if 'decay' in params_unfolded_snmf:
                decay = params_unfolded_snmf['decay']
            else:
                decay = 0.
            optimizer = Adam(lr=params_unfolded_snmf['learning_rate'], clipnorm=params_unfolded_snmf['clipnorm'], decay=decay)
        else:
            ValueError("Unknown 'optimizer' of '%s'" % params_unfolded_snmf['optimizer'])

        # if doing pretraining, compile the pretrain model
        if 'pretrain_with_snmf_cost' in params_unfolded_snmf and params_unfolded_snmf['pretrain_with_snmf_cost']:
            print("Compiling pretraining model...")
            model_pretrain.compile(loss=loss_pretrain,
                                   loss_weights=loss_weights_pretrain,
                                   optimizer=optimizer,
                                   sample_weight_mode='temporal')

        # compile the unfolded SNMF model
        print("Compiling unfolded SNMF model...")
        model.compile(loss=loss,
                      optimizer=optimizer,
                      sample_weight_mode='temporal')

        # train the model
        #-------------------
        print("")
        print("Training the unfolded SNMF model...")
        hash_params_unfolded_snmf = hashlib.md5(json.dumps(params_unfolded_snmf, sort_keys=True)).hexdigest()
        paramsfile = folder_exp + "/configs/params_unfolded_snmf_%s.yaml" % hash_params_unfolded_snmf
        with open(paramsfile, 'wb') as f:
            yaml.dump(params_unfolded_snmf, f)
        print("paramsfile is %s" % paramsfile)

        savefile = folder_exp + "/models/model_unfolded_snmf_%s.hdf5" % hash_params_unfolded_snmf
        histfile = folder_exp + "/history/history_unfolded_snmf_%s" % hash_params_unfolded_snmf

        # if doing pretraining, train the pretrain model
        if 'pretrain_with_snmf_cost' in params_unfolded_snmf and params_unfolded_snmf['pretrain_with_snmf_cost']:
            savefile_pretrain = savefile.replace(".hdf5", "_pretrain.hdf5")
            histfile_pretrain = histfile + "_pretrain"
            print("savefile_pretrain is %s" % savefile_pretrain)
            print("histfile_pretrain is %s" % histfile_pretrain)
            if flag_recompute or (not os.path.exists(savefile_pretrain)):
                history = LossHistory(histfile_pretrain)
                checkpointer = ModelCheckpoint(filepath=savefile_pretrain, verbose=2, save_best_only=True, save_weights_only=True)
                earlystopping = EarlyStopping(monitor='val_loss', patience=params_unfolded_snmf['patience'], verbose=1, mode='auto')

                # load training and validation data:
                if not train_data_loaded:
                    x_train, y_train, mask_train = load_data_tensors(params_data, datafile_train, 'train', params_data['maxlen'], downsample=params_data['downsample'])
                    train_data_loaded = True
                if not valid_data_loaded:
                    x_valid, y_valid, mask_valid = load_data_tensors(params_data, datafile_valid, 'valid', maxlen_valid)
                    valid_data_loaded = True

                mask_train = np.squeeze(mask_train)
                mask_valid = np.squeeze(mask_valid)
                model_pretrain.fit(x_train, [x_train, x_train], sample_weight=[mask_train, mask_train],
                                   batch_size=params_unfolded_snmf['batch_size'],
                                   epochs=params_unfolded_snmf['epochs'],
                                   verbose=1,
                                   validation_data=(x_valid, [x_valid, x_valid], [mask_valid, mask_valid]),
                                   callbacks=[history, checkpointer, earlystopping])
            else:
                print("Savefile for pretraining '%s' already exists, loading weights..." % (savefile_pretrain))

            # load the best weights from the model savefile
            model_pretrain.load_weights(savefile_pretrain)
            model.load_weights(savefile_pretrain)

        print("savefile is %s" % savefile)
        print("histfile is %s" % histfile)

        # if a savefile is specified for model initialization weights,
        # load it up
        if 'savefile_init' in params_unfolded_snmf:
            print "Initialization savefile of '%s' specified, loading weights from that file..." % (params_unfolded_snmf['savefile_init'])
            model.load_weights(params_unfolded_snmf['savefile_init'])

        # if we are recomputing or a savefile doesn't exist, train the unfolded
        # SNMF model
        if flag_recompute or (not os.path.exists(savefile)):
            history = LossHistory(histfile)
            checkpointer = ModelCheckpoint(filepath=savefile, verbose=2, save_best_only=True, save_weights_only=True)
            earlystopping = EarlyStopping(monitor='val_loss', patience=params_unfolded_snmf['patience'], verbose=1, mode='auto')

            # load training and validation data, if they haven't been loaded
            # already
            if not train_data_loaded:
                x_train, y_train, mask_train = load_data_tensors(params_data, datafile_train, 'train', params_data['maxlen'], downsample=params_data['downsample'])
                train_data_loaded = True
            if not valid_data_loaded:
                x_valid, y_valid, mask_valid = load_data_tensors(params_data, datafile_valid, 'valid', maxlen_valid)
                valid_data_loaded = True

            # need to squeeze the masks to be the right shape
            mask_train = np.squeeze(mask_train)
            mask_valid = np.squeeze(mask_valid)

            # train the unfolded SNMF model
            model.fit(x_train, y_train, sample_weight=mask_train,
                      batch_size=params_unfolded_snmf['batch_size'],
                      epochs=params_unfolded_snmf['epochs'],
                      verbose=1,
                      validation_data=(x_valid, y_valid, mask_valid),
                      callbacks=[history, checkpointer, earlystopping])

            if params_unfolded_snmf['epochs'] == 0:
                model.save_weights(savefile)

        else:
            print("Savefile '%s' already exists, loading weights..." % (savefile))

        # load the best weights from the model savefile
        model.load_weights(savefile)

        # scoring
        #-------------------
        print("")
        print("Scoring the results...")

        print("Validation set:")
        # load up validation data tensors that aren't limited by maxlen
        print("  Loading data to reconstruct audio...")
        """
        params_data_valid = copy.deepcopy(params_data)
        params_data_valid.update({'maxlen': None})
        x_valid, y_valid, mask_valid = load_data(params_data_valid, dataset='valid')
        """
        x_valid, y_valid, mask_valid = load_data_tensors(params_data, datafile_valid_no_maxlen, 'valid', None)

        # predict the ideal ratio mask from the network
        print("  Predicting time-frequency masks using the trained model...")
        params_unfolded_snmf_build['maxseq'] = x_valid.shape[1]
        model_irm = build_unfolded_snmf(params_unfolded_snmf_build)
        model_irm.set_weights(model.get_weights())
        irm = np.zeros(x_valid.shape, dtype=np.float32)
        batch_size = 250
        start_idx=0
        while start_idx < irm.shape[0]:
            irm[start_idx:start_idx+batch_size,:,:] = model_irm.predict_on_batch(x_valid[start_idx:start_idx+batch_size,:,:])
            start_idx = start_idx + batch_size

        # reconstruct the enhanced audio
        print("  Reconstructing audio...")
        n_wavfiles = len(D_valid.x_wavfiles)
        description_enh_valid = ('unfolded_snmf_%s' % hash_params_unfolded_snmf) + '_valid'
        for j in range(n_wavfiles):
            len_cur = D_valid.fidx[j,1] - D_valid.fidx[j,0]
            D_valid.reconstruct_audio(description_enh_valid,
                                      idx=j,
                                      irm=irm[j,:len_cur,:].T)

        if flag_score_test:
            print("")
            print("Test set:")

            # load up evaluation data
            print("  Loading data to reconstruct audio...")
            x_test, y_test, mask_test = load_data_tensors(params_data, datafile_test, 'test', None)

            # predict the ideal ratio mask from the network
            print("  Predicting time-frequency masks using the trained model...")
            params_unfolded_snmf_build['maxseq'] = x_test.shape[1]
            model_irm = build_unfolded_snmf(params_unfolded_snmf_build)
            model_irm.set_weights(model.get_weights())
            irm = np.zeros(x_test.shape, dtype=np.float32)
            batch_size = 250
            start_idx=0
            while start_idx < irm.shape[0]:
                irm[start_idx:start_idx+batch_size,:,:] = model_irm.predict_on_batch(x_test[start_idx:start_idx+batch_size,:,:])
                start_idx = start_idx + batch_size

            # reconstruct the enhanced audio
            print("  Reconstructing audio...")
            n_wavfiles = len(D_test.x_wavfiles)
            description_enh_test = ('unfolded_snmf_%s' % hash_params_unfolded_snmf) + '_test'
            for j in range(n_wavfiles):
                len_cur = D_test.fidx[j,1] - D_test.fidx[j,0]
                D_test.reconstruct_audio(description_enh_test,
                                          idx=j,
                                          irm=irm[j,:len_cur,:].T)

        #------------------------------ 
        # end of unfolded SNMF model section


    elif config['model'] == 'lstm':
        print("")
        print("Using a LSTM model with parameters:")
        params_lstm = config['params_lstm']
        pprint(params_lstm)

        #input_dim = x_train.shape[2]
        #output_dim = y_train.shape[2]
        mask_value = -1. # when transform_x, transform_y are 'mag'
        maxseq = params_data['maxlen']

        train_data_loaded = False
        valid_data_loaded = False

        # build the model
        #-------------------
        print("")
        print("Building the model...")

        params_lstm_build = copy.deepcopy(params_lstm)
        params_lstm_build.update({'mask_value':mask_value, 'maxseq':maxseq, 'input_dim':input_dim, 'output_dim':output_dim})
        model = build_lstm(params_lstm_build)

        if params_lstm['loss'] == 'mse_of_masked':
            irm_predicted = model.output
            output_masked = multiply([model.layers[0].output, irm_predicted])
            model = Model(inputs=model.input, outputs=output_masked)
            
            loss = 'mse'
        else:
            ValueError("Unknown 'loss' of '%s'" % params_lstm['loss'])

        if params_lstm['optimizer'] == 'adam':
            optimizer = Adam(lr=params_lstm['learning_rate'], clipnorm=params_lstm['clipnorm'])
        else:
            ValueError("Unknown 'optimizer' of '%s'" % params_lstm['optimizer'])

        model.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal')


        # train the model
        #-------------------
        print("")
        print("Training the model...")
        hash_params_lstm = hashlib.md5(json.dumps(params_lstm, sort_keys=True)).hexdigest()
        paramsfile = folder_exp + "/configs/params_lstm_%s.yaml" % hash_params_lstm
        with open(paramsfile, 'wb') as f:
            yaml.dump(params_lstm, f)
        print("paramsfile is %s" % paramsfile)

        savefile = folder_exp + "/models/model_lstm_%s.hdf5" % hash_params_lstm
        histfile = folder_exp + "/history/history_lstm_%s" % hash_params_lstm
        print("savefile is %s" % savefile)
        print("histfile is %s" % histfile)

        if flag_recompute or (not os.path.exists(savefile)):
            history = LossHistory(histfile)
            checkpointer = ModelCheckpoint(filepath=savefile, verbose=2, save_best_only=True, save_weights_only=True)
            earlystopping = EarlyStopping(monitor='val_loss', patience=params_lstm['patience'], verbose=1, mode='auto')

            # load training and validation data:
            if not train_data_loaded:
                x_train, y_train, mask_train = load_data_tensors(params_data, datafile_train, 'train', params_data['maxlen'], downsample=params_data['downsample'])
                train_data_loaded = True
            if not valid_data_loaded:
                x_valid, y_valid, mask_valid = load_data_tensors(params_data, datafile_valid, 'valid', maxlen_valid)
                valid_data_loaded = True

            mask_train = np.squeeze(mask_train)
            mask_valid = np.squeeze(mask_valid)
            model.fit(x_train, y_train, sample_weight=mask_train,
                      batch_size=params_lstm['batch_size'],
                      epochs=params_lstm['epochs'],
                      verbose=1,
                      validation_data=(x_valid, y_valid, mask_valid),
                      callbacks=[history, checkpointer, earlystopping])
        else:
            print("Savefile '%s' already exists, loading weights..." % (savefile))

        # load the best weights from the model savefile
        model.load_weights(savefile)


        # scoring
        #-------------------
        print("")
        print("Scoring the results...")

        print("Validation set:")
        # load up validation data tensors that aren't limited by maxlen
        print("  Loading data to reconstruct audio for validation set...")
        """
        params_data_valid = copy.deepcopy(params_data)
        params_data_valid.update({'maxlen': None})
        x_valid, y_valid, mask_valid = load_data(params_data_valid, dataset='valid')
        """
        x_valid, y_valid, mask_valid = load_data_tensors(params_data, datafile_valid_no_maxlen, 'valid', None)

        # predict the ideal ratio mask from the network
        print("  Predicting time-frequency masks using the trained model...")
        params_lstm_build['maxseq'] = x_valid.shape[1]
        model_irm = build_lstm(params_lstm_build)
        model_irm.set_weights(model.get_weights())
        irm = np.zeros(x_valid.shape, dtype=np.float32)
        batch_size = 250
        start_idx=0
        while start_idx < irm.shape[0]:
            irm[start_idx:start_idx+batch_size,:,:] = model_irm.predict_on_batch(x_valid[start_idx:start_idx+batch_size,:,:])
            start_idx = start_idx + batch_size

        # reconstruct the enhanced audio
        print("  Reconstructing audio...")
        n_wavfiles = len(D_valid.x_wavfiles)
        description_enh_valid = ('lstm_%s' % hash_params_lstm) + '_valid'
        for j in range(n_wavfiles):
            len_cur = D_valid.fidx[j,1] - D_valid.fidx[j,0]
            D_valid.reconstruct_audio(description_enh_valid,
                                      idx=j,
                                      irm=irm[j,:len_cur,:].T)

        if flag_score_test:
            print("Test set:")
            # load up evaluation data
            print("  Loading data to reconstruct audio...")
            x_test, y_test, mask_test = load_data_tensors(params_data, datafile_test, 'test', None)

            # predict the ideal ratio mask from the network
            print("  Predicting time-frequency masks using the trained model...")
            params_lstm_build['maxseq'] = x_test.shape[1]
            model_irm = build_lstm(params_lstm_build)
            model_irm.set_weights(model.get_weights())
            irm = np.zeros(x_test.shape, dtype=np.float32)
            batch_size = 250
            start_idx=0
            while start_idx < irm.shape[0]:
                irm[start_idx:start_idx+batch_size,:,:] = model_irm.predict_on_batch(x_test[start_idx:start_idx+batch_size,:,:])
                start_idx = start_idx + batch_size

            # reconstruct the enhanced audio
            print("  Reconstructing audio...")
            n_wavfiles = len(D_test.x_wavfiles)
            description_enh_test = ('lstm_%s' % hash_params_lstm) + '_test'
            for j in range(n_wavfiles):
                len_cur = D_test.fidx[j,1] - D_test.fidx[j,0]
                D_test.reconstruct_audio(description_enh_test,
                                          idx=j,
                                          irm=irm[j,:len_cur,:].T)
 
        #------------------------------ 
        # end of LSTM model section

    else:
        ValueError("Unknown 'model' of '%s'" % config['model'])


    # score the enhanced data
    #-------------------------
    snrs = ["m6dB","m3dB","0dB","3dB","6dB","9dB"]

    # score enhanced validation data
    for snr in snrs:
        print("")
        print("  Scoring data for SNR of %s" % snr)
        scores_mat = D_valid.score_audio(description=description_enh_valid, snr=snr, verbose=False, datadir=(folder_exp+"/"), flag_rescore=flag_rescore)
        scores, labels = scoresMat_to_arrayAndLabels(scores_mat)
        print_scores(scores, labels, prefix="  ")
        if snr==snrs[0]:
            scores_mean = np.sum(scores, axis=0, keepdims=True)
        else:
            scores_mean = scores_mean + np.sum(scores, axis=0, keepdims=True)

    print("")
    print("  Overall scores (validation):")
    n_wavfiles = len(D_valid.x_wavfiles)
    scores_mean = scores_mean/n_wavfiles
    print_scores(scores_mean, labels, prefix="  ")

    if flag_score_test:
        # score enhanced test data
        for snr in snrs:
            print("")
            print("  Scoring data for SNR of %s" % snr)
            scores_mat = D_test.score_audio(description=description_enh_test, snr=snr, verbose=False, datadir=(folder_exp+"/"), flag_rescore=flag_rescore)
            scores, labels = scoresMat_to_arrayAndLabels(scores_mat)
            print_scores(scores, labels, prefix="  ")
            if snr==snrs[0]:
                scores_mean = np.sum(scores, axis=0, keepdims=True)
            else:
                scores_mean = scores_mean + np.sum(scores, axis=0, keepdims=True)

        print("")
        print("  Overall scores (test):")
        n_wavfiles = len(D_test.x_wavfiles)
        scores_mean = scores_mean/n_wavfiles
        print_scores(scores_mean, labels, prefix="  ")


if __name__ == "__main__":
    main(sys.argv[1:])

