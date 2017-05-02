import numpy as np
import scipy.io as sio
import yaml
import cPickle
import h5py

from enhance_snmf import scoresMat_to_arrayAndLabels



def print_row(model, hash_model, hash_data, datasets, snrs, scores_to_print, properties, scoredir='scores/', print_per_snr=True, model_label=None):

    if model_label is None:
        model_label = model

    row = ""

    # print the properties of the model
    datadir = 'data_setup_%s/' % (hash_data)
    with open(datadir + 'params_data.yaml', 'rb') as f:
        params_data = yaml.load(f)
    if len(properties)>0:

        if not(len(properties)==1 and properties[0]=='model'):
            with open(datadir + 'configs/params_%s_%s.yaml' % (model, hash_model), 'rb') as f:
                params_model = yaml.load(f)

        for prop in properties:
            if prop == "model":
                row = row + model_label
            elif prop == 'K_layers':
                row = row + ('%d' % (params_model['K_layers']))
            elif prop == "num_train":
                row = row + ('%d' % np.round(7138./params_data['downsample']))
            elif prop == "num_params":
                modelfile = datadir + 'models/' + ('model_%s_%s.hdf5' % (model, hash_model))
                f = h5py.File(modelfile)
                num_params = 0
                for key in f.keys():
                    for key2 in f[key]:
                        for key3 in f[key][key2].keys():
                            if 'params_trainable' in params_model.keys():

                                params_trainable = params_model['params_trainable'] + ['h0']

                                param_is_trainable = False
                                for name in params_trainable:
                                    # for each name of the trainable weights
                                    if name in key3:
                                        # only count these weights as trainable if the parameter is trainable
                                        param_is_trainable = True
                                        #print "%s is trainable, since it contains %s" % (key3,name)
                                if not param_is_trainable:
                                    continue 
                            num_params = num_params + np.prod(f[key][key2][key3].shape)
                row = row + ('%d' % num_params)
            elif prop == "hidden_dim":
                if 'hidden_dim' in params_model:
                    row = row + ('%d' % (params_model['hidden_dim']))
                elif 'r' in params_model:
                    row = row + ('%d' % (2*params_model['r']))
                else:
                    ValueError("params_model does not have keys 'hidden_dim' or 'r'!")
            elif prop == "val_loss":
                histfile=datadir + '/history/' + ('history_%s_%s' % (model, hash_model))
                with open(histfile,'rb') as f:
                    hist = cPickle.load(f)
                val_loss = np.min(hist['on_epoch_end']['val_loss'])
                row = row + ('%.4f' % (val_loss))
            else:
                ValueError("Unknown property of '%s'" % prop)

            row = row + " & "


    # extract the scores and their labels from the score files
    scores_all_datasets = {}
    scores_per_snr_datasets = {}
    for dataset in datasets:
        scores_all = None
        scores_per_snr = {}
        for snr in snrs:
            scorefile_cur = scoredir + ("scores_%s_%s_%s_%s.mat" % (model, hash_model, dataset, snr))
            #print("Loading scorefile '%s'..." % scorefile_cur)
            scores_mat = sio.loadmat(scorefile_cur)
            scores, labels = scoresMat_to_arrayAndLabels(scores_mat)
            if scores_all is None:
                scores_all = scores
            else:
                scores_all = np.concatenate((scores_all, scores), axis=0)
            scores_per_snr[snr] = scores
        scores_all_datasets[dataset] = scores_all
        if print_per_snr:
            scores_per_snr_datasets[dataset] = scores_per_snr

    for iscore, label in enumerate(labels):
        if label in scores_to_print:
            if print_per_snr:
                # print mean score per SNR:
                for snr in snrs:
                    for dataset in datasets:
                        row = row + ('%.2f & ' % np.mean(scores_per_snr_datasets[dataset][snr][:, iscore]))
                    row = row[:-2]
                    row = row + "& "
            # print mean score over SNRs:
            for dataset in datasets:
                row = row + ('%.2f & ' % np.mean(scores_all_datasets[dataset][:, iscore]))
            row = row[:-2]
            row = row + " & "    

    row = row[:-3]
    row = row + ' \\\\'

    return row

exp={
     'hashes_data': ['f08b123f0c7e6c53de219053285f5bc0',
                     '12569d7f7743eead0af2efb1626a2661'],
     'hashes_snmf': ['2f3e430c0449e095d297dcb7f7f097db',
                     'f4aa2524d346e2b84a3cd925df0e75f8'],
     'hashes_lstm': ['46666e232751074bd609167dc440df8c',
                     'b6da76df68cf530d091aa499d61143de',
                     '6a4fc9017283c9f89380f765a60087ce',
                     '4561bd13e267026c3f3d1c936b15f709'],
     'hashes_unfolded_snmf': ['a45e86a1cc146e1e9d7a7f8100d9d2d7',
                              'a23657edf96a44331501d773db837a1c',
                              'ea1e7d485421e527486476ef696da2da',
                              '364ccd17a3e187bcccd30cfaa6bd9422'],
     'datasets': ['valid',
                  'test'
                 ],
     'snrs': ['m6dB', 'm3dB', '0dB', '3dB', '6dB', '9dB'],
     'print_per_snr': False,
     'scores_to_print': ['SDR'],     
     'properties': ['model', 
                    'K_layers', 
                    'hidden_dim', 
                    #'num_train', 
                    'num_params',
                    'val_loss']
}

"""
labels = exp['properties'] + exp['snrs'] + ['Mean']
labels_formatted = ''
for label in labels:
    labels_formatted = labels_formatted + label + ' & '
labels_formatted = labels_formatted[:-2] + '\\\\'
print labels_formatted
"""


# print scores for SNMF
labels_snmf = ['& SNMF, MU & $\leq 200$ & 200 & 50k &',
               '& SNMF, MU & $\leq 200$ & 2000 & 500k &']
for ihash, hash_snmf in enumerate(exp['hashes_snmf']):
    for idata, hash_data in enumerate(exp['hashes_data']):
        
        # set the directory of the current scores
        scoredir = 'data_setup_%s/scores/' % (hash_data)

        if idata==0:
            properties = ['model', 'val_loss']
        else:
            properties = ['val_loss']

        # generate the (partial) row in the table:
        row = print_row('snmf', hash_snmf, hash_data, exp['datasets'], exp['snrs'], exp['scores_to_print'], properties=properties, print_per_snr=exp['print_per_snr'], model_label=labels_snmf[ihash], scoredir=scoredir)
        
        # link the new partial row to the overall row
        if idata < (len(exp['hashes_data'])-1):
            # clip off the new line at the end of the row string:
            row = row[:-3]
            row = row + ' & '
        
        print row

# print scores for LSTM
for hash_lstm in exp['hashes_lstm']:
    for idata, hash_data in enumerate(exp['hashes_data']):

        # set the directory of the current scores
        scoredir = 'data_setup_%s/scores/' % (hash_data)

        # only print properties for the first data condition:
        if idata==0:
            properties = exp['properties']
        else:
            properties = ['val_loss']

        # generate the (partial) row in the table:
        row = print_row('lstm', hash_lstm, hash_data, exp['datasets'], exp['snrs'], exp['scores_to_print'], properties, print_per_snr=exp['print_per_snr'], model_label='& LSTM', scoredir=scoredir) 
        
        # link the new partial row to the overall row
        if idata < (len(exp['hashes_data'])-1):
            # clip off the new line at the end of the row string:
            row = row[:-3]
            row = row + ' & '

        print row

# print scores for DR-NMF
for hash_lstm in exp['hashes_unfolded_snmf']:
    for idata, hash_data in enumerate(exp['hashes_data']):

        # set the directory of the current scores
        scoredir = 'data_setup_%s/scores/' % (hash_data)

        # only print properties for the first data condition:
        if idata==0:
            properties = exp['properties']
        else:
            properties = ['val_loss']

        # generate the (partial) row in the table:
        row = print_row('unfolded_snmf', hash_lstm, hash_data, exp['datasets'], exp['snrs'], exp['scores_to_print'], properties, print_per_snr=exp['print_per_snr'], model_label='& DR-NMF', scoredir=scoredir) 
        
        # link the new partial row to the overall row
        if idata < (len(exp['hashes_data'])-1):
            # clip off the new line at the end of the row string:
            row = row[:-3]
            row = row + ' & '

        print row
"""
# print scores for unfolded SNMF
for hash_unfolded_snmf in exp['hashes_unfolded_snmf']:
    row = print_row('unfolded_snmf', hash_unfolded_snmf, exp['hash_data'], exp['datasets'], exp['snrs'], exp['scores_to_print'], exp['properties'], print_per_snr=exp['print_per_snr'], model_label='DR-NMF', scoredir=scoredir) 
    print row
"""

"""
exps=[ \
    {'label': '100\% train',
     'hash_data': 'cc061d1dc474f44165340bb36f11b16d',
     'data_label': '100\%',
     'hashes_snmf': ['2f3e430c0449e095d297dcb7f7f097db',
                     'f4aa2524d346e2b84a3cd925df0e75f8'],
     'hashes_lstm': ['46666e232751074bd609167dc440df8c',
                     'b6da76df68cf530d091aa499d61143de',
                     '6a4fc9017283c9f89380f765a60087ce',
                     '4561bd13e267026c3f3d1c936b15f709'],
     'hashes_unfolded_snmf': ['a45e86a1cc146e1e9d7a7f8100d9d2d7',
                              'a23657edf96a44331501d773db837a1c',
                              'ea1e7d485421e527486476ef696da2da',
                              '364ccd17a3e187bcccd30cfaa6bd9422'],
     'datasets': ['valid',
                  'test'
                 ],
     'snrs': ['m6dB', 'm3dB', '0dB', '3dB', '6dB', '9dB'],
     'print_per_snr': False,
     'scores_to_print': ['SDR'],     
     'properties': ['model', 
                    'K_layers', 
                    'hidden_dim', 
                    #'num_train', 
                    'num_params',
                    'val_loss']
    },
    {'label': '10\% train',
     'hash_data': 'db3355248efc7ce949ff0bc5206f0a81',
     'data_label': '10\%',
     'hashes_snmf': ['2f3e430c0449e095d297dcb7f7f097db',
                     'f4aa2524d346e2b84a3cd925df0e75f8'],
     'hashes_lstm': ['46666e232751074bd609167dc440df8c',
                     'b6da76df68cf530d091aa499d61143de',
                     '6a4fc9017283c9f89380f765a60087ce',
                     '4561bd13e267026c3f3d1c936b15f709'],
     'hashes_unfolded_snmf': ['a45e86a1cc146e1e9d7a7f8100d9d2d7',
                              'a23657edf96a44331501d773db837a1c',
                              'ea1e7d485421e527486476ef696da2da',
                              '364ccd17a3e187bcccd30cfaa6bd9422'],
     'datasets': ['valid',
                  'test'
                 ],
     'snrs': ['m6dB', 'm3dB', '0dB', '3dB', '6dB', '9dB'],
     'print_per_snr': False,
     'scores_to_print': ['SDR'],     
     'properties': ['model', 
                    'K_layers', 
                    'hidden_dim', 
                    #'num_train', 
                    'num_params',
                    'val_loss']
    }
]

for idata, exp in enumerate(exps):

    print("")
    print(exp['label'])

    labels = exp['properties'] + exp['snrs'] + ['Mean']
    labels_formatted = ''
    for label in labels:
        labels_formatted = labels_formatted + label + ' & '
    labels_formatted = labels_formatted[:-2] + '\\\\'
    print labels_formatted

    hash_data = exp['hash_data']

    print exp['data_label']

    scoredir = 'data_setup_%s/scores/' % (exp['hash_data'])
    
    # print scores for SNMF
    for hash_snmf in exp['hashes_snmf']:
        #row = print_row('tune_snmf', hash_snmf, exp['hash_data'], exp['datasets'], exp['snrs'], exp['scores_to_print'], properties=[], print_per_snr=exp['print_per_snr'], model_label='SNMF', scoredir=scoredir)
        row = print_row('snmf', hash_snmf, exp['hash_data'], exp['datasets'], exp['snrs'], exp['scores_to_print'], properties=[], print_per_snr=exp['print_per_snr'], model_label='SNMF', scoredir=scoredir)
        print row

    # print scores for LSTM
    for hash_lstm in exp['hashes_lstm']:
        row = print_row('lstm', hash_lstm, exp['hash_data'], exp['datasets'], exp['snrs'], exp['scores_to_print'], exp['properties'], print_per_snr=exp['print_per_snr'], model_label='LSTM', scoredir=scoredir) 
        print row

    # print scores for unfolded SNMF
    for hash_unfolded_snmf in exp['hashes_unfolded_snmf']:
        row = print_row('unfolded_snmf', hash_unfolded_snmf, exp['hash_data'], exp['datasets'], exp['snrs'], exp['scores_to_print'], exp['properties'], print_per_snr=exp['print_per_snr'], model_label='DR-NMF', scoredir=scoredir) 
        print row

"""
