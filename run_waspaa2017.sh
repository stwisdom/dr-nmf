#!/bin/bash

# write taskfiles for CHiME2 dataset
# **********
# Make sure you change the variable 'chime2_path' in
# create_taskfiles.sh to your CHiME2 directory
# **********
MY_DIR=$(dirname $(readlink -f $0))
$MY_DIR/create_taskfiles.sh

# list of model hashess on
#------------------------------
# 10% of training data
# The hashes are, respectively:
# SNMF with N=200 (used to initialize all DR-NMF with N=200)
# SNMF with N=2000 (used to initialize all DR-NMF with N=2000)
# LSTM with K=2, N=54
# DR-NMF with K=2, N=200
# LSTM with K=5, N=70
# DR-NMF with K=5, N=200
# LSTM with K=2, N=244
# DR-NMF with K=2, N=2000
# LSTM with K=5, N=250
# DR-NMF with K=5, N=2000
hashes_exp="
snmf_2f3e430c0449e095d297dcb7f7f097db
snmf_f4aa2524d346e2b84a3cd925df0e75f8
lstm_46666e232751074bd609167dc440df8c
unfolded_snmf_a45e86a1cc146e1e9d7a7f8100d9d2d7
lstm_6a4fc9017283c9f89380f765a60087ce
unfolded_snmf_ea1e7d485421e527486476ef696da2da
lstm_b6da76df68cf530d091aa499d61143de
unfolded_snmf_a23657edf96a44331501d773db837a1c
lstm_4561bd13e267026c3f3d1c936b15f709
unfolded_snmf_364ccd17a3e187bcccd30cfaa6bd9422"

# train and score models on 10% of training data:
#------------------------------
hash_data="downsample10"
data_setup="data_setup_${hash_data}"
data_config="${data_setup}/params_data.yaml"

for hash_exp in ${hashes_exp}
do

    model_config="${data_setup}/configs/params_${hash_exp}.yaml"
    echo "Data config is ${data_config}"
    echo "Model config is ${model_config}"
    cmd="python -u enhance.py -d ${data_config} -c ${model_config}"
    echo $cmd
    eval $cmd
done

# train and score models on 100% of training data:
#------------------------------
hash_data="downsample1"
data_setup="data_setup_${hash_data}"
data_config="${data_setup}/params_data.yaml"

for hash_exp in ${hashes_exp}
do

    model_config="${data_setup}/configs/params_${hash_exp}.yaml"
    echo "Data config is ${data_config}"
    echo "Model config is ${model_config}"
    cmd="python -u enhance.py -d ${data_config} -c ${model_config}"
    echo $cmd
    eval $cmd
done

# print the scores table:
eval "python -u print_scores.py"

# generate learning curve plots:
eval "jupyter nbconvert --execute plot_learning_curves_waspaa2017.ipynb"

