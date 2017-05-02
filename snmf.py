import os
import sys
import copy

import numpy as np

import scipy.io as sio

def sparse_nmf_matlab(V, params, verbose=True, useGPU=True, gpuIndex=1, save_H=True):
    """
    Uses sparse_nmf.m to learn the parameters of a well-done sparse
    NMF model for the nonnegative input data V.
    
    Automatically chunks V into appropriately-sized chunks so that
    Matlab can train SNMF on many input frames with a large number
    of SNMF basis vectors.

    Inputs:
    V         - shape (n_feats, n_frames) nonnegative data matrix
    paramfile - dictionary of sparse_nmf parameters
        Outputs:
    W         - shape (n_feats, r) nonnegative sparse NMF dictionary with unit-L2 norm columns
    H         - shape (r, n_frames) nonnegative activation matrix
    obj       - dictionary containing 'cost' (divergence+sparsity) and 'div' (divergence)
    """
 
    # make a copy of the params dictionary, since we might modify it
    params_copy = copy.deepcopy(params)

    # get the shape of the data and determine the number of chunks
    (n_feats, n_frames) = V.shape
    r = int(params['r'])
    r_for_max_frame_batch_size = 200
    max_frame_batch_size = 700000 # max number of frames that fit on 12GB GPU when r=100
    frame_batch_size = int( float(max_frame_batch_size) * (float(r_for_max_frame_batch_size)/float(r)) )
    n_chunks = int(np.ceil( float(n_frames) / float(frame_batch_size) ))
 
    if save_H:
        # initialize the full H
        H = np.zeros((r,n_frames))
    else:
        H = None

    # iterate through the chunks
    obj_snmf = {'obj_snmf_per_chunk': []}
    initial_cost = 0.
    final_cost = 0.
    initial_div = 0.
    final_div = 0.
    for i in range(n_chunks):
        print("")
        if i==10:
            temp='We are at chunk 10'
        print("sparse NMF: processing chunk %d of %d..." %  (i+1, n_chunks))
        start_idx = i * frame_batch_size
        end_idx = ( i + 1 ) * frame_batch_size
        W, H_tmp, obj_snmf_tmp = sparse_nmf_matlab_on_chunk(V[:,start_idx:end_idx], params_copy, verbose=verbose, gpuIndex=gpuIndex)

        # update the current dictionary:
        if 'w_update_ind' in params_copy.keys():
            idx_update = np.where(params_copy['w_update_ind'])[0]
            params_copy['init_w'][:, idx_update] = W[:, idx_update]
        else:
            params_copy['init_w'] = W

        # accumulate the cost function
        obj_snmf['obj_snmf_per_chunk'].append(obj_snmf_tmp) # we append instead of accum because we might run different number of iterations per chunk
        initial_cost = initial_cost + obj_snmf_tmp['cost'][0]
        initial_div = initial_div + obj_snmf_tmp['div'][0]
        final_cost = final_cost + obj_snmf_tmp['cost'][-1]
        final_div = final_div + obj_snmf_tmp['div'][-1]

        if save_H:
            # write the portion of H we just computed from the chunk
            H[:,start_idx:end_idx] = H_tmp

    print("sparse NMF: initial overall cost %e, final overall cost %e" % (initial_cost, final_cost))
    print("sparse NMF: initial overall div %e, final overall div %e" % (initial_div, final_div))

    obj_snmf['cost'] = [initial_cost, final_cost]
    obj_snmf['div'] = [initial_div, final_div]
    if n_chunks==1:
        obj_snmf = obj_snmf['obj_snmf_per_chunk'][0]

    return W, H, obj_snmf


def sparse_nmf_matlab_on_chunk(V, params, verbose=True, useGPU=True, gpuIndex=1):
    (m,n)=V.shape
    # write the V matrix to a .mat file
    sio.savemat(open("V.mat","wb"),{"V":V})
    # write the params dictionary to a .mat file
    params_save = copy.deepcopy(params)
    params_save.update({'display': float(verbose)})
    sio.savemat(open("sparse_nmf_params.mat","wb"),params_save)
    # run the Matlab script that uses hard-coded .mat files as input, and returns
    # results in sparse_nmf_output.mat

    #cmd_matlab = "/usr/local/MATLAB/R2016a/bin/matlab -c /usr/local/MATLAB/R2016a/licenses/license_turbine_1094417_R2016a.lic -nosplash -nodesktop -nodisplay -r \"addpath('sparseNMF'); sparse_nmf_exec(); quit();\""
    cmd_matlab = "/usr/local/MATLAB/R2017a/bin/matlab -nosplash -nodesktop -nodisplay -r \"addpath('sparseNMF'); useGPU=%d; gpuIndex=%d; sparse_nmf_exec(); quit();\"" % (useGPU, gpuIndex)
    if not verbose:
        cmd_matlab = cmd_matlab + " > /dev/null"
    
    print("Running matlab command: %s" % cmd_matlab)
    err=os.system(cmd_matlab)
    if not (err==0):
        OSError("Error running Matlab command '%s' using os.system: error %s. If you are running Linux, you might be able to fix this problem by setting vm.overcommit=1 on your system, which will launch the Matlab process even if this python process has a large memory footprint, which can happen when there are a large number of frames and/or basis vectors. But this is a pretty hacky fix." % (err,cmd_matlab)) 

    L=sio.loadmat(open("sparse_nmf_output.mat","rb"))
    W=np.asarray(L['W'],dtype=V.dtype)
    H=np.asarray(L['H'],dtype=V.dtype)
    obj={'cost':np.squeeze(np.asarray(L['cost'])),'div':np.squeeze(np.asarray(L['div']))}

    return W,H,obj

