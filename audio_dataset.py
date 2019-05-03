import numpy as np
import scipy
import h5py
from pprint import pprint
import os
import util
from util import pad_axis_toN_with_constant
import scipy.io as sio


def get_mask_value(config):
    if config['transform_x']=='mag':
        return -1.
    elif config['transform_y']=='logmag':
        return -1.
    else:
        return 0.


def load_data(config, dataset='train', downsample=1):

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
        print "  Loading test data..."
        x_test, y_test, mask_test = D_test.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=None)

        print "  Padding data to ensure equal sequence lengths..."
        maxseq=x_test.shape[1]
        x_test = pad_axis_toN_with_constant(x_test, 1, maxseq, mask_value)
        y_test = pad_axis_toN_with_constant(y_test, 1, maxseq, mask_value)
        mask_test = pad_axis_toN_with_constant(mask_test, 1, maxseq, 0.)

        return x_test, y_test, mask_test

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
        print "  Loading training data..."
        x_train, y_train, mask_train = D_train.get_padded_data_matrix(transform_x=transform_x, transform_y=transform_y, pad_value=mask_value, maxlen=config['maxlen'])

        print "  Padding data to ensure equal sequence lengths..."
        maxseq=x_train.shape[1]
        x_train = pad_axis_toN_with_constant(x_train, 1, maxseq, mask_value)
        y_train = pad_axis_toN_with_constant(y_train, 1, maxseq, mask_value)
        mask_train = pad_axis_toN_with_constant(mask_train, 1, maxseq, 0.)

        return x_train, y_train, mask_train

    else:
        ValueError("Unsupported dataset '%s'" % dataset)


def clip_x_to_y(x,y,xfidx,yfidx):
    """
    Clips the length of x to the length of y
    """
    xlens=xfidx[:,1]-xfidx[:,0]
    ylens=yfidx[:,1]-yfidx[:,0]
    nutt=xfidx.shape[0]
    idx=0
    for iutt in range(nutt):
        xcur=x[:,xfidx[iutt,0]:xfidx[iutt,1]]
        x[:,idx:idx+ylens[iutt]]=xcur[:,0:ylens[iutt]]
        idx=idx+ylens[iutt]
    yframes=y.shape[1]
    x=x[:,0:yframes]
    return x


def add_to_table(f,data,label,filters):
    try:
        atom = tables.Atom.from_dtype(data.dtype)
        t_data = f.createCArray(f.root,label,atom,data.shape,filters=filters)
        t_data[:] = data
    except:
        f.createArray(f.root,label,data)


def reshape_and_pad_stacks(x_stack,y_stack,fidx,transform_x=(lambda x: x),transform_y=(lambda y: y),pad_value=0., maxlen=None, verbose=False):
        #convert the concatenated STFTs of shape (2(N/2+1), <total STFT frames>) into
        #into shape (<total number of wavfiles>, maxseq, 2(N/2+1)). Use a mask to
        #keep track of the padding of the arrays:
        maxseq = np.max(fidx[:,1]-fidx[:,0])
        if maxlen is None or (maxlen > maxseq):
            maxlen = maxseq
        d = transform_x(x_stack[:,0:1]).shape[0]
        if maxlen == maxseq:
            n_sequences=fidx.shape[0]
        else:
            n_sequences=0
            for i in range(fidx.shape[0]):
                t = 0
                while t < (fidx[i,1] - fidx[i,0]):
                    n_sequences = n_sequences + 1
                    t = t + maxlen

        """
        x_test = pad_value*np.ones((n_sequences, maxlen, d)).astype(x_stack.dtype)
        y_test = pad_value*np.ones((n_sequences, maxlen, d)).astype(y_stack.dtype)
        mask_test = np.zeros((n_sequences, maxlen, 1)).astype(x_stack.dtype)
        for i in range(n_sequences):
            x_test[i, :(fidx[i,1]-fidx[i,0]), :] = transform_x(x_stack[:, fidx[i,0]:fidx[i,1]]).T
            y_test[i, :(fidx[i,1]-fidx[i,0]), :] = transform_y(y_stack[:, fidx[i,0]:fidx[i,1]]).T
            mask_test[i, :(fidx[i,1]-fidx[i,0]), :] = 1.
        """

        x = pad_value*np.ones((n_sequences, maxlen, d)).astype(x_stack.dtype)
        y = pad_value*np.ones((n_sequences, maxlen, d)).astype(y_stack.dtype)
        mask = np.zeros((n_sequences, maxlen, 1)).astype(x_stack.dtype)
        t = 0
        i_wavfile = 0
        for i in range(n_sequences):
            t_end = t + maxlen
            flag_increment_i_wavfile = False
            if t_end >= fidx[i_wavfile,1]:
                t_end = fidx[i_wavfile,1]
                flag_increment_i_wavfile = True

            if verbose:
                print "Sequence %d of %d: t0=%d, t1=%d, duration=%d" % (i+1, n_sequences, t, t_end, t_end-t)

            x[i, :t_end-t, :] = transform_x(x_stack[:, t:t_end]).T
            y[i, :t_end-t, :] = transform_y(y_stack[:, t:t_end]).T
            mask[i, :t_end-t, :] = 1.

            if flag_increment_i_wavfile and (i < (n_sequences-1)):
                i_wavfile += 1
                t = fidx[i_wavfile,0]
            else:
                t += maxlen

        return x, y, mask


class AudioDataset:
    """
    Object for an audio dataset. The load function computes the short-time Fourier transform for each wav file. These STFTs are return in a form that can easily be passed into Keras. The STFT uses sqrt-Hann analysis and synthesis windows.

    Required inputs:
    taskfile_input: text file that consists of a list of desired input audio files in .wav format
    taskfile_output: text file that consists of a list of desired output audio files in .wav format. Each line of this file should correspond to a line in taskfile_input.

    Optional inputs:
    datafile: HDF5 file to save the dataset to. If None, no HDF5 file is created
    params_stft: parameters of the short-time Fourier transform (STFT)
                 Keys:
                 'N': STFT window duration in samples (default 320, which is 20ms for fs=16kHz)
                 'hop': STFT window hop in samples (default 160, which is 10ms for fs=16kHz)
                 'nch': number of channels to have in the output; if input is multichannel and nch is less than number of input channels, the first nch channels are returned (default 1)
    """

    def __init__(self, taskfile_input, taskfile_output, datafile=None, params_stft={'N':320, 'hop': 160, 'nch': 1}, downsample=1):
        self.taskfile_input = taskfile_input
        self.taskfile_output = taskfile_output
        self.datafile = datafile
        self.params_stft=params_stft
        self.params_stft['window']=np.sqrt(scipy.signal.hann(params_stft['N'],sym=False).astype(np.float32))
        self.downsample = downsample
        self.load_from_wavfiles()


    def load_from_wavfiles(self):

        taskfile_input=self.taskfile_input
        taskfile_output=self.taskfile_output
        datafile=self.datafile
        params_stft=self.params_stft

        if os.path.isfile(datafile):
            #print "Specified data file '%s' already exists. Use 'get_data_stacks' or 'get_padded_data_matrix' to retrieve the data." % datafile
            f = h5py.File(datafile,'r')
        else:
            #read the wavfiles:
            with open(taskfile_input) as f:
                x_wavfiles = f.readlines()
            x_wavfiles = [wavfile.strip() for wavfile in x_wavfiles]
            with open(taskfile_output) as f:
                y_wavfiles = f.readlines()
            y_wavfiles = [wavfile.strip() for wavfile in y_wavfiles]

            x_wavfiles = x_wavfiles[::self.downsample]
            y_wavfiles = y_wavfiles[::self.downsample]

            #Compute the STFTs; input is 'x', output is 'y'. THe outputs of
            #util.compute_STFTs are the concatenated STFTs in an array of
            #shape (2(N/2+1), <total number of STFT frames>), and the "fidx"
            #variable is an array of shape (<total number of wavfiles>, 2)
            #that contains the starting and ending indices of the STFT frames
            #for each wavfile. The output dimension of the stack is "2(N/2+1)"
            #because the complex numbers are encoded in real-composite form,
            #which stacks the real part on top of the imaginary part
            print "Computing STFTs..."
            x_stack, x_fidx = util.compute_STFTs(x_wavfiles, params_stft)
            y_stack, y_fidx = util.compute_STFTs(y_wavfiles, params_stft)

            fidx_are_the_same = np.allclose(x_fidx, y_fidx)
            inputs_length_gte_outputs_length = all(x_fidx[:,1]>=y_fidx[:,1])
            if not fidx_are_the_same:
                if inputs_length_gte_outputs_length:
                    #clip the lengths of the input STFTs to the lengths of the output STFTs
                    x_stack = clip_x_to_y(x_stack, y_stack, x_fidx, y_fidx)
                else:
                    ValueError("Not all input files have greater than or equal length to all output files!")
            # the indices within the stacks should be the same now:
            fidx = y_fidx

            #save the STFTs to the datafile, if one is specified
            if datafile is not None:
                print "Saving data to file '%s'..." % datafile

                f = h5py.File(datafile, 'w')
                f.create_dataset("x_stack", data=x_stack)
                f.create_dataset("y_stack", data=y_stack)
                f.create_dataset("fidx", data=fidx)
                f.create_dataset("x_wavfiles", data=x_wavfiles)
                f.create_dataset("y_wavfiles", data=y_wavfiles)
                grp_stft=f.create_group("stft")
                for key in params_stft:
                    grp_stft.attrs[key] = params_stft[key]

        self.data = f
        self.x_stack = f['x_stack']
        self.y_stack = f['y_stack']
        self.fidx = f['fidx']
        self.x_wavfiles = f['x_wavfiles']
        self.y_wavfiles = f['y_wavfiles']
        f.close()


    def reconstruct_x(self, idx, mask=None):
        X_stft = self.x_stack[:,self.fidx[idx,0]:self.fidx[idx,1]]
        if not(mask is None):
            if mask.shape[0] < X_stft.shape[0]:
                mask = np.tile(mask,(X_stft.shape[0]/mask.shape[0],1))
            X_stft = mask*X_stft
        if len(X_stft.shape) == 2:
            X_stft = np.expand_dims(X_stft, 2)
        X_stft = X_stft[:X_stft.shape[0]/2,:,:] + np.complex64(1j)*X_stft[X_stft.shape[0]/2:,:,:]
        xr,_=util.istft_mc(X_stft,self.params_stft['hop'],flag_noDiv=1,window=self.params_stft['window'])

        return xr


    def reconstruct_y(self, idx, mask=None):
        Y_stft = self.y_stack[:,self.fidx[idx,0]:self.fidx[idx,1]]
        if not(mask is None):
            if mask.shape[0] < Y_stft.shape[0]:
                mask = np.tile(mask,(Y_stft.shape[0]/mask.shape[0],1))
            Y_stft = mask*Y_stft
        if len(Y_stft.shape) == 2:
            Y_stft = np.expand_dims(Y_stft, 2)
        Y_stft = Y_stft[:Y_stft.shape[0]/2,:,:] + np.complex64(1j)*Y_stft[Y_stft.shape[0]/2:,:,:]
        yr,_=util.istft_mc(Y_stft,self.params_stft['hop'],flag_noDiv=1,window=self.params_stft['window'])

        return yr

    def reconstruct_audio(self, description, irm=None, mask=None, idx=None, test=False):
        n_wavfiles = len(self.x_wavfiles)
        if idx is None:
            for j in range(n_wavfiles):
                if irm is None or mask is None:
                    yest = self.reconstruct_x(j)
                else:
                    yest = self.reconstruct_x(j, mask=irm[j, :np.sum(mask[j,:]),:].T)
                y = self.reconstruct_y(j)
                wavfile_enhanced = self.y_wavfiles[j].replace('scaled', 'enhanced_%s' % description)
                if not os.path.exists(os.path.dirname(wavfile_enhanced)):
                    os.makedirs(os.path.dirname(wavfile_enhanced))
                util.wavwrite(wavfile_enhanced, 16e3, yest)
        elif isinstance(idx, list):
            for j in idx:
                if irm is None or mask is None:
                    yest = self.reconstruct_x(j)
                else:
                    yest = self.reconstruct_x(j, mask=irm[j, :np.sum(mask[j,:]),:].T)
                y = self.reconstruct_y(j)
                if test:
                    y_orig = util.wavread(self.y_wavfiles[j])[0:1,:]
                    x = util.wavread(self.x_wavfiles[j])[0:1,:]
                    if yest.shape[1] > x.shape[1]:
                        yest = yest[:, :x.shape[1]]
                    if y.shape[1] > y_orig.shape[1]:
                        y = y[:, :y_orig.shape[1]]
                    print "For file %d, NMSE between original x and yest is %e" % (j, np.mean( (x-yest)**2)/np.mean(x**2))
                    print "For file %d, NMSE between original y_orig and y is %e" % (j, np.mean( (y_orig-y)**2)/np.mean(y_orig**2))
                else:
                    wavfile_enhanced = self.y_wavfiles[j].replace('scaled', 'enhanced_%s' % description)
                    if not os.path.exists(os.path.dirname(wavfile_enhanced)):
                        os.makedirs(os.path.dirname(wavfile_enhanced))
                    util.wavwrite(wavfile_enhanced, 16e3, yest)
        else:
            if irm is None:
                yest = self.reconstruct_x(idx)
            else:
                yest = self.reconstruct_x(idx, mask=irm)

            wavfile_enhanced = self.y_wavfiles[idx].replace('scaled', 'enhanced_%s' % description)
            if not os.path.exists(os.path.dirname(wavfile_enhanced)):
                os.makedirs(os.path.dirname(wavfile_enhanced))
            util.wavwrite(wavfile_enhanced, 16e3, yest)

        return


    def get_data_stacks(self):
        """
        Returns the x and y data stacks, along with the frame indices fidx
        """
        datafile=self.datafile
        if not os.path.isfile(datafile):
            self.load_from_wavfiles()

        print "Loading data from file '%s'..." % datafile
        #data=tables.open_file(datafile,"r")
        f=h5py.File(datafile,"r")

        for key in ['N', 'hop', 'nch', 'window']:
            if not np.all(self.params_stft[key] == f['stft'].attrs[key]):
                ValueError("STFT parameter '%s' of loaded data does not match specified STFT parameter '%s'" % (key,key))

        x_stack = f['x_stack'][:]
        y_stack = f['y_stack'][:]
        fidx = f['fidx'][:]
        x_wavfiles = f['x_wavfiles'][:]
        y_wavfiles = f['y_wavfiles'][:]

        f.close()

        return x_stack, y_stack, fidx


    def get_padded_data_matrix(self, transform_x=(lambda x: x), transform_y=(lambda y: y), pad_value=0., maxlen=None):
        """
        Reshapes the x and y data stacks to shape (<total num. wavfiles>, maxseq, 2(N/2+1)) arrays, where 'maxseq' is the maximum number of STFT frames for any wavfile. This procedure wastes memory, since zero-padding is used to store the variable-length sequences.

        TODO: implement ability to pass 'maxlen', which allows chunking long sequences into multiple shorter sequences of max length 'maxlen'

        Outputs:
        x: data matrix for input data x
        y: data matrix for output data y
        mask: a binary matrix, equal to 1. where there is data and 0. where there is padding
        """
        x, y, mask = reshape_and_pad_stacks(self.x_stack, self.y_stack, self.fidx, transform_x=transform_x, transform_y=transform_y, pad_value=pad_value, maxlen=maxlen)

        return x, y, mask


    def score_audio_savefile_exists(self, description, snr=None, savefile=None, verbose=False, datadir=""):
        """
        Returns true if scores already exist
        """

        if savefile is None:
            if snr is None:
                savefile = datadir + ("scores/scores_%s.mat" %(description))
            else:
                savefile = datadir + ("scores/scores_%s_%s.mat" %(description,snr))

        return os.path.isfile(savefile)


    def score_audio(self, description, snr=None, savefile=None, verbose=False, datadir="", flag_rescore=False):
        """
        Computes scores for enhanced audio files
        """
        if snr is None:
            enhanced_wavfiles = [wavfile.replace('scaled', 'enhanced_%s' % description) for wavfile in self.y_wavfiles]
            reference_wavfiles = [wavfile for wavfile in self.y_wavfiles]
        else:
            enhanced_wavfiles = [wavfile.replace('scaled', 'enhanced_%s' % description) for wavfile in self.y_wavfiles if ('/' + snr + '/') in wavfile]
            reference_wavfiles = [wavfile for wavfile in self.y_wavfiles if ('/' + snr + '/') in wavfile]

        enhanced_taskfile = "taskfile_enhanced.txt"
        with open(enhanced_taskfile, 'w') as f:
            for wavfile in enhanced_wavfiles:
                f.write("%s\n" % wavfile)
        reference_taskfile = "taskfile_reference.txt"
        with open(reference_taskfile, 'w') as f:
            for wavfile in reference_wavfiles:
                f.write("%s\n" % wavfile)

        if savefile is None:
            if snr is None:
                savefile = datadir + ("scores/scores_%s.mat" %(description))
            else:
                savefile = datadir + ("scores/scores_%s_%s.mat" %(description,snr))

        if (not os.path.isfile(savefile)) or flag_rescore:
            cmd_matlab = "/usr/local/MATLAB/R2017a/bin/matlab -nosplash -nodesktop -nodisplay -r \"score_audio('%s', '%s', '%s', %d); quit();\"" %(enhanced_taskfile, reference_taskfile, savefile, verbose)
            #%(enhanced_taskfile, reference_taskfile, savefile, verbose)
            if not verbose:
                cmd_matlab = cmd_matlab + " > /dev/null"
            print("Running Matlab command %s" % cmd_matlab)
            os.system(cmd_matlab)

        print("Loading scores from savefile '%s'..." % (savefile))
        scores = sio.loadmat(open(savefile,'rb'))

        return scores
