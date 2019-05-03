## Implementation of deep recurrent nonnegative matrix factorization (DR-NMF) for speech separation

DR-NMF is a recurrent neural network constructed from the unfolded iterations of the iterative soft-thresholding algorithm (ISTA) applied to sparse NMF inference. Sparse NMF inference is the task of inferring the nonnegative sparse coefficients H given a nonnegative dictionary W such that WH approximates a nonnegative observation matrix X. For speech separation, the observation matrix X is the raw spectrogram of noisy audio, and the dictionary W is partitioned into speech and noise components. This partitioning of the dictionary W allows computation of an enhancement mask in the STFT domain.

Read the paper here: https://arxiv.org/abs/1709.07124

[Presentation slides from WASPAA 2017 (received best student paper award)](https://www.dropbox.com/s/mib76xfwepuclwd/waspaa2017_drnmf.pdf?dl=0)

[Download pretrained sparse NMF dictionaries and weights for the trained networks here](https://www.dropbox.com/s/u8bsw5orkw2l73s/pretrained_models.zip?dl=1)

### Instructions:

Uses the [task 2 data from the 2nd CHiME Challenge](http://spandh.dcs.shef.ac.uk/chime_challenge/chime2013/chime2_task2.html), which is available from the [LDC (free for 2017 members, $50 for non-members)](https://catalog.ldc.upenn.edu/LDC2017S10).

1) Set up environment (updated 05-03-19). This code depends on some older versions of packages (see `requirements.txt`). To set up a conda environment, run this command:
```
conda create --name drnmf_orig3 cudnn=5.1 gxx_linux-64=5.4.0 python=2.7 theano=0.9.0 numpy=1.11 pygpu=0.6.9
pip install keras==2.0.4 librosa==0.5.1 joblib==0.11.0 hickle jupyter
```
2) Download required toolboxes by running `download_toolboxes.sh`.
3) Generate taskfiles by replacing the variable `chime2_path` in `create_taskfiles.sh` by your local CHiME2 path and running `create_taskfiles.sh`.
4) Use `enhance.py` to train, reconstruct, and score audio. Use the `run_waspaa2017.sh` script to replicate results from the WASPAA 2017 paper.

Uses code from the following sources, which are automatically downloaded and unzipped by `download_toolboxes.sh`:
- sparseNMF by Jonathan Le Roux from http://www.jonathanleroux.org/software/sparseNMF.zip (put Matlab files in "sparseNMF" directory)
- BSS Eval by Emmanuel Vincent from http://bass-db.gforge.inria.fr/bss_eval/bss_eval.zip (put "bss-eval" directory in "evaluation" directory)
- Matlab PESQ implementation by Y. Hu and P. Loizou from http://ecs.utdallas.edu/loizou/speech/composite.zip (put "composite" directory in "evaluation" directory)
- Matlab STOI implementation by Cees Taal from http://ceestaal.nl/stoi.zip (put "stoi" directory in "evaluation" directory)
- Matlab Voicebox toolbox by Mike Brookes from http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.zip (put "voicebox" directory in "evaluation" directory)
