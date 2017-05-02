#!/bin/bash

# REPLACE WITH YOUR CHIME2 PATH HERE
chime2_path='/data1/swisdom/chime2/chime2-wsj0'

# write taskfiles for training data
find ${chime2_path}/isolated/si_tr_s -name '*.wav' -type f | sort -u > taskfile_chime2_train_noisy.txt
find ${chime2_path}/scaled/si_tr_s -name '*.wav' -type f | sort -u > taskfile_chime2_train_clean.txt

# write taskfiles for development (validaiton) data
find ${chime2_path}/isolated/si_dt_05 -name '*.wav' -type f | sort -u > taskfile_chime2_valid_noisy.txt
find ${chime2_path}/scaled/si_dt_05 -name '*.wav' -type f | sort -u > taskfile_chime2_valid_clean.txt

# write taskfiles for evaluation data
find ${chime2_path}/isolated/si_et_05 -name '*.wav' -type f | sort -u > taskfile_chime2_test_noisy.txt
find ${chime2_path}/scaled/si_et_05 -name '*.wav' -type f | sort -u > taskfile_chime2_test_clean.txt
