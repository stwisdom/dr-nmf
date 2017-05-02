#!/bin/bash

dt_or_et="et"

# evaluation data
path_base="/data1/swisdom/chime2/chime2-wsj0/enhanced_$1/si_${dt_or_et}_05"
find ${path_base} -name '*.wav' -type f | sort -u > taskfiles/taskfile_${dt_or_et}_enh_$1.txt

# per-SNR taskfiles
for snr in '0dB' '3dB' '6dB' '9dB' 'm3dB' 'm6dB'; do
    cmd="find ${path_base} -path '*/${snr}/*' -name '*.wav' -type f | sort -u > taskfiles/taskfile_${dt_or_et}_enh_$1_${snr}.txt"
    eval $cmd
done

