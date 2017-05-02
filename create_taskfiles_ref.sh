dt_or_et="et"

# evaluation data
find /data1/swisdom/chime2/chime2-wsj0/scaled/si_${dt_or_et}_05 -name '*.wav' -type f | sort -u > taskfiles/taskfile_${dt_or_et}_ref.txt
find /data1/swisdom/chime2/chime2-wsj0/isolated/si_${dt_or_et}_05 -name '*.wav' -type f | sort -u > taskfiles/taskfile_${dt_or_et}_noisy.txt

# per-SNR taskfiles
for snr in '0dB' '3dB' '6dB' '9dB' 'm3dB' 'm6dB'; do

    cmd="find /data1/swisdom/chime2/chime2-wsj0/scaled/si_${dt_or_et}_05 -path '*/${snr}/*' -name '*.wav' -type f | sort -u > taskfiles/taskfile_${dt_or_et}_ref_${snr}.txt"
    eval $cmd
    
    cmd="find /data1/swisdom/chime2/chime2-wsj0/isolated/si_${dt_or_et}_05 -path '*/${snr}/*' -name '*.wav' -type f | sort -u > taskfiles/taskfile_${dt_or_et}_noisy_${snr}.txt"
    eval $cmd

done

