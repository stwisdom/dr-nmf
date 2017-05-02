#!/bin/bash

dt_or_et="et"

#cmd_matlab="/usr/local/MATLAB/R2016a/bin/matlab -c /usr/local/MATLAB/R2016a/licenses/license_turbine_1094417_R2016a.lic"
cmd_matlab="matlab"

cmd_taskfile="create_taskfiles_enh.sh $1"
eval ${cmd_taskfile}

#First, evaluate the scores on noisy. Then, evaluate scores on enhanced.
for snr in 0dB 3dB 6dB 9dB m3dB m6dB; do
    
    echo ""
    echo "Scoring for SNR of ${snr}"
    echo "=============================="
    
    taskfile_enh=taskfiles/taskfile_${dt_or_et}_noisy_${snr}.txt
    taskfile_ref=taskfiles/taskfile_${dt_or_et}_ref_${snr}.txt
    echo "Scoring noisy files with taskfile_enh=${taskfile_enh}, taskfile_ref=${taskfile_ref}..."
    cmd="${cmd_matlab} -nodisplay -nodesktop -r \"try score_audio('${taskfile_enh}','${taskfile_ref}','noisy_${snr}.mat',1); catch E; disp(E); disp(E.stack); end; quit\""
    echo $cmd
    eval $cmd

    taskfile_enh=taskfiles/taskfile_${dt_or_et}_enh_$1_${snr}.txt
    taskfile_ref=taskfiles/taskfile_${dt_or_et}_ref_${snr}.txt
    echo "Scoring enhanced files with taskfile_enh=${taskfile_enh}, taskfile_ref=${taskfile_ref}..."
    cmd="${cmd_matlab} -nodisplay -nodesktop -r \"try score_audio('${taskfile_enh}','${taskfile_ref}','enhanced_$1_${snr}.mat',1); catch E; disp(E); disp(E.stack); end; quit\""
    echo $cmd
    eval $cmd
done

