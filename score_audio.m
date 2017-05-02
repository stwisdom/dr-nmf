function score_audio(enh_list,ref_list,savefile,verbose,compute_pesq,use_parfor)
    
    if ~exist('verbose','var')
        verbose=false;
    end
    if ~exist('use_parfor','var')
        use_parfor=true;
    end
    if ~exist('compute_pesq','var')
        if use_parfor
            % PESQ implementation not currently compatible with parallel workers
            compute_pesq=true;
        else
            compute_pesq=true;
        end
    end

    paths = {'evaluation/bss_eval', 'evaluation/bss_eval', 'evaluation/obj_evaluation', 'evaluation/stoi'};

    addpath('evaluation/bss_eval') %add BSS_Eval path
    addpath('evaluation/voicebox') %add voicebox path                                
    addpath('evaluation/obj_evaluation') %add Loizou objective evaluation path 
    %addpath('evaluation/matlab-pesq-loizou') %add Loizou objective evaluation path       
    addpath('evaluation/stoi')   %add STOI path 

    %nfiles_enh=nfiles(enh_list);
    %nfiles_ref=nfiles(ref_list);
    files_enh = read_lines(enh_list);
    files_ref = read_lines(ref_list);
    nfiles_enh = length(files_enh);
    nfiles_ref = length(files_ref);

    if ~(nfiles_enh==nfiles_ref)
        error(sprintf('enh_list and ref_list have different lengths of %s and %s', nfiles_enh, nfiles_ref))
    end

    nfiles=nfiles_enh;

    fprintf('Scoring %d files...\n',nfiles);

    %hash = strsplit(enh_list, '_');
    %hash = [hash{3} '_' hash{4}];
    %hash = strsplit(hash, '.txt');
    %hash = hash{1};
    %savefile=['scores/', hash, '_', snr, '_scores.mat'];
    %savefile=['scores/', savefile];
    fprintf('Using savefile of %s\n', savefile);

    if exist(savefile,'file')

        L=load(savefile);
        S=L.S;
        labels=L.labels;

    else

        enh_file = files_enh{1};
        ref_file = files_ref{1};
        ifile = 1;
        fprintf('Comparing enhanced file %s to reference file %s, %d of %d total...\n',enh_file,ref_file,ifile,nfiles);
        [S1,labels]=compute_scores(enh_file,ref_file, compute_pesq);
        nscores = length(S1);
        S=cell(nfiles,1);;
        S{1}=S1;
        if verbose
            for iscore=1:nscores
                fprintf('%s = %.2f\n',labels{iscore},S{ifile}(iscore));
            end
            fprintf('\n');
        end

        if use_parfor
            % add paths of evaluation code to the worker pool
            pool = gcp;
            addAttachedFiles(pool, paths);
            parfor ifile=2:nfiles

                if verbose
                    fprintf('Comparing enhanced file %s to reference file %s, %d of %d total...\n',files_enh{ifile},files_ref{ifile},ifile,nfiles);
                end

                S_cur=compute_scores(files_enh{ifile},files_ref{ifile}, compute_pesq);
                S{ifile} = S_cur;

                %for iscore=1:nscores
                %    S(ifile,iscore) = S_cur(iscore);
                %end
                %S(ifile,:) = S_cur;

                if verbose
                    for iscore=1:nscores
                        fprintf('%s = %.2f\n',labels{iscore},S{ifile}(iscore));
                    end
                    fprintf('\n');
                end

            end

        else

            for ifile=2:nfiles

                if verbose
                    fprintf('Comparing enhanced file %s to reference file %s, %d of %d total...\n',files_enh{ifile},files_ref{ifile},ifile,nfiles);
                end

                S_cur=compute_scores(files_enh{ifile},files_ref{ifile}, compute_pesq);
                S{ifile} = S_cur;

                %for iscore=1:nscores
                %    S(ifile,iscore) = S_cur(iscore);
                %end
                %S(ifile,:) = S_cur;

                if verbose
                    for iscore=1:nscores
                        fprintf('%s = %.2f\n',labels{iscore},S{ifile}(iscore));
                    end
                    fprintf('\n');
                end

            end

        end
        
        Smat = zeros(nfiles,nscores);
        for ifile=1:nfiles
            Smat(ifile,:) = S{ifile};
        end
        S=Smat;
        save(savefile,'S','labels');
 
    end

    for iscore=1:size(S,2)
        fprintf('Mean %s = %.2f\n',labels{iscore},mean(S(:,iscore)));
    end

end


function n=nfiles(file_list)

    fid_list = fopen(file_list,'r');

    n=0;
    tline = fgetl(fid_list);
    while ischar(tline)
        tline = fgetl(fid_list);
        n=n+1;
    end
    fclose(fid_list);
end


function lines=read_lines(file_list)

    nlines = nfiles(file_list);
    lines = cell(nlines,1);

    fid_list = fopen(file_list,'r');

    n=2;
    tline = fgetl(fid_list);
    lines{1} = tline;
    while ischar(tline)
        tline = fgetl(fid_list);
        if ischar(tline)
            lines{n} = tline;
        end
        n=n+1;
    end
    fclose(fid_list);
end


function [S,labels]=compute_scores(est,ref,compute_pesq)
    
    if ~exist('compute_pesq','var')
        compute_pesq = false;
    end

    % read audio
    [xest,fs_est,~]=wavread(est);
    [xref,fs_ref,~]=wavread(ref);

    if ~(fs_est==fs_ref)
        error(sprintf('Sampling frequency %d of estimated is not equal to sampling frequency %d of reference',fs_est,fs_ref))
    end

    if size(xest,2)>1
        xest=xest(:,1);
    end

    if size(xref,2)>1
        xref=xref(:,1);
    end

    len_est=length(xest);
    len_ref=length(xref);
    len_min=min(len_est,len_ref);
    xest=xest(1:len_min);
    xref=xref(1:len_min);

    % BSS Eval
    SDR=bss_eval_sources(xest',xref');

    % raw SNR
    SNR=10.*log10(sum(xref.^2)/sum((xref-xest).^2));

    % segmental SNR
    [loc,glo]=snrseg(xest,xref,fs_est);

    % PESQ
    if compute_pesq
        %[~,est_file,~] = fileparts(est);
        %est_tmp = [est_file, '_tmp.wav'];
        %[~,ref_file,~] = fileparts(ref);
        %ref_tmp = [ref_file, '_tmp.wav'];
        %wavwrite(xest,fs_est,est_tmp);
        %wavwrite(xref,fs_est,ref_tmp);
        %pesq_mos=pesq(ref_tmp,est_tmp);
        %delete(est_tmp);
        %delete(ref_tmp);
        pesq_mos=pesq_16kHz(xref, xest);
    else    
        pesq_mos=-1;
    end

    % STOI
    stoi_score=stoi(xref,xest,fs_est);

    S=[SDR, SNR, loc, glo, pesq_mos, stoi_score];
    if nargout>1
        labels={'SDR','SNR','SegSNR local','SegSNR global','PESQ','STOI'};
    end

end

