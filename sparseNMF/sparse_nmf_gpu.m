function [w, h, objective] = sparse_nmf(v, params, useGPU)

% Modified by Scott Wisdom (swisdom@uw.edu) to use a GPU with
% the Matlab Parallel Processing Toolbox.
%
% SPARSE_NMF Sparse NMF with beta-divergence reconstruction error, 
% L1 sparsity constraint, optimization in normalized basis vector space.
%
% [w, h, objective] = sparse_nmf(v, params)
%
% Inputs:
% v:  matrix to be factorized
% params: optional parameters
%     beta:     beta-divergence parameter (default: 1, i.e., KL-divergence)
%     cf:       cost function type (default: 'kl'; overrides beta setting)
%               'is': Itakura-Saito divergence
%               'kl': Kullback-Leibler divergence
%               'ed': Euclidean distance
%     sparsity: weight for the L1 sparsity penalty (default: 0)
%     max_iter: maximum number of iterations (default: 100)
%     conv_eps: threshold for early stopping (default: 0, 
%                                             i.e., no early stopping)
%     display:  display evolution of objective function (default: 0)
%     random_seed: set the random seed to the given value 
%                   (default: 1; if equal to 0, seed is not set)
%     init_w:   initial setting for W (default: random; 
%                                      either init_w or r have to be set)
%     r:        # basis functions (default: based on init_w's size;
%                                  either init_w or r have to be set)
%     init_h:   initial setting for H (default: random)
%     w_update_ind: set of dimensions to be updated (default: all)
%     h_update_ind: set of dimensions to be updated (default: all)
%
% Outputs:
% w: matrix of basis functions
% h: matrix of activations
% objective: objective function values throughout the iterations
%
%
%
% References: 
% J. Eggert and E. Korner, "Sparse coding and NMF," 2004
% P. D. O'Grady and B. A. Pearlmutter, "Discovering Speech Phones 
%   Using Convolutive Non-negative Matrix Factorisation
%   with a Sparseness Constraint," 2008
% J. Le Roux, J. R. Hershey, F. Weninger, "Sparse NMF –- half-baked or well 
%   done?," 2015
%
% This implementation follows the derivations in:
% J. Le Roux, J. R. Hershey, F. Weninger, 
% "Sparse NMF –- half-baked or well done?," 
% MERL Technical Report, TR2015-023, March 2015
%
% If you use this code, please cite:
% J. Le Roux, J. R. Hershey, F. Weninger, 
% "Sparse NMF –- half-baked or well done?," 
% MERL Technical Report, TR2015-023, March 2015
%   @TechRep{LeRoux2015mar,
%     author = {{Le Roux}, J. and Hershey, J. R. and Weninger, F.},
%     title = {Sparse {NMF} -– half-baked or well done?},
%     institution = {Mitsubishi Electric Research Labs (MERL)},
%     number = {TR2015-023},
%     address = {Cambridge, MA, USA},
%     month = mar,
%     year = 2015
%   }
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2015 Mitsubishi Electric Research Labs (Jonathan Le Roux,
%                                         Felix Weninger, John R. Hershey)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = size(v, 1);
n = size(v, 2);

if ~exist('params', 'var')
    params = struct;
end

if ~exist('useGPU', 'var')
    useGPU = false;
end

if ~isfield(params, 'max_iter')
    params.max_iter = 100;
end

if ~isfield(params, 'random_seed')
    params.random_seed = 1;
end

if ~isfield(params, 'sparsity')
    params.sparsity = 0;
end

if ~isfield(params, 'conv_eps')
    params.conv_eps = 0;
end

if ~isfield(params, 'cf')
    params.cf = 'kl';
end

switch params.cf
    case 'is'
        params.beta = 0;
    case 'kl'
        params.beta = 1;
    case 'ed'
        params.beta = 2;
    otherwise
        if ~isfield(params, 'beta')
            params.beta = 1;
        end
end

if params.random_seed > 0
    rand('seed', params.random_seed);
end

if ~isfield(params, 'init_w')
    if ~isfield(params, 'r')
        error('Number of components or initialization must be given')
    end
    r = params.r;
    w = rand(m, r);
else
    ri = size(params.init_w, 2);
    w(:, 1:ri) = params.init_w;
    if isfield(params, 'r') && ri < params.r
        w(:, (ri + 1) : params.r) = rand(m, params.r - ri);
        r = params.r;
    else
        r = ri;
    end
end

if ~isfield(params, 'init_h')
    h = rand(r, n);
elseif ischar(params.init_h) && strcmp(params.init_h, 'ones')
    fprintf('sup_nmf: Initalizing H with ones.\n');
    h = ones(r, n);
else
    h = params.init_h;
end

if ~isfield(params, 'w_update_ind')
    params.w_update_ind = true(r, 1);
end

if ~isfield(params, 'h_update_ind')
    params.h_update_ind = true(r, 1);
end

% sparsity per matrix entry
if length(params.sparsity) == 1
    params.sparsity = ones(r, n) * params.sparsity;
elseif size(params.sparsity, 2) == 1
    params.sparsity = repmat(params.sparsity, 1, n);
end

% Normalize the columns of W and rescale H accordingly
wn = sqrt(sum(w.^2));
w  = bsxfun(@rdivide,w,wn);
h  = bsxfun(@times,  h,wn');

if ~isfield(params, 'display') 
    params.display = 0;
end

flr = 1e-9;
lambda = max(w * h, flr);
last_cost = Inf;

objective = struct;
objective.div = zeros(1,params.max_iter);
objective.cost = zeros(1,params.max_iter);

div_beta  = params.beta;
h_ind = params.h_update_ind;
w_ind = params.w_update_ind;
update_h = sum(h_ind);
update_w = sum(w_ind);

fprintf(1,'Performing sparse NMF with beta-divergence, beta=%.1f\n',div_beta);

if useGPU
    g = gpuDevice()
    fprintf(1,'Using GPU %d...\n',g.Index);
    w = gpuArray(w);
    params.sparsity = gpuArray(params.sparsity);
    flr = gpuArray(flr);
    v = gpuArray(v);
    lambda = gpuArray(lambda);
    div_beta = gpuArray(div_beta);
    h = gpuArray(h);
    
end

if ~(div_beta==2)
    % make sure any zero elements of v are a little greater than 0
    % to prevent NaNs when computing the divergence
    v(v==0) = min(v(v>0));
end

tic
for it = 1:params.max_iter
    
    % H updates
    if update_h > 0
        if div_beta==1
            dph = bsxfun(@plus, sum(w(:, h_ind))', params.sparsity);
            dph = max(dph, flr);
            dmh = w(:, h_ind)' * (v ./ lambda);
            h(h_ind, :) = bsxfun(@rdivide, h(h_ind, :) .* dmh, dph);
        elseif div_beta==2
            dph = w(:, h_ind)' * lambda + params.sparsity;
            dph = max(dph, flr);
            dmh = w(:, h_ind)' * v;
            h(h_ind, :) = h(h_ind, :) .* dmh ./ dph;
        else
            dph = w(:, h_ind)' * lambda.^(div_beta - 1) + params.sparsity;
            dph = max(dph, flr);
            dmh = w(:, h_ind)' * (v .* lambda.^(div_beta - 2));
            h(h_ind, :) = h(h_ind, :) .* dmh ./ dph;                
        end
        lambda = max(w * h, flr);
    end

    
    % W updates
    if update_w > 0
        if div_beta==1
            dpw = bsxfun(@plus,sum(h(w_ind, :), 2)', ...
                bsxfun(@times, ...
                sum((v ./ lambda) * h(w_ind, :)' .* w(:, w_ind)), w(:, w_ind)));
            dpw = max(dpw, flr);
            dmw = v ./ lambda * h(w_ind, :)' ...
                + bsxfun(@times, ...
                sum(bsxfun(@times, sum(h(w_ind, :),2)', w(:, w_ind))), w(:, w_ind));
            w(:, w_ind) = w(:,w_ind) .* dmw ./ dpw;
        elseif div_beta==2
            dpw = lambda * h(w_ind, :)' ...
                + bsxfun(@times, sum(v * h(w_ind, :)' .* w(:, w_ind)), w(:, w_ind));
            dpw = max(dpw, flr);
            dmw = v * h(w_ind, :)' + ...
                bsxfun(@times, sum(lambda * h(w_ind, :)' .* w(:, w_ind)), w(:, w_ind));
            w(:, w_ind) = w(:,w_ind) .* dmw ./ dpw;
        else
            dpw = lambda.^(div_beta - 1) * h(w_ind, :)' ...
                + bsxfun(@times, ...
                sum((v .* lambda.^(div_beta - 2)) * h(w_ind, :)' .* w(:, w_ind)), ...
                w(:, w_ind));
            dpw = max(dpw, flr);
            dmw = (v .* lambda.^(div_beta - 2)) * h(w_ind, :)' ...
                + bsxfun(@times, ...
                sum(lambda.^(div_beta - 1) * h(w_ind, :)' .* w(:, w_ind)), w(:, w_ind));
            w(:, w_ind) = w(:,w_ind) .* dmw ./ dpw;
        end
        % Normalize the columns of W
        w = bsxfun(@rdivide,w,sqrt(sum(w.^2)));
        lambda = max(w * h, flr);
    end
    
    
    % Compute the objective function
    if div_beta==1
        div = sum(sum(v .* log(v ./ lambda) - v + lambda));
    elseif div_beta==2
        div = sum(sum((v - lambda) .^ 2));
    elseif div_beta==0
        div = sum(sum(v ./ lambda - log ( v ./ lambda) - 1)); 
    else
        div = sum(sum(v.^div_beta + (div_beta - 1)*lambda.^div_beta ...
            - div_beta * v .* lambda.^(div_beta - 1))) / (div_beta * (div_beta - 1));
    end
    cost = div + sum(sum(params.sparsity .* h));
    
    objective.div(it)  = gather(div);
    objective.cost(it) = gather(cost);
    
    if params.display ~= 0
        fprintf('iteration %d div = %.3e cost = %.3e\n', it, div, cost);
    end
    
    % Convergence check
    if it > 1 && params.conv_eps > 0
        e = abs(cost - last_cost) / last_cost;
        if (e < params.conv_eps)
            disp('Convergence reached, aborting iteration')
            objective.div = objective.div(1:it);
            objective.cost = objective.cost(1:it);
            break
        end
    end
    last_cost = cost;
end
toc

w = gather(w);
h = gather(h);

end
