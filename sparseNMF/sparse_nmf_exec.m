V=load('V.mat');
V=V.V;
params=load('sparse_nmf_params.mat');

if ~exist('gpuIndex','var')
    gpuIndex = 3
end
if ~exist('useGPU','var')
    useGPU = true;
end

if useGPU
    gpuDevice(gpuIndex);
    [W,H,objective]=sparse_nmf_gpu(V,params, useGPU);
else
    [W,H,objective]=sparse_nmf(V,params);
end


cost=objective.cost;
div=objective.div;
fprintf('initial cost=%e, initial div=%e, final cost=%e, final div=%e\n',objective.cost(1),objective.div(1),objective.cost(end),objective.div(end));
save('sparse_nmf_output.mat','W','H','cost','div');

