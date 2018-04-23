function [dicts]=traindict(filters)
% train dict for GR/ANR/Aplus
% filters is used to get features
% return a struct contain dl, dh, V_pca(used for PCA), filters...

traindir = 'trainset';

upscale_factor = 3; % zoom factor
upsample_factor=3; % upscale factor for mid img

if nargin < 1
    % set filters
    O = zeros(1, upsample_factor-1);
    G = [1 O -1]; % Gradient
    L = [1 O -2 O 1]/2; % Laplacian
    filters = {G, G.', L, L.'}; % 2D versions
end

%% load imgs in a single cell
imgs = load_imgs(traindir);


%% exact lr features to a matrix

% cut hr_img and get mid_img lr_img
himg = cell(size(imgs));
limg =  cell(size(imgs));
mid = cell(size(imgs));
for i = 1:numel(imgs)
    sz = size(imgs{i});
    sz = sz - mod(sz, upscale_factor);  %mod ШЁгр
    himg{i} = imgs{i}(1:sz(1), 1:sz(2));
    % down scale (img cell)
    limg{i} = imresize(himg{i}, 1/upscale_factor, 'bicubic');
    
    mid{i} =  imresize(limg{i}, upsample_factor, 'bicubic');
end
clear imgs;


% extract features
features = collectpatches(mid, [3, 3], filters, upsample_factor, [1,1]);
clear mid;


%% get hr patches
hrps = cell(size(limg));
for i = 1:numel(limg)
    interpolated = imresize(limg{i}, upscale_factor, 'bicubic');
    hrps{i} = himg{i} - interpolated;    % Remove low frequencies
end
clear limg himg;
hrps = collectpatches(hrps, [3, 3], {}, upscale_factor,[1,1]);


%%  PCA dimensionality reduction for features
C = double(features * features');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix
D = cumsum(D) / sum(D);
k = find(D >= 1e-3, 1); % ignore 0.1% energy
V_pca = V(:, k:end); % choose the largest eigenvectors' projection
features_pca = V_pca' * features;
clear C D V


%%  train dicts using K-SVD
%  Set KSVD configuration
ksvd_conf.data = double(features_pca);
clear features_pca
ksvd_conf.iternum = 20; 
ksvd_conf.memusage = 'normal'; % higher usage doesn't fit...
ksvd_conf.dictsize = 1024;
ksvd_conf.Tdata = 3; % maximal sparsity: TBD
ksvd_conf.samples = size(hrps,2);

%  Training lr_dict and compute hr_dict
tic;
fprintf('Training [%d x %d] dictionary on %d vectors using K-SVD\n', ...
    size(ksvd_conf.data, 1), ksvd_conf.dictsize, size(ksvd_conf.data, 2))
[dl, gamma] = ksvd(ksvd_conf);
toc;

fprintf('Computing high-res. dictionary from low-res. dictionary\n');
hrps = double(hrps); % Since it is saved in single-precision.
dh = (hrps * gamma')/(full(gamma * gamma'));

%%
dicts.dl = dl;
dicts.dh = dh;
dicts.V_pca = V_pca;
dicts.filters = filters;
dicts.upscale_factor = upscale_factor; 
dicts.upsample_factor = upsample_factor; 
save('dicts','dicts')
return
