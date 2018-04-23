
clear
traindir = 'trainset';

upscale_factor = 3; %zoom factor
upsample_factor=3; % upscale factor for mid img

%% load imgs in a single cell
disp('reading imgs....')
imgs = load_imgs(traindir);


%% exact lr features to a matrix
disp('exact features....')

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

% set filters
FOEkernels = load('csf_3x3.mat');
filters = FOEkernels.model.f(:,5);

% extract features
features = collectpatches(mid, [3, 3], filters, upsample_factor, [1,1]);
clear mid;


%%  PCA dimensionality reduction for features
C = double(features * features');
[V, D] = eig(C);
D = diag(D); % perform PCA on features matrix
D = cumsum(D) / sum(D);
k = find(D >= 1e-3, 1); % ignore 0.1% energy
V_pca = V(:, k:end); % choose the largest eigenvectors' projection
features_pca = V_pca' * features;
clear C D V

%% get hr patches
hrps = cell(size(limg));
for i = 1:numel(limg)
    interpolated = imresize(limg{i}, upscale_factor, 'bicubic');
    hrps{i} = himg{i} - interpolated;    % Remove low frequencies
end
clear limg himg;
hrps = collectpatches(hrps, [3, 3], {}, upscale_factor, [1,1]);


%% K-means
K = 32;
disp('K-means clustering...')


% [Idx,C] = kmeans(features_pca', K) ;
% addpath('Clustering');
% addpath('Clustering\lib');
% addpath('Clustering\tool');
% [Idx,C] = Clustering(features_pca, 'kmeans++', K , 5000);
tic;
opts = statset('Display','final','MaxIter',5000);
[Idx,C]  = kmeans(features_pca', K,'Options',opts);
Idx = Idx'; C =C';
toc;

 
num_cluster = zeros(K,1);
for i= 1:K
    num_cluster(i)=sum(Idx==i);
end

%%  train dicts using K-SVD
%  Set KSVD configuration
ksvd_conf.iternum = 20; 
ksvd_conf.memusage = 'normal'; % higher usage doesn't fit...
ksvd_conf.dictsize = 512;
ksvd_conf.Tdata = 3; % maximal sparsity: TBD
% ksvd_conf.samples = size(hrps,2);

 
disp('dict trainning')
hrps = double(hrps); % Since it is saved in single-precision.
dl = cell(K,1); dh = cell(K,1); 
tic;
for j = 1:K
    ksvd_conf.data = double(features_pca(:, Idx==j));
    ksvd_conf.samples = sum(Idx==j);
%     if ksvd_conf.samples > 512*10
%         ksvd_conf.dictsize = 512;
%     else
%         ksvd_conf.dictsize = round(ksvd_conf.samples/10);
%     end
    fprintf('trainning %d dict ... ... \n' ,j)
   
    [dl{j}, gamma] = ksvd(ksvd_conf);
    dh{j} = (hrps(:, Idx==j) * gamma')/(full(gamma * gamma'));
end
toc;


%%
dicts.dl = dl;
dicts.dh = dh;
dicts.V_pca = V_pca;
dicts.features_pca = features_pca;
dicts.cluster_index = Idx;
dicts.cluster_center = C;
dicts.hrps = hrps;
dicts.filters = filters;
dicts.upscale_factor = upscale_factor; 
dicts.upsample_factor = upsample_factor; 
save('dicts_psanr','dicts')









