function [ProjM_GR, ProjM_ANR, ProjM_Aplus, dl, dh, V_pca] =trianP()
% train projection matrix for GR/ANR/Aplus

%%  load or train dict
 if exist('dicts.mat','file')
     disp('Load trained dictionary...');
        load('dicts', 'dicts');
    else     %trian hr and lr dict 
        disp('Training dictionary ...');
        dicts = traindict();
 end
 dl = dicts.dl;
 dh = dicts.dh;
 V_pca = dicts.V_pca;
 
%%  count projection matrix for GR
disp('Compute GR regressors...');
tic
lambda = 0.01;
PP = ( dl'*dl + lambda*eye(size(dl,2)) ) \ dl';
ProjM_GR = dh*PP;
% ProjM_GR = (1+lambda)*dh*PP;
toc

%%  count projection matrix for ANR
ProjM_ANR = cell(size(dl,2),1);

disp('Compute ANR regressors...');
tic
lambda = 0.01;
% clustersz=K ,  K neighbours for each atom
if  size(dl,2) < 40
    clustersz = size(dl,2);
else
    clustersz = 40;
end

D = abs(dl'*dl);       % Correlation matrix

for i = 1:size(dl,2)
    [~,idx] = sort(D(i,:), 'descend');  % idx represent the origin index in i row of D 
    if (clustersz >= size(dl,2)/2)
        ProjM_ANR{i} = ProjM_GR;
    else
        Lo = dl(:, idx(1:clustersz)); 
        ProjM_ANR{i} = dh(:,idx(1:clustersz))/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';
    end  
end
toc

%%  count projection matrix for A+
ProjM_Aplus = cell(size(dl,2),1);
clusterszA = 1024 ;  %2048
disp('Compute A+ regressors...');
tic

% get samples for multiple scales
% numscales = 12;  scalefactor = 0.98;
numscales = 1;  scalefactor = 1;   % just one scale size
[lrps, hrps] = collectSamples( 'trainset', numscales, dicts, scalefactor);
if size(lrps,2) > 500000
    lrps = lrps(:,1:500000);
    hrps = hrps(:,1:500000);
end
% num_samples = size(lrps,2);

%  l2 normalize LR patches to compute distance
l2 = sum(lrps.^2).^0.5 + eps;
l2 = repmat(l2,size(lrps,1),1);
lrps_l2 = lrps./l2;
clear l2;


lambda = 0.1;
% count projection matrix with K=clusterszA neighborhood patches
% use normlized pacthes to compute distance, but origin ones to compute P
for i = 1:size(dl,2)
    D = pdist2(single(lrps_l2'),single(dl(:,i)'));  %Distance matrix, use Euclidean  /return a cloumn vector
    [~, idx] = sort(D);
    Lo = lrps(:, idx(1:clusterszA));
    Hi = hrps(:, idx(1:clusterszA));
    ProjM_Aplus{i} = Hi/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';
end
clear lrps_l2 lrps hrps;

toc

%%
save('ProjM_A', 'ProjM_GR', 'ProjM_ANR', 'ProjM_Aplus', 'dl', 'dh', 'V_pca');

return
