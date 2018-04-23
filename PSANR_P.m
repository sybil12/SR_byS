function [ProjM_PSANR, dicts] = PSANR_P()

%%  load or train dict
if exist('dicts_psanr.mat','file')
    disp('Load trained dictionary...');
    load('dicts_psanr', 'dicts');
else     %trian hr and lr dict 
    disp('Training dictionary ...');
    dicts = traindict();
end

dl = dicts.dl;
dh = dicts.dh;
V_pca = dicts.V_pca;
lrps = dicts.features_pca;
cluster_index = dicts.cluster_index;
% cluster_center = dicts.cluster_center;
hrps = dicts.hrps;
 
 
%%  count projection matrix for PSANR
disp('Compute PSANR regressors...');


ProjM_PSANR = cell(size(dl,1), size(dl{1},2));
num_subanchor = size(dl,1)* size(dl{1},2);
num_samples = size(lrps,2);
clusterszA = 1024 ;  % num of neighboors for each subancher

%  l2 normalize LR patches to compute distance
l2 = sum(lrps.^2).^0.5 + eps;
l2 = repmat(l2,size(lrps,1),1);
lrps_l2 = lrps./l2;
clear l2;

lambda = 0.1;

% %% Partially supervised anchored neighboor regression
% tic;
% for i = 1:size(dl,1)
% %     D = single( zeros(num_samples,1));
%     subdl = dl{i};
%     W = single( ones(num_samples,1));
%     W(idx==i) = 0;
%     for j = 1:size(dl{i},2)
%         D = pdist2(single(lrps_l2'),single(subdl(:,j)'));  %Distance matrix, use Euclidean
%         DMAX = max(D);
%         Dd = zeros(num_samples,length(0:0.1:1) );
%         for deta = 0:0.1:1
%             Dd(:,ceil(deta*10+1)) = D + deta*DMAX*W;
%         end
%         [~, idx] = sort(min(Dd,[],2));
%         Lo = lrps(:, idx(1:clusterszA));
%         Hi = hrps(:, idx(1:clusterszA));
%         ProjM_PSANR{i,j} = Hi/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';
%     end
%     
% end
% toc;
% clear lrps_l2 lrps hrps;

%% learn projection matrix in subspace

tic;
for i = 1:size(dl,1)
%     D = single( zeros(num_samples,1));
    subdl = dl{i};
    sub_lrps_l2 = lrps_l2(:,cluster_index==i);
    sub_lrps = lrps(:,cluster_index==i);
    sub_hrps = hrps(:,cluster_index==i);
    if sum(cluster_index ==i) > clusterszA
        for j = 1:size(dl{i},2)
            D = pdist2(single(sub_lrps_l2'),single(subdl(:,j)'));  %Distance matrix, use Euclidean
            [~, idx] = sort(D);
            Lo = sub_lrps(:, idx(1:clusterszA));
            Hi = sub_hrps(:, idx(1:clusterszA));
            ProjM_PSANR{i,j} = Hi/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';
        end
    else
        for j = 1:size(dl{i},2)
            Lo = sub_lrps;
            Hi = sub_hrps;
            ProjM_PSANR{i,j} = Hi/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';
        end
    end

end
toc;
clear lrps_l2 lrps hrps;

%%
save('ProjM_PA', 'ProjM_PSANR', 'dicts');

end
