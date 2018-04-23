
clear;

p = pwd;
testdir = 'Set5';
pattern = '*.bmp';
upscale_factor = 3; % zoom factor
upsample_factor=3; % upscale factor for mid img
border = [0,0];




%%  load projections for  PSANR and test imgs
if exist('ProjM_PA.mat','file')
     disp('Load ProjM_PA...');
     load('ProjM_PA.mat')
 else
     disp('compute ProjM for PSANR ...');
     [ProjM_PSANR, dicts]=PSANR_P();
     disp('____________________');
end
 
dirs = fullfile(testdir, pattern);
files = dir(dirs);
filenames = cell( length(files),1 );
for i =1:length(files)
    filenames{i} = fullfile( testdir, files(i).name );
end


%% upscaling i_th image
for ii = 1 : numel(filenames)
    f = filenames{ii};
    [img, imgCB, imgCR] = load_img(f);

	%% cut hr_img and get lr_img interpolated_img
    sz = size(img);
    sz = sz - mod(sz, upscale_factor);  %mod È¡Óà
    img = img(1:sz(1), 1:sz(2));
    imgCB = imgCB(1:sz(1), 1:sz(2));
    imgCR = imgCR(1:sz(1), 1:sz(2));  
    clear sz;
    
    % down scale (img cell)
    limg = imresize(img, 1/upscale_factor, 'bicubic');
    % cheap upscaling
    interpolated = imresize(limg, upscale_factor, 'bicubic');

    if ~isempty(imgCB)
        limgCB = imresize(imgCB, 1/upscale_factor, 'bicubic');
        limgCR = imresize(imgCR, 1/upscale_factor, 'bicubic');
        interpolatedCB = imresize(limgCB, upscale_factor, 'bicubic');
        interpolatedCR = imresize(limgCR, upscale_factor, 'bicubic');
    end
    
	%%  extract features
    filters = dicts.filters;
    V_pca = dicts.V_pca;
    
    mid =  imresize(limg, upsample_factor, 'bicubic');
    features = collectpatches(mid, [3, 3], filters, upsample_factor, [2,2]);
    midres = collectpatches( mid, [3, 3], { }, upscale_factor, [2,2] );
    features = V_pca'*features;
    clear mid V_pca£»
	
	%%  upcale by PSANR
	fprintf('Scale-Up PSANR ...');
	patches_PSANR = zeros(size(ProjM_PSANR{1,1},1),size(features,2));
    dl = dicts.dl;
%     num_subanchor = size(dl,1)* size(dl{1},2);

    tic;
    for l = 1:size(features,2)
        feature = features(:,l);
        % dc = single( zeros(size(cluster_center,2),1)); 
        % distance between the feature and cluster center
        dc = pdist2(single(feature'), single(dicts.cluster_center')); 
        [~,cluster_idx] = min(dc);
        
        D = single( zeros(size(dl,1), size(dl{1},2)));
        W = single( ones(size(dl,1),1));
        W(cluster_idx) = 0;
        for i = 1:size(dl,1)
            subdl = dl{i};
            D(i,:) = pdist2(single(feature'),single(subdl'));  %Distance matrix, use Euclidean
        end
        DMAX = max(max(D));

        deta = 0.5;
        D = D + repmat(deta*DMAX.*W,1,size(D,2));
        [idx1,idx2] = find(D == min(min(D)) );
        if length(idx1)>1,    idx1=idx1((1)); idx2=idx2((1));     end
        patches_PSANR(:,l) = ProjM_PSANR{idx1,idx2} * features(:,l);
        
    end
    
    patches_PSANR = patches_PSANR +midres;
    img_size = size(limg)*upscale_factor;
    img_PSANR = overlap_add(patches_PSANR, img_size, [3,3], upscale_factor, [2,2] );
    
    fprintf('   done!\n');
    toc;
    %% show and save result
    result = uint8(img_PSANR * 255);
    
    if ~isempty(interpolatedCB)
        resultCB =  uint8(interpolatedCB * 255);
        resultCB = resultCB(1+border(1):end-border(1), 1+border(2):end-border(2), :, :);
        resultCR =  uint8(interpolatedCR * 255);
        resultCR = resultCR(1+border(1):end-border(1), 1+border(2):end-border(2), :, :);
    end
    
    methods ='PSANR';
    if ~isempty(imgCB)
        rgbImg = cat(3,result,resultCB,resultCR);
        rgbImg = ycbcr2rgb(rgbImg);
    else
        rgbImg = cat(3,result,result,result);
    end
    RGBdir = fullfile('resultRGB',sprintf('%d-%s.bmp', ii, methods) );
    imwrite(rgbImg, RGBdir);
    
    %%  compute PSNR
    ps_PSANR = PSNR(img,img_PSANR);
    fprintf('PSNR: PSANR=%5.2fdB\n', ps_PSANR);

    
end
    
    
    
    

    
