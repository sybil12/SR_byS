% upscaling test, use Set5  as test set

clear;

p = pwd;
testdir = 'Set5';
pattern = '*.bmp';
upscale_factor = 3; % zoom factor
upsample_factor=3; % upscale factor for mid img
border = [0,0];

% set filters
O = zeros(1, upsample_factor-1);
G = [1 O -1]; % Gradient
L = [1 O -2 O 1]/2; % Laplacian
filters = {G, G.', L, L.'}; % 2D versions

 if exist('ProjM_A.mat','file')
     disp('Load ProjM for GR, ANR and Aplus...');
     load('ProjM_A.mat')
 else
     disp('compute ProjM for GR, ANR and Aplus...');
     [ProjM_GR, ProjM_ANR, ProjM_Aplus, dl, dh, V_pca]=trianP();
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
    fprintf('--------------------Scale-Up %d img---------------\n', ii);
    f = filenames{ii};
    [img, imgCB, imgCR] = load_img(f);

    %%  cut hr_img and get lr_img interpolated_img
    sz = size(img);
    sz = sz - mod(sz, upscale_factor);  %mod ШЁгр
    img = img(1:sz(1), 1:sz(2));
    imgCB = imgCB(1:sz(1), 1:sz(2));
    imgCR = imgCR(1:sz(1), 1:sz(2));  

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
    
    % extract features
    mid =  imresize(limg, upsample_factor, 'bicubic');
    features = collectpatches(mid, [3, 3], filters, upsample_factor, [2,2]);
    midres = collectpatches( mid, [3, 3], { }, upscale_factor, [2,2] );
    features = V_pca'*features;
    
    %%  upscale by GR
    fprintf('Scale-Up GR...'); tic;
    patches_GR = ProjM_GR * features;
    
    patches_GR = patches_GR + midres;
    img_size = size(limg)*upscale_factor;
    img_GR = overlap_add(patches_GR, img_size, [3,3], upscale_factor, [2,2]);
    
    fprintf('   ');toc;
    
    
    %%  upscale by ANR
    fprintf('Scale-Up ANR ...'); tic;
    patches_ANR = zeros(size(ProjM_ANR{1},1),size(features,2));
    
    D = abs(dl'*features); 
    [~, idx] = max(D);
    
    for l = 1:size(features,2) 
        patches_ANR(:,l) = ProjM_ANR{idx(l)} * features(:,l);
    end
    
    patches_ANR = patches_ANR + midres;
    img_size = size(limg)*upscale_factor;
    img_ANR = overlap_add(patches_ANR, img_size, [3,3], upscale_factor, [2,2]);
    
    fprintf('   '); toc;
    
    %%  upscale by  Aplus
    fprintf('Scale-Up Aplus ...'); tic;
    patches_Aplus = zeros(size(ProjM_Aplus{1},1),size(features,2));
    
    D = abs(dl'*features); 
    [~, idx] = max(D);
    
    for l = 1:size(features,2) 
        patches_Aplus(:,l) = ProjM_Aplus{idx(l)} * features(:,l);
    end
    
    patches_Aplus = patches_Aplus +midres;
    img_size = size(limg)*upscale_factor;
    img_Aplus = overlap_add(patches_Aplus, img_size, [3,3], upscale_factor, [2,2] );
    
    toc; 
%     fprintf('   done! \n');
    
    %%  show and save result
%     figure(1);imshow(img);
%     figure(2);imshow(limg);
%     figure(3);imshow(img_GR);
%     figure(4);imshow(img_ANR);
%     figure(5);imshow(img_Aplus);
    result = cat(3, img_GR, img_ANR, img_Aplus,  img, interpolated);
    result = uint8(result * 255);
    
    if ~isempty(interpolatedCB)
        resultCB =  uint8(interpolatedCB * 255);
        resultCB = resultCB(1+border(1):end-border(1), 1+border(2):end-border(2), :, :);
        resultCR =  uint8(interpolatedCR * 255);
        resultCR = resultCR(1+border(1):end-border(1), 1+border(2):end-border(2), :, :);
    end
    
    methods = {'GR','ANR','Aplus'} ;
    for j =1:length(methods)
        if ~isempty(imgCB)
            rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
            rgbImg = ycbcr2rgb(rgbImg);
        else
            rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
        end
        RGBdir = fullfile('resultRGB',sprintf('%d-%s.bmp', ii, methods{j}) );
        imwrite(rgbImg, RGBdir);
    end
    
    %%  compute PSNR
    ps_GR = PSNR(img,img_GR);
    ps_ANR = PSNR(img,img_ANR);
    ps_Aplus = PSNR(img,img_Aplus);
    fprintf('PSNR:  GR=%5.2fdB      ANR=%5.2fdB      Aplus=%5.2fdB\n', ps_GR, ps_ANR, ps_Aplus)
    
end











