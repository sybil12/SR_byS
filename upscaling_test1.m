% upscaling test, use Set5  as test set
% add AIS using  Spherical Hashing to get closest regressor
% add

clear;

p = pwd;
testdir = 'Set14';
pattern = '*.bmp';
upscale_factor = 3; % zoom factor
upsample_factor=3; % upscale factor for mid img
border = [0,0];

methods = {'GR','ANR','Aplus','AIS'} ;
use_i = [1 1 1 1];  % if use_i ==1 ,the  corresponding method is used
IBP_i = 0; % if IBP_i == 1 , then use IBP for coarse upscaling
multi_upscale = 1; % if multi_upscale = 1, then transform the image into multiple versions before upsclaing 


disp(['upscale'  testdir  'using method:'  methods(use_i==1) 'IBP=' num2str(IBP_i) ])

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
 
 %%  Load Spherical Hashing conf...
  addpath('Spherical_Hashing');
  
  if exist('S_hash.mat','file')
     fprintf('Load Spherical Hashing conf...');
     load('S_hash.mat')
     disp([' with ' num2str(length(radii)) ' Spherical'])
 else
     % train Spherical Hashing ... ...
     [centers, radii] = train_SphericalHash( dl' , 7 );
     fprintf('Spherical num = %d\n',length(radii)); 
     disp('____________________');
  end

% xData=dl'; 
% [nData, dim] = size(xData);
% bit =7;                                     % binary code length
% nTrain = nData;                           % number of training samples
% R = randperm( nData );  
% xTrain = xData(R(1:nTrain),:);
% % compute centers and radii of hyper-spheres with training set
% [centers, radii] = SphericalHashing( xTrain , bit );

% compute binary codes for dl
xData=dl'; 
dData = distMat( xData , centers ,1);  % distances from centers
th = repmat( radii' , size( dData , 1 ) , 1 );
binary_r = zeros( size(dData) );
fprintf('distance over th count %.2f%%\n', sum(dData > th)/size(dData,1)*100)
binary_r( dData <= th ) = 1;
binary_r = compactbit(binary_r); %convert to compact string
clear xData xTrain R;
table = tabulate(binary_r);

 
dirs = fullfile(testdir, pattern);
files = dir(dirs);
filenames = cell( length(files),1 );
for i =1:length(files)
    filenames{i} = fullfile( testdir, files(i).name );
end

%% upscaling i_th image
ps = zeros( numel(filenames) , length(methods)+1);
ps_m = zeros( numel(filenames) , 2);  %PSNR for multiple upscaling
tts = zeros( numel(filenames) , length(methods));
for ii = 1 : numel(filenames)
    fprintf('--------------------Scale-Up img %d: %s---------------\n', ii,filenames{ii});
    f = filenames{ii};
    [img, imgCB, imgCR] = load_img(f);

    %%  cut hr_img and get lr_img interpolated_img
    sz = size(img);
    sz = sz - mod(sz, upscale_factor);  %mod 取余
    img = img(1:sz(1), 1:sz(2));

    % down scale (img cell)
    limg = imresize(img, 1/upscale_factor, 'bicubic');
    
    % cheap upscaling
    interpolated = imresize(limg, upscale_factor, 'bicubic');
    if IBP_i ==1
        for ll = 1:2
            limg_1 = imresize(interpolated, 1/upscale_factor, 'bicubic');
            error = limg - limg_1;
            interpolated = interpolated + imresize(error, upscale_factor, 'bicubic');
        end
    end
    
    % extract features
    mid =  imresize(limg, upsample_factor, 'bicubic');
    if IBP_i ==1
        for ll = 1:2
            limg_1 = imresize(mid, 1/upscale_factor, 'bicubic');
            error = limg - limg_1;
            mid = mid + imresize(error, upscale_factor, 'bicubic');
        end
    end
    features = collectpatches(mid, [3, 3], filters, upsample_factor, [2,2]);
    midres = collectpatches( mid, [3, 3], { }, upscale_factor, [2,2] );
    features = V_pca'*features;
    
    
    if multi_upscale == 1
        features1= cell(8,1);
%         midres1 = cell(8,1);
        for i =1:4
            features1{i} = collectpatches(imrotate(mid,90*(i-1)), [3, 3], filters, upsample_factor, [2,2]);
            features1{i} =  V_pca'*features1{i};
%             midres1{i} = collectpatches( imrotate(mid,90*(i-1)), [3, 3], { }, upscale_factor, [2,2] );
        end
        for i =5:8
            features1{i} = collectpatches(imrotate(mid(end:-1:1,:,:),90*(i-5)), [3, 3], filters, upsample_factor, [2,2]);
            features1{i} =  V_pca'*features1{i};
%             midres1{i} = collectpatches( imrotate(mid(end:-1:1,:,:),90*(i-1)), [3, 3], { }, upscale_factor, [2,2] );
        end
    end
    
    
    
 
    if ~isempty(imgCB)
        imgCB = imgCB(1:sz(1), 1:sz(2));
        imgCR = imgCR(1:sz(1), 1:sz(2)); 
        limgCB = imresize(imgCB, 1/upscale_factor, 'bicubic');
        limgCR = imresize(imgCR, 1/upscale_factor, 'bicubic');
        interpolatedCB = imresize(limgCB, upscale_factor, 'bicubic');
        interpolatedCR = imresize(limgCR, upscale_factor, 'bicubic');
    end
    
    %%  upscale by GR
    if use_i(1)==1
        fprintf('Scale-Up GR...'); tic;
        patches_GR = ProjM_GR * features;

        patches_GR = patches_GR + midres;
        img_size = size(limg)*upscale_factor;
        img_GR = overlap_add(patches_GR, img_size, [3,3], upscale_factor, [2,2]);

        fprintf('   ');
        tts(ii,1)=toc; disp(['用时' num2str(tts(ii,1)) 's'])
    end
    
    %%  upscale by ANR
    if use_i(2) ==1
        fprintf('Scale-Up ANR ...'); tic;
        patches_ANR = zeros(size(ProjM_ANR{1},1),size(features,2));

        D = abs(dl'*features); 
        [~, idx] = max(D);  clear D;
        
        for l = 1:size(features,2) 
            patches_ANR(:,l) = ProjM_ANR{idx(l)} * features(:,l);
        end
        
        patches_ANR = patches_ANR + midres;
        img_size = size(limg)*upscale_factor;
        img_ANR = overlap_add(patches_ANR, img_size, [3,3], upscale_factor, [2,2]);
        clear patches_ANR;
        
        fprintf('   '); 
        tts(ii,2)=toc; disp(['用时' num2str(tts(ii,2)) 's'])
        
        if multi_upscale == 1
            disp('multiple ANR')
            patches_ANR1 = cell(8:1);
            for i =1:8
                patches_ANR1{i}=zeros(size(ProjM_ANR{1},1),size(features,2));
                
                D = abs(dl'*features1{i}); 
                [~, idx] = max(D);  clear D;

                for l = 1:size(features,2) 
                    patches_ANR1{i}(:,l) = ProjM_ANR{idx(l)} * features1{i}(:,l);
                end
            end
            
            img_ANR1 = cell(8,1);
            img_size1 =[img_size(end:-1:1); img_size ];            
            for i =1:4
%                 patches_ANR1{i}  = patches_ANR1{i} + midres;
                patches_ANR1{i}  = patches_ANR1{i} + collectpatches( imrotate(mid,90*(i-1)), [3, 3], { }, upscale_factor, [2,2] );
                img_ANR1{i} = overlap_add(patches_ANR1{i} , img_size1(mod(i,2)+1,:),  [3,3], upscale_factor, [2,2]);
                img_ANR1{i} = imrotate(img_ANR1{i}, 360-90*(i-1));                
            end
            for i =5:8
                patches_ANR1{i}  = patches_ANR1{i} + collectpatches( imrotate(mid(end:-1:1,:,:),90*(i-1)), [3, 3], { }, upscale_factor, [2,2] );
                img_ANR1{i} = overlap_add(patches_ANR1{i} , img_size1(mod(i,2)+1,:), [3,3], upscale_factor, [2,2]);
                img_ANR1{i} = imrotate(img_ANR1{i}, 360-90*(i-1));
                img_ANR1{i} = img_ANR1{i}(end:-1:1,:,:);                
            end
            
            img_ANR_m = zeros(size(limg)*upscale_factor);
            for i= 1:8
                img_ANR_m = img_ANR_m + img_ANR1{i};
            end
            img_ANR_m = img_ANR_m/8;
            
            clear patches_ANR1 img_ANR1;
        end
        
        
    end
    
    %%  upscale by  Aplus
    if use_i(3)==1
        fprintf('Scale-Up Aplus ...'); tic;
        patches_Aplus = zeros(size(ProjM_Aplus{1},1),size(features,2));

        D = abs(dl'*features); 
        [~, idx] = max(D);  clear D;

        for l = 1:size(features,2) 
            patches_Aplus(:,l) = ProjM_Aplus{idx(l)} * features(:,l);
        end

        patches_Aplus = patches_Aplus +midres;
        img_size = size(limg)*upscale_factor;
        img_Aplus = overlap_add(patches_Aplus, img_size, [3,3], upscale_factor, [2,2] );
        clear patches_Aplus;
        
        fprintf('   '); 
        tts(ii,3)=toc; disp(['用时' num2str(tts(ii,3)) 's'])
        
        if multi_upscale == 1
            disp('multiple Aplus')
            patches_Aplus1 = cell(8:1);
            for i =1:8
                patches_Aplus1{i}=zeros(size(ProjM_Aplus{1},1),size(features,2));
                
                D = abs(dl'*features1{i}); 
                [~, idx] = max(D);  clear D;

                for l = 1:size(features,2) 
                    patches_Aplus1{i}(:,l) = ProjM_Aplus{idx(l)} * features1{i}(:,l);
                end
            end
            
            img_Aplus1 = cell(8,1);
            img_size1 =[img_size(end:-1:1); img_size ];
            for i =1:4
%                 patches_Aplus1{i}  = patches_Aplus1{i} + midres;
                patches_Aplus1{i}  = patches_Aplus1{i} + collectpatches( imrotate(mid,90*(i-1)), [3, 3], { }, upscale_factor, [2,2] );
                img_Aplus1{i} = overlap_add(patches_Aplus1{i} , img_size1(mod(i,2)+1,:), [3,3], upscale_factor, [2,2]);
                img_Aplus1{i} = imrotate(img_Aplus1{i}, 360-90*(i-1));
            end
            for i =5:8
                patches_Aplus1{i}  = patches_Aplus1{i} + collectpatches( imrotate(mid(end:-1:1,:,:),90*(i-1)), [3, 3], { }, upscale_factor, [2,2] );
                img_Aplus1{i} = overlap_add(patches_Aplus1{i} , img_size1(mod(i,2)+1,:), [3,3], upscale_factor, [2,2]);
                img_Aplus1{i} = imrotate(img_Aplus1{i}, 360-90*(i-1));
                img_Aplus1{i} = img_Aplus1{i}(end:-1:1,:,:);
            end
            
            img_Aplus_m = zeros(size(limg)*upscale_factor);
            for i= 1:8
                img_Aplus_m = img_Aplus_m + img_Aplus1{i};
            end
            img_Aplus_m = img_Aplus_m/8;
            
            clear patches_Aplus1 img_Aplus1;
        end
        
        
        
    end
    
    %%  upscale by AIS(use Spherical hash to find the closet regeressor)
    if use_i(4) == 1
        % features should be normlized   % mention that spdiag in ksvd is used
        norm_f = double(features) *spdiag(double(1./sqrt(sum(features.*features))));
        norm_f = single(norm_f);

        fprintf('Scale-Up AIS ...'); tic;
        patches_AIS = zeros(size(ProjM_Aplus{1},1),size(features,2));
        
        % compute binary codes for features
        dData = distMat( norm_f' , centers, 1 ); %distances from centers
        th = repmat( radii' , size( dData , 1 ) , 1 );
        binary_f = zeros( size(dData) );
        binary_f( dData <= th ) = 1;
        binary_f = compactbit(binary_f); %convert to compact string
        clear dData th;

        % set a common regressor
        table = tabulate(binary_f);
        [~, most_binary] = max( table(:,2) ); 
        most_binary = table(most_binary,1);
        common_r = find(binary_r == most_binary );
        if ~isempty(common_r)
            common_r=common_r(1);
        else
            common_r = 1;
        end
        clear table most_binary;

        none_r = 0;
        for l = 1:size(features,2) 
            idx_r = find(binary_r == binary_f(l));
            if isempty(idx_r)
                none_r=none_r+1;
                patches_AIS(:,l) = ProjM_Aplus{ common_r } * features(:,l);
                continue
            end
            s_dl = dl(:,idx_r); % regressor with the same hash code
            D = abs(s_dl'*norm_f(:,l)); 
            [~, idx] = max(D);  clear D;
            patches_AIS(:,l) = ProjM_Aplus{idx_r(idx)} * features(:,l);
        end


        patches_AIS = patches_AIS +midres;
        img_size = size(limg)*upscale_factor;
        img_AIS = overlap_add(patches_AIS, img_size, [3,3], upscale_factor, [2,2] );
        clear patches_AIS;
        
        fprintf('  none_r = %d in %d features ',none_r, size(features,2 ))
        fprintf('   '); 
        tts(ii,4)=toc; disp(['用时' num2str(tts(ii,4)) 's'])
%         fprintf('   done!\n');
    end
    
    %%  show and save result
%     figure(1);imshow(img);
%     figure(2);imshow(limg);
%     figure(3);imshow(img_GR);
%     figure(4);imshow(img_ANR);
%     figure(5);imshow(img_Aplus);
%     figure(6);imshow(img_AIS);
    if ~exist('img_GR',  'var')
        img_GR= interpolated;
    end
    if ~exist('img_ANR', 'var')
        img_ANR= interpolated;
    end
    if ~exist('img_Aplus', 'var')
        img_Aplus= interpolated;
    end
    if ~exist('img_AIS', 'var')
        img_AIS= interpolated;
    end
    
    result = cat(3, img, interpolated, img_GR, img_ANR, img_Aplus,  img_AIS);
    result = uint8(result * 255);
    
    if ~isempty(interpolatedCB)
        resultCB =  uint8(interpolatedCB * 255);
        resultCB = resultCB(1+border(1):end-border(1), 1+border(2):end-border(2), :, :);
        resultCR =  uint8(interpolatedCR * 255);
        resultCR = resultCR(1+border(1):end-border(1), 1+border(2):end-border(2), :, :);
    end
    
%     methods = {'GR','ANR','Aplus','AIS'} ;
    for j =3:2+length(methods)
        if ~isempty(imgCB)
            rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
            rgbImg = ycbcr2rgb(rgbImg);
        else
            rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
        end
        RGBdir = fullfile('resultRGB',sprintf('%d-%s-%s.bmp', ii, files(ii).name(1:end-4), methods{j-2}) );
        imwrite(rgbImg, RGBdir);
    end
    
    if multi_upscale == 1
        result1 = cat(3, img_ANR_m, img_Aplus_m);
        result1 = uint8(result1 * 255);
        for j =4:5 %ANR A+使用多版本放大
            if ~isempty(imgCB)
                rgbImg = cat(3,result1(:,:,j-3),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result1(:,:,j-3),result(:,:,j),result(:,:,j));
            end
            RGBdir = fullfile('resultRGB',sprintf('%d-%s-%s_m.bmp', ii, files(ii).name(1:end-4), methods{j-2}) );
            imwrite(rgbImg, RGBdir);
        end
    end
    
    %%  compute PSNR  only channel one
    ps_bicubic = PSNR(img,interpolated);
    ps_GR = PSNR(img,img_GR);
    ps_ANR = PSNR(img,img_ANR);
    ps_Aplus = PSNR(img,img_Aplus);
    ps_AIS = PSNR(img,img_AIS);
    fprintf('PSNR: bicubic=%5.2fdB      GR=%5.2fdB      ANR=%5.2fdB      Aplus=%5.2fdB      AIS=%5.2fdB\n',...
       ps_bicubic, ps_GR, ps_ANR, ps_Aplus, ps_AIS)
   
   if multi_upscale == 1
       ps_ANR_m = PSNR(img,img_ANR_m);
       ps_Aplus_m = PSNR(img,img_Aplus_m);
       fprintf('ps_ANR_m = %5.2fdB    ps_Aplus_m = %5.2fdB\n',ps_ANR_m,ps_Aplus_m)
       ps_m(ii,:) = [ps_ANR_m,ps_Aplus_m];
   end 
   
    ps(ii,:) = [ps_bicubic ps_GR ps_ANR ps_Aplus ps_AIS];
    clear img  interpolated img_GR  img_ANR  img_Aplus img_AIS;
    clear features features1;
    
end

disp('filenames:')
disp(filenames)
disp('PSNR for bicubic GR ANR A+ AIS')
disp(ps)
disp(ps_m)

disp('time for GR ANR A+ AIS')
disp(tts)











