function [imgs, imgsCB, imgsCR]  = load_imgs(path, pattern)
% load imgs in path to a cell, with shold be consistent with pattern
if nargin < 2
    pattern = '*.bmp';
end

dirs = fullfile(path, pattern);
files = dir(dirs);
imgs = cell(numel(files), 1 );
imgsCB = cell(size(path));
imgsCR = cell(size(path));

for i = 1:numel(files)
    img_path = fullfile( path, files(i).name );
    X = imread(img_path);
    if size(X, 3) == 3 % we extract our features from Y channel
        X = rgb2ycbcr(X);
        imgsCB{i} = im2single(X(:,:,2));
        imgsCR{i} = im2single(X(:,:,3));
        X = X(:, :, 1);
    end
    X = im2single(X); % to reduce memory usage
    imgs{i} = X;
end