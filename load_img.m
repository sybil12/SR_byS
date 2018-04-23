function [imgs, imgsCB, imgsCR]  = load_img(filename)
% load imgs in path to a cell, with shold be consistent with pattern

X = imread(filename);
if size(X, 3) == 3 % we extract our features from Y channel
    X = rgb2ycbcr(X);
    imgsCB = im2single(X(:,:,2));
    imgsCR = im2single(X(:,:,3));
    X = X(:, :, 1);
end
X = im2single(X); % to reduce memory usage
imgs = X;
