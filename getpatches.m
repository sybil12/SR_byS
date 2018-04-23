function [features] = getpatches(img, window, filters, overlap, border)

img_size = size(img);

% generate 3D grid
index = reshape(1:prod(img_size), img_size);
grid = index(1:window(1), 1:window(2)) - 1;         % one grid
skip = window - overlap; 
offset = index(1+border(1):skip(1):img_size(1)-window(1)+1-border(1), ...
               1+border(2):skip(2):img_size(2)-window(2)+1-border(2));
offset = reshape(offset, [1 1 numel(offset)]);          
grid = repmat(grid, [1 1 numel(offset)]) + repmat(offset, [window 1]);      %3D grid

% get features

if isempty(filters)
    f = img(grid);
    features = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);    % no filter, then just vectorization
else
    feature_size = prod(window) * numel(filters); 
    features = zeros([feature_size size(grid, 3)], 'single');   % one feature column for each pacth
    for i = 1:numel(filters)
        f = conv2(img, filters{i}, 'same');
        f = f(grid);
        f = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
        features((1:size(f, 1)) + (i-1)*size(f, 1), :) = f;
    end
end






