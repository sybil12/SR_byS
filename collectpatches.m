function [features] = collectpatches(imgs,  window, filters , scale, overlap, border )

if nargin < 6
    border = [0 0];   
end
if nargin < 5
    overlap =[0 0];    
end
if nargin < 4
    scale = 1;   
end
% Scale all grid parameters
window = window * scale;
overlap = overlap * scale;
border = border * scale;

if iscell(imgs)
    img_num = numel(imgs);
    feature_cell = cell(img_num); 
    num_of_features = 0;

    for i = 1 : img_num
        F = getpatches(imgs{i}, window, filters , overlap, border );
        feature_cell{i} = F;
        num_of_features = num_of_features + size(F, 2);
     end
    clear imgs;

    if isempty(filters)
        feature_size = prod(window);
    else
        feature_size = prod(window) * numel(filters);
    end
    features = zeros([feature_size num_of_features], 'single');
    offset = 0;
    for i = 1 : img_num
        F = feature_cell{i};
        N = size(F, 2); 
        features(:, (1:N) + offset) = F;
        offset = offset + N;
    end
    
else 
    features = getpatches(imgs, window, filters , overlap, border );
end
