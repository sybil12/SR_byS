% Image construction from overlapping patches
function [result] = overlap_add(patches, img_size, window, scale, overlap, border)

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

% generate 3D grid
index = reshape(1:prod(img_size), img_size);
grid = index(1:window(1), 1:window(2)) - 1;         % one grid
skip = window - overlap; 
offset = index(1+border(1):skip(1):img_size(1)-window(1)+1-border(1), ...
               1+border(2):skip(2):img_size(2)-window(2)+1-border(2));
offset = reshape(offset, [1 1 numel(offset)]);          
grid = repmat(grid, [1 1 numel(offset)]) + repmat(offset, [window 1]);      %3D grid

%%  Combine 
result = zeros(img_size);
weight = zeros(img_size);
 
for i = 1:size(grid, 3)
    patch = reshape(patches(:, i), size(grid, 1), size(grid, 2));
    result(grid(:, :, i)) = result(grid(:, :, i)) + patch;
    weight(grid(:, :, i)) = weight(grid(:, :, i)) + 1;
end

I = logical(weight);
result(I) = result(I) ./ weight(I);





