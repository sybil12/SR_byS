function [lopatches, hipatches] = collectSamples(traindir, numscales, dicts, scalefactor)

lopatches = [];
hipatches = [];
upscale_factor = dicts.upscale_factor;
upsample_factor = dicts.upsample_factor;
filters = dicts.filters;

imgs = load_imgs(traindir);


for scale = 1:numscales
    sfactor = scalefactor^(scale-1);
    
    himg = cell(size(imgs));
    limg =  cell(size(imgs));
    mid = cell(size(imgs));
    for i = 1:numel(imgs)
        himg{i} = imresize(imgs{i}, sfactor, 'bicubic');
        
        % cut
        sz = size(himg{i});
        sz = sz - mod(sz, upscale_factor);  %mod ШЁгр
        himg{i} = himg{i}(1:sz(1), 1:sz(2));
        
        % down scale (img cell)
        limg{i} = imresize(himg{i}, 1/upscale_factor, 'bicubic');
        
        mid{i} =  imresize(limg{i}, upsample_factor, 'bicubic');
    end

    % extract features
    features = collectpatches(mid, [3, 3], filters, upsample_factor);
    clear mid
    
    hrps = cell(size(limg));
    for i = 1:numel(limg)
        interpolated = imresize(limg{i}, upscale_factor, 'bicubic');
        hrps{i} = himg{i} - interpolated;    % Remove low frequencies
    end
    clear limg himg;
    hrps = collectpatches(hrps, [3, 3], {}, upscale_factor);
    
    lopatches = [lopatches dicts.V_pca' * features];
    hipatches = [hipatches hrps];
end


return