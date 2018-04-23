function result_psnr = PSNR( im1,im2 )

%------------------------计算峰值信噪比程序———————————————-----
% im1 : the original image matrix
% im2 : the modified image matrix


if (size(im1,1))~=(size(im2,1))
    s = (size(im1,1)-size(im2,1))/2;
    im1 = im1(1+s:size(im1,1)-s,1+s:size(im1,2)-s,:);
end
    

 
    A = double(im1);
    B = double(im2);
%     E = A - B; % error signal
%     N = numel(E); % Assume the original signal is at peak (|F|=1)
%     result_psnr = 10*log10( N / sum(E(:).^2) );

    [m,n] = size(im1);
    D = sum( sum( (A-B).^2 ) );%||A-B||^2
    MSE = D / (m * n);
%     MSE = (MSE(:,:,1)+MSE(:,:,2)+MSE(:,:,3))/3;

if  D == 0
    error('两幅图像完全一样');
    result_psnr = 200;
else
%     result_psnr = 10*log10( (255)^2 / MSE );
    result_psnr = 10*log10( 1 / MSE );
end