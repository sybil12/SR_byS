 % trian Spherical Hashing
 function [centers, radii] = train_SphericalHash(xData, bit)
 %     xData  -  matrix ( number of data * dimension )
 %     bit  -  binary code length

    disp('train Spherical Hashing ... ...')
    addpath('Spherical_Hashing');
    [nData, ~] = size(xData);
    if nData>10000
        nTrain=10000;
    else
        nTrain = nData;                           % number of training samples
    end
    R = randperm( nData );  
    xTrain = xData(R(1:nTrain),:);
    % compute centers and radii of hyper-spheres with training set
    [centers, radii] = SphericalHashing( xTrain , bit );
    
    save('S_hash', 'centers', 'radii');
 end