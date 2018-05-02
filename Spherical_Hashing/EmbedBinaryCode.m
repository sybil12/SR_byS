                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

% xData : matrix ( number of data * dimension )
load('../dicts.mat')
xData=dicts.dl';
bit =7;                      % binary code length
nTrain = size(dicts.dl,2);                 % number of training samples

% compute training set with random sampling
[nData, dim] = size(xData);
R = randperm( nData );  xTrain = xData(R(1:nTrain),:);

% compute centers and radii of hyper-spheres with training set
[centers, radii] = SphericalHashing( xTrain , bit );

% compute distances from centers
dData = distMat( xData , centers );
% compute binary codes for data points
th = repmat( radii' , size( dData , 1 ) , 1 );
bData = zeros( size(dData) );
bData( dData <= th ) = 1;
% sum(bData)  %samples in each spherical
bData = compactbit(bData); %convert to compact string
