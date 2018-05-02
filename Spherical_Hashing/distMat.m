function D=distMat(P1, P2, invar)
%
% distances between vectors (use Euclidian distances as default )
% each vector is one row
% if invar==1 then use AVCS instead of Euclidian distances
 if nargin < 3
    invar = 0;
 end


if invar
    P1 = double(P1);
    P2 = double(P2);
    D = abs(pdist2(P1, P2, 'cosine')) ;
    
else

    if nargin == 2
        P1 = double(P1);
        P2 = double(P2);

        X1=repmat(sum(P1.^2,2),[1 size(P2,1)]);
        X2=repmat(sum(P2.^2,2),[1 size(P1,1)]);
        R=P1*P2';
        D=real(sqrt(X1+X2'-2*R));
    else
        P1 = double(P1);

        % each vector is one row
        X1=repmat(sum(P1.^2,2),[1 size(P1,1)]);
        R=P1*P1';
        D=X1+X1'-2*R;
        D = real(sqrt(D));
    end

end
