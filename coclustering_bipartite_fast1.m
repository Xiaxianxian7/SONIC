% min_{S>=0, S'*1=1, S*1=1, F'*F=I}  ||S - A||^2 + 2*lambda*trace(F'*Ln*F)
function [y1,y2,SS,U,V,cs] = coclustering_bipartite_fast1(A, c, NITER,distX, alpha,islocal)

if nargin < 6
    islocal = 0;
end

if nargin < 5
    NITER = 30;
end

zr = 10e-11;
lambda = 0.1;

[n,m] = size(A);
onen = 1/n*ones(n,1);
onem = 1/m*ones(m,1);

A = sparse(A);
a1 = sum(A,2);
D1a = spdiags(1./sqrt(a1),0,n,n); 
a2 = sum(A,1);
D2a = spdiags(1./sqrt(a2'),0,m,m); 
A1 = D1a*A*D2a;

SS2 = A1'*A1; 
SS2 = full(SS2);

% automatically determine the cluster number
[V, ev0, ev]=eig1(SS2,m); 
aa = abs(ev); aa(aa>1-zr)=1-eps;
ad1 = aa(2:end)./aa(1:end-1);
ad1(ad1<0.15)=0; ad1 = ad1-eps*(1:m-1)'; ad1(1)=1;
ad1 = 1 - ad1;
[scores, cs] = sort(ad1,'descend');
cs = [cs, scores];

if nargin == 1
    c = cs(1);
end

V = V(:,1:c); 
U=(A1*V)./(ones(n,1)*sqrt(ev0(1:c)'));
U = sqrt(2)/2*U; V = sqrt(2)/2*V;  


a(:,1) = ev;
A = full(A); 



idxa = cell(n,1);
for i=1:n
    if islocal == 1
        idxa0 = find(A(i,:)>0);
    else
        idxa0 = 1:m;
    end
    idxa{i} = idxa0; 
end


idxam = cell(m,1);
for i=1:m
    if islocal == 1
        idxa0 = find(A(:,i)>0);
    else
        idxa0 = 1:n;
    end
    idxam{i} = idxa0; 
end

%D1 = D1a; D2 = D2a;
D1 = 1; D2 = 10;
for iter = 1:NITER
    
    U1 = D1*U;
    V1 = D2*V;
    dist = L2_distance_1(U1',V1');  % only local distances need to be computed. speed will be increased using C
   
    %S = sparse(n,m);
    %S = spalloc(n,m,10*5);
    
    S = zeros(n,m);
    for i=1:n
        idxa0 = idxa{i};
        dfi = dist(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(dxi+lambda*dfi)./(2*alpha);
        S(i,idxa0) = EProjSimplex_new(ad);
        
%         nn = length(ad);
%         %v0 = ad-mean(ad) + 1/nn;
%         v0 = ad-sum(ad)/nn + 1/nn;
%         vmin = min(v0);
%         if vmin < 0
%             lambda_m = 0;
%             while 1
%                 v1 = v0 - lambda_m;
%                 %posidx = v1>0; npos = sum(posidx);
%                 posidx = find(v1>0); npos = length(posidx);
%                 g = -npos;
%                 f = sum(v1(posidx)) - 1;
%                 if abs(f) < 10^-6
%                     break;
%                 end
%                 lambda_m = lambda_m - f/g;
%             end
%             vv = max(v1,0);
%             S(i,idxa0) = vv;
%         else
%             S(i,idxa0) = v0;
%         end
    end
    
    
    %Sm = sparse(m,n);
    
    Sm = zeros(m,n);
    for i=1:m
        idxa0 = idxam{i};
        dfi = dist(idxa0,i);
        dxi = distX(idxa0,i);
        ad = -(dxi+lambda*dfi)./(2*alpha);
        Sm(i,idxa0) = EProjSimplex_new(ad);
    end

    S = sparse(S);
    Sm = sparse(Sm);    
    SS = (S+Sm')/2;
    %SS = sparse(SS);
    d1 = sum(SS,2);
    D1 = spdiags(1./sqrt(d1),0,n,n);
    d2 = sum(SS,1);
    D2 = spdiags(1./sqrt(d2'),0,m,m);
    SS1 = D1*SS*D2;
    
    SS2 = SS1'*SS1;
    SS2 = full(SS2);
    [V, ev0, ev]=eig1(SS2,c);
    U=(SS1*V)./(ones(n,1)*sqrt(ev0'));
    U = sqrt(2)/2*U; V = sqrt(2)/2*V;
    
    
    a(:,iter+1) = ev;
    U_old = U;
    V_old = V;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c)); 
    if fn1 < c-0.0000001
        lambda = 2*lambda;
    elseif fn2 > c-0.0000001
        lambda = lambda/2;   U = U_old; V = V_old; 
    else
        break;
    end
end

SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=SS; SS0(n+1:end,1:n)=SS';  
[clusternum, y]=graphconncomp(SS0);

y1=y(1:n)'; 
y2=y(n+1:end)'; 

% if clusternum ~= c
%     sprintf('Can not find the correct cluster number: %d', c)
% end;



