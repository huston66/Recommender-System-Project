function xoverKids  = crossoverscattered(parents,options,GenomeLength,FitnessFcn,unused,thisPopulation)
%CROSSOVERSCATTERED Position independent crossover function.
%   XOVERKIDS = CROSSOVERSCATTERED(PARENTS,OPTIONS,GENOMELENGTH, ...
%   FITNESSFCN,SCORES,THISPOPULATION) creates the crossover children XOVERKIDS
%   of the given population THISPOPULATION using the available PARENTS.
%   In single or double point crossover, genomes that are near each other tend
%   to survive together, whereas genomes that are far apart tend to be
%   separated. The technique used here eliminates that effect. Each gene has an
%   equal chance of coming from either parent. This is sometimes called uniform
%   or random crossover.
%
%   Example:
%    Create an options structure using CROSSOVERSCATTERED as the crossover
%    function
%     options = gaoptimset('CrossoverFcn' ,@crossoverscattered);

%   Copyright 2003-2007 The MathWorks, Inc.


% How many children to produce?
nKids = length(parents)/2;
% Extract information about linear constraints, if any
linCon = options.LinearConstr;
constr = ~isequal(linCon.type,'unconstrained');
% Allocate space for the kids
xoverKids = zeros(nKids,GenomeLength);

% To move through the parents twice as fast as thekids are
% being produced, a separate index for the parents is needed
index = 1;
i=1;
% for each kid...
while i<=nKids
    % get parents
    r1 = parents(index);
    index = index + 1;
    r2 = parents(index);
    index = index + 1;
    % Randomly select half of the genes from each parent
    % This loop may seem like brute force, but it is twice as fast as the
    % vectorized version, because it does no allocation.
    for j = 1:GenomeLength
        if(rand > 0.5)
            xoverKids(i,j) = thisPopulation(r1,j);
        else
            xoverKids(i,j) = thisPopulation(r2,j);
        end
    end
    
    pop=xoverKids(i,:);
    ub=floor(1682*0.05);
    attacksize=94;
    
    for j=1:94
        temp=pop((j-1)*5046+1:j*5046);
        count=0;
        largezero=zeros(1,1682);
        for k=1:3:5046
            tmp=0;
            tmp = tmp + 4*temp(k);
            tmp = tmp + 2*temp(k+1);
            tmp = tmp + 1*temp(k+2);
            if (tmp > 5)
                tmp=5;
            end

            if tmp>0
                count=count+1;
                largezero(1,count)=(k+2)/3;
            end
        end
       
        if count>ub
            largezero=largezero(1,1:count);
            pos=randperm(count,ub);
            p=largezero(1,pos);
            tep=temp;
            temp=zeros(1,1682*3);
           
            for t=1:ub
                    temp(1,(p(t)-1)*3+1:p(t)*3)=tep(1,(p(t)-1)*3+1:p(t)*3);
            end
        end
        target=470;
        temp(1,(target-1)*3+1:target*3)=[1 0 1];
        pop((j-1)*5046+1:j*5046)=temp;
    end
    xoverKids(i,:)=pop;
    i=i+1;
    % Make sure that offspring are feasible w.r.t. linear constraints
    if constr
        feasible  = isTrialFeasible(xoverKids(i,:)',linCon.Aineq,linCon.bineq,linCon.Aeq, ...
            linCon.beq,linCon.lb,linCon.ub,sqrt(options.TolCon));
        if ~feasible % Kid is not feasible
            % Children are arithmetic mean of two parents (feasible w.r.t
            % linear constraints)
            alpha = rand;
            xoverKids(i,:) = alpha*thisPopulation(r1,:) + ...
                (1-alpha)*thisPopulation(r2,:);
        end
    end
end