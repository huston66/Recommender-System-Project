function mutationChildren = mutationuniform(parents,options,GenomeLength,FitnessFcn,state,thisScore,thisPopulation,mutationRate)
%MUTATIONUNIFORM Uniform multi-point mutation.
%   MUTATIONCHILDREN = MUTATIONUNIFORM(PARENTS,OPTIONS,GENOMELENGTH,...
%                      FITNESSFCN,STATE,THISSCORE,THISPOPULATION, ...
%                      MUTATIONRATE) Creates the mutated children using
%   uniform mutations at multiple points. Mutated genes are uniformly 
%   distributed over the range of the gene. The new value is NOT a function
%   of the parents value for the gene.
%
%   Example:
%     options = gaoptimset('MutationFcn', @mutationuniform); 
%
%   This will create an options structure specifying the mutation
%   function to be used is MUTATIONUNIFORM.  Since the MUTATIONRATE is
%   not specified, the default value of 0.01 is used.
%
%     mutationRate = 0.05;
%     options = gaoptimset('MutationFcn', {@mutationuniform, mutationRate});
%
%   This will create an options structure specifying the mutation
%   function to be used is MUTATIONUNIFORM and the MUTATIONRATE is
%   user specified to be 0.05.
%

%   Copyright 2003-2007 The MathWorks, Inc.

if nargin < 8 || isempty(mutationRate)
    mutationRate = 0.01; % default mutation rate
end

if(strcmpi(options.PopulationType,'doubleVector'))
    mutationChildren = zeros(length(parents),GenomeLength);
    for i=1:length(parents)
        child = thisPopulation(parents(i),:);
        % Each element of the genome has mutationRate chance of being mutated.
        mutationPoints = find(rand(1,length(child)) < mutationRate);
        % each gene is replaced with a value chosen randomly from the range.
        range = options.PopInitRange;
        % range can have one column or one for each gene.
        [r,c] = size(range);
        if(c ~= 1)
            range = range(:,mutationPoints);
        end   
        lower = range(1,:);
        upper = range(2,:);
        span = upper - lower;
        
        
        child(mutationPoints) = lower + rand(1,length(mutationPoints)) .* span;
        mutationChildren(i,:) = child;
    end
elseif(strcmpi(options.PopulationType,'bitString'))
    
    mutationChildren = zeros(length(parents),GenomeLength);
    
    i=1;
    while i<=length(parents)
        child = thisPopulation(parents(i),:);
        mutationPoints = find(rand(1,length(child)) < mutationRate);
        child(mutationPoints) = ~child(mutationPoints);
        mutationChildren(i,:) = child;
        
        pop=mutationChildren(i,:);
        ub=floor(1682*0.05);
        
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
        
        mutationChildren(i,:)=pop;
        i=i+1;
    end
end
