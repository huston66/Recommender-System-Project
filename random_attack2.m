function [randomattack]=random_attack2(target,attacksize,usernum,itemnum,fillersize,a,b)

randomattack=[];
maxrate=5;

for i=usernum+1:usernum+attacksize
    p=randperm(itemnum,fillersize+1);
    tmp=find(p(1,:)==target);
    [r,c]=size(tmp);
    if c==1
        p(1,tmp(1,1))=p(1,fillersize+1);
    end
    filleritem=p(1,1:fillersize)';

    for j=1:fillersize
        while(1)
            c=normrnd(a,b,[1,1]);
            if c<1||c>5
                continue;
            else
                fillerrate(j,1) = c;
                break;
            end
        end
    end
    fillerrate=floor(fillerrate);
    fillerid=ones(fillersize,1)*i;
    temp=[fillerid filleritem fillerrate];
    temp=[temp;[i target maxrate]];
    randomattack=[randomattack;temp];
end


