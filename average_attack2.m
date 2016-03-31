function [averageattack]=average_attack2(target,attacksize,usernum,itemnum,fillersize,a,b)

averageattack=[];
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
        cnt=0;
        while(1)
            if cnt==20
                fillerrate(j,1)=a(filleritem(j,1),1);
                break;
            end
            c=normrnd(a(filleritem(j,1),1),b(filleritem(j,1),1),[1,1]);
            if c<1||c>5
                cnt=cnt+1;
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
    averageattack=[averageattack;temp];
end


