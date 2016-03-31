function [bandwagonattack]=bandwagon_attack2(target,attacksize,usernum,itemnum,itemcnt,fillersize,a,b)

%find frequently set
itemselect=find(itemcnt(:,1)>=500);
fillersize=fillersize-4;
itemselect=itemselect(itemselect(:,1)~=target);
[r,c]=size(itemselect);
len=r;

bandwagonattack=[];
maxrate=5;
siz=itemnum-r-1;
itemrand=zeros(1,siz);
cnt=1;

for i=1:itemnum
    if i==target
        continue;
    end
    tmp=find(itemselect(:,1)==i);
    [r,c]=size(tmp);
    if r==1
        continue;
    else
        itemrand(1,cnt)=i;
        cnt=cnt+1;
    end
end

selectid=zeros(len,1);
selectrate=zeros(len,1);
selectrate(:,1)=maxrate;

for i=usernum+1:usernum+attacksize
    selectid(:,1)=i;
    p=randperm(siz,fillersize);
    
    filleritem=itemrand(p(1,:))';
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
    fillerrate=round(fillerrate);
    fillerid=ones(fillersize,1)*i;
    temp=[fillerid filleritem fillerrate];
    temp=[temp;[i target maxrate]];
    temp=[temp;[selectid itemselect selectrate]];
    bandwagonattack=[bandwagonattack;temp];
end


