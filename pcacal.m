function [prec,id,removeattack]=pcacal(users,usernum,itemnum,bandwagonattack,attacksize)

users=[users(:,1:3);bandwagonattack];
userpre=usernum;
[tnum,r]=size(users);
usernum=usernum+attacksize;

useritemcnt=zeros(usernum,1);
useravg=zeros(usernum,1);

for i=1:usernum
    tmp=find(users(:,1)==i);
    [r,c]=size(tmp);
    useritemcnt(i,1)=r;
    useravg(i,1)=mean(users(tmp,3));
end

rating=zeros(itemnum,usernum);

for i=1:tnum
    cuser=users(i,1);
    citem=users(i,2);
    crate=users(i,3);
    rating(citem,cuser)=crate;
end

for i=1:usernum
    idx=find(rating(:,i)~=0);
    rating(idx,i)=rating(idx,i)-useravg(i,1);
end

rating=zscore(rating);
Y=cov(rating);
[V,D]=eig(Y);
DD=[];
for i=userpre+attacksize:-1:1
    DD=[DD;D(i,i)];
end

OFFER=DD/sum(DD);
cumOFFER=cumsum(DD)/sum(DD);
OUTCOME=[DD,OFFER,cumOFFER];
PCACOV=V(:,end:-1:end-100+1);
P=abs(PCACOV);
R=sum(P,2);
[a,id]=sort(R,'ascend');

totcnt=0;
attackcnt=0;
for i=1:attacksize
    if(id(i,1)>userpre)
        attackcnt=attackcnt+1;
    end
    totcnt=totcnt+1;
    remove(totcnt,1)=id(i,1);
end

prec=attackcnt/attacksize;

%profiles after dectection include normal and attack
count=0;
removeattack=[];
for i=1:usernum
    id=i;
    [r,c]=find(remove(:,1)==id);
    if r>0
        continue;
    end
    
    count=count+1;
    pos=find(users(:,1)==id);
    bk=users(pos,:);
    bk(:,1)=count;
    removeattack=[removeattack;bk];
end