function [retval,B,A] = fitness(pop)
[m,n]=size(pop);
attacksize=94;
usernum=943;
itemnum=1682;
X=zeros(attacksize,itemnum);

row=1;
col=1;

for i=1:3:n
    tmp=0;
    tmp = tmp + 4*pop(i);
    tmp = tmp + 2*pop(i+1);
    tmp = tmp + 1*pop(i+2);
    if (tmp > 5)
        tmp=5;
    end
    
    X(row,col)=tmp;
    col=col+1;
    if itemnum<col
        row=row+1;
        col=1;
    end
end

[user_id,movie_id,rating_id,timestamp]=textread('ml-100k/u.data','%d%d%d%d');
users=[user_id,movie_id,rating_id,timestamp];

movies=sortrows(users,2);

itemavg=zeros(itemnum,1);
itemcnt=zeros(itemnum,1);

%item info
for i=1:itemnum
    tmp=find(movies(:,2)==i);
    [r,c]=size(tmp);
    if r==0
        continue;
    end
    itemavg(i,1)=mean(movies(tmp,3));
    itemcnt(i,1)=r;
end

[retval,B,A]=gatest(X,itemavg,itemcnt,attacksize);