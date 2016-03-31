function [prec,ret,hitratio,ret2,hitratio2] = prep_new(X)
[user_id,movie_id,rating_id,timestamp]=textread('ml-100k/u.data','%d%d%d%d');

users=[user_id,movie_id,rating_id,timestamp];
inum=1682;
unum=943;
tnum=100000;

totavg=mean(users(:,3));
totstd=std(users(:,3));

users=sortrows(users,1);
movies=sortrows(users,2);

itemnum=inum;
usernum=unum;
totnum=tnum;

itemavg=zeros(itemnum,1);
itemcnt=zeros(itemnum,1);
itemstd=zeros(itemnum,1);
useritemcnt=zeros(usernum,1);
useravg=zeros(usernum,1);

%item info
for i=1:itemnum
    tmp=find(movies(:,2)==i);
    [r,c]=size(tmp);
    if r==0
        continue;
    end
    itemavg(i,1)=mean(movies(tmp,3));
    itemstd(i,1)=std(movies(tmp,3));
    itemcnt(i,1)=r;
end

%user info
for i=1:usernum
    tmp=find(users(:,1)==i);
    [r,c]=size(tmp);
    useritemcnt(i,1)=r;
    useravg(i,1)=mean(users(tmp,3));
end

%user number
B01=[40;81;126;138;155;168;203;218;227;274;
    306;340;353;383;422;438;483;502;565;581;
    595;609;614;676;689;720;728;739;744;800];
B12=[58;99;188;250;290;325;453;601;629;659;705];
B23=[1;18;244;407;643];
B34=[642;796;804];
B45=[234];
useridx=[B01;B12;B23;B34;B45];
p=randperm(50,5)';
useridx=useridx(p,1);

[testnum,c]=size(useridx);

%item number
itemidx=470;
totuser=zeros(usernum,1);
for i=1:usernum
    totuser(i,1)=i;
end

%A: train sample, test:useridx 
A=setdiff(totuser,useridx);
trainset=[];
testset=[];

%test
cnt=0;
for i=1:testnum
    ix=useridx(i,1);
    cnt=cnt+1;
    tmp=users(find(users(:,1)==ix),:);
    tmp(:,1)=cnt;
    testset=[testset;tmp];
end

%train
[r,c]=size(A);
trainnum=r;
cnt=0;
for i=1:trainnum
    ix=A(i,1);
    cnt=cnt+1;
    tmp=users(find(users(:,1)==ix),:);
    tmp(:,1)=cnt;
    trainset=[trainset;tmp];
end

%fillersize, attacksize
attacksize=floor(usernum*0.1);
fs(1,1)=floor(itemnum*0.05);
ret=zeros(4,1);
hitratio=zeros(4,4);

%prediction
totavg=mean(trainset(:,3));
totstd=std(trainset(:,3));
[itemn,c]=size(itemidx);
inum=1682;

trainitemavg=zeros(itemnum,1);
trainitemstd=zeros(itemnum,1);
trainitemcnt=zeros(itemnum,1);
%recalculate training itemcnt
for i=1:itemnum
    tmp=find(trainset(:,2)==i);
    [r,c]=size(tmp);
    if r==0
        continue;
    end
    trainitemavg(i,1)=mean(trainset(tmp,3));
    trainitemstd(i,1)=std(trainset(tmp,3));
    trainitemcnt(i,1)=r;
end

%previous without attack target item's score
prev=zeros(itemn,testnum);

%m: the number of target item;n: the number of target user
for i=1:itemn
    target=itemidx(i,1);
    trainuser=trainset(:,1:3);
     
     trainuseravg=zeros(trainnum,1);
     for j=1:trainnum
        tmp=find(trainuser(:,1)==j);
        trainuseravg(j,1)=mean(trainuser(tmp,3));
     end
    
    %delete target in the test set
    newtestset=testset(find(testset(:,2)~=target),:);
    testuseravg=zeros(testnum,1);
    
    for j=1:testnum
        tmp=find(newtestset(:,1)==j);
        testuseravg(j,1)=mean(newtestset(tmp,3));
    end

    %new correlation matrix, n: number of target user
    userrel=zeros(trainnum,testnum);
    for j=1:testnum  
        itemx=newtestset(find(newtestset(:,1)==j),:);
        for k=1:trainnum
            itemy=trainuser(find(trainuser(:,1)==k),:);
            [t,ia,ib]=intersect(itemx(:,2),itemy(:,2));
            [r,c]=size(t);

            if(r==0)
                continue;
            end

%             if r<5
%                 userrel(k,j)=0;
%                 continue;
%             end
            
            avg1=mean(itemx(ia,3));
            avg2=mean(itemy(ib,3));
            itx=itemx(ia,3)-avg1;
            ity=itemy(ib,3)-avg2;
            sum1=sum( itx.*ity );
            sum2=sum( itx.^2 );
            sum3=sum( ity.^2 );
            
            if sum1==0
                csum=0;
            else
                csum=sum1/((sum2^0.5)*(sum3^0.5));
            end
            userrel(k,j)=csum;
        end
    end
    
    for j=1:testnum
        rel=userrel(:,j);
        [rell,id]=sort(rel,'descend');
        cu=0;
        cd=0;
        cnt=0;
        for k=1:trainnum
            cid=id(k,1);
            tmp=trainuser(find(trainuser(:,1)==cid),:);
            pos=find(tmp(:,2)==target);
            [r,c]=size(pos);
            if r==0
                continue;
            end
            value=tmp(pos,3);
            if value<=0.1
                break;
            end
            cu=cu+rell(k,1)*(value-trainuseravg(cid,1));
            cd=cd+abs(rell(k,1));
            cnt=cnt+1;
            if cnt==20
                break;
            end
        end
            prev(i,j)=testuseravg(j,1)+cu/cd;
     end
end

%next step
trainnum=trainnum+attacksize;
    
fillersize=fs(1,1);
%prediction target item
pred=zeros(itemn,testnum);
%m: the number of target item;n: the number of target user

target=itemidx;
[bandwagonattack2]=bandwagon_attack2(target,attacksize,usernum-testnum,inum,trainitemcnt,fillersize,totavg,totstd);
[randomattack]=random_attack2(target,attacksize,usernum-testnum,inum,fillersize,totavg,totstd);
[averageattack]=average_attack2(target,attacksize,usernum-testnum,inum,fillersize,itemavg,itemstd);

%produce bandwagonattack6
bandwagonattack6=[];
for i=1:attacksize
    pos=find(X(i,:)>0);
    [r,c]=size(pos);
    for j=1:c
        temp=[i+usernum-testnum pos(j) X(i,pos(j))];
        bandwagonattack6=[bandwagonattack6;temp];
    end
end

trainuser=trainset(:,1:3);
[prec2,prec4,prec6,prec8,re2,re4,re6,re8] = pca_new(trainuser,bandwagonattack2,bandwagonattack6,averageattack,randomattack);
[ret2,hitratio2] = prep_new2(useridx,bandwagonattack2,bandwagonattack6,averageattack,randomattack);
prec=zeros(4,1);
prec(1,1)=prec2;
prec(2,1)=prec4;
prec(3,1)=prec6;
prec(4,1)=prec8;

%attack 1
ht1=0;
ht2=0;
ht3=0;
ht4=0;
for i=1:itemn
    trainuser=re2;
    for j=1:trainnum
        tmp=find(trainuser(:,1)==j);
        trainuseravg(j,1)=mean(trainuser(tmp,3));
    end
    
    %delete target in the test set
    newtestset=testset(find(testset(:,2)~=target),:);
    
    for j=1:testnum
        tmp=find(newtestset(:,1)==j);
        testuseravg(j,1)=mean(newtestset(tmp,3));
    end

    %new correlation matrix, n: number of target user
    userrel=zeros(trainnum,testnum);
    for j=1:testnum  
        itemx=newtestset(find(newtestset(:,1)==j),:);
        for k=1:trainnum
            itemy=trainuser(find(trainuser(:,1)==k),:);
            [t,ia,ib]=intersect(itemx(:,2),itemy(:,2));
            [r,c]=size(t);

            if(r==0)
                continue;
            end

%             if r<5
%                 userrel(k,j)=0;
%                 continue;
%             end
            
            avg1=mean(itemx(ia,3));
            avg2=mean(itemy(ib,3));
            itx=itemx(ia,3)-avg1;
            ity=itemy(ib,3)-avg2;
            sum1=sum( itx.*ity );
            sum2=sum( itx.^2 );
            sum3=sum( ity.^2 );
            
            if sum1==0
                csum=0;
            else
                csum=sum1/((sum2^0.5)*(sum3^0.5));
            end
            userrel(k,j)=csum;
        end
    end
    
    for j=1:testnum
        rel=userrel(:,j);
        [rell,id]=sort(rel,'descend');
    
        newtrainuser=[];
    
        for k=1:trainnum
            cid=id(k,1);
            tmp=trainuser(find(trainuser(:,1)==cid),:);
            newtrainuser=[newtrainuser;tmp];
        end

        tp=find(newtestset(:,1)==j);
        curritem=newtestset(tp,2);
        relist=zeros(itemnum,1);
        for b=1:itemnum
            pos=find(curritem(:,1)==b);
            [r,c]=size(pos);
            if r==0
                pos=find(newtrainuser(:,2)==b);
                [row,col]=size(pos);
                if row==0
                    relist(b,1)=0;
                else
                    row=min(row,20);
                    pos=pos(1:row,1);
                    uid=newtrainuser(pos,1);
                    temp=find(rel(uid,1)>=0.1);
                    [row,col]=size(temp);
                    if row==0
                        relist(b,1)=0;
                        continue;
                    end
                    pos=pos(1:row,1);
                    val=newtrainuser(pos,3);
                    uid=newtrainuser(pos,1);
                    
                    cu=sum(rel(uid,1).*(val-trainuseravg(uid,1)));
                    cd=sum(abs(rel(uid,1)));
                    relist(b,1)=testuseravg(j,1)+cu/cd;
                end
            else
                relist(b,1)=0;
            end
        end
            
        pred(i,j)=relist(target,1);
        
        [newlist,idx]=sort(relist,'descend');
        pos=find(idx(1:10,1)==target);
        [r,c]=size(pos);
        if r==1
            ht1=ht1+1;
        end
        
        pos=find(idx(1:20,1)==target);
        [r,c]=size(pos);
        if r==1
            ht2=ht2+1;
        end
        
        pos=find(idx(1:40,1)==target);
        [r,c]=size(pos);
        if r==1
            ht3=ht3+1;
        end
        
        pos=find(idx(1:60,1)==target);
        [r,c]=size(pos);
        if r==1
            ht4=ht4+1;
        end
    end
end

ps2=mean(mean(abs(pred-prev)));
ret(1,1)=ps2;
hitratio(1,1)=ht1;
hitratio(1,2)=ht2;
hitratio(1,3)=ht3;
hitratio(1,4)=ht4;

%attack2
ht1=0;
ht2=0; 
ht3=0;
ht4=0;
%prediction target item
pred=zeros(itemn,testnum);
%m: the number of target item;n: the number of target user
for i=1:itemn
    trainuser=re4;
    
    for j=1:trainnum
        tmp=find(trainuser(:,1)==j);
        trainuseravg(j,1)=mean(trainuser(tmp,3));
    end
    
    %delete target in the test set
    newtestset=testset(find(testset(:,2)~=target),:);
    
    for j=1:testnum
        tmp=find(newtestset(:,1)==j);
        testuseravg(j,1)=mean(newtestset(tmp,3));
    end

    %new correlation matrix, n: number of target user
    userrel=zeros(trainnum,testnum);
    for j=1:testnum  
        itemx=newtestset(find(newtestset(:,1)==j),:);
        for k=1:trainnum
            itemy=trainuser(find(trainuser(:,1)==k),:);
            [t,ia,ib]=intersect(itemx(:,2),itemy(:,2));
            [r,c]=size(t);

            if(r==0)
                continue;
            end

%             if r<5
%                 userrel(k,j)=0;
%                 continue;
%             end
            
            avg1=mean(itemx(ia,3));
            avg2=mean(itemy(ib,3));
            itx=itemx(ia,3)-avg1;
            ity=itemy(ib,3)-avg2;
            sum1=sum( itx.*ity );
            sum2=sum( itx.^2 );
            sum3=sum( ity.^2 );
            
            if sum1==0
                csum=0;
            else
                csum=sum1/((sum2^0.5)*(sum3^0.5));
            end
            userrel(k,j)=csum;
        end
    end
    
    for j=1:testnum
        rel=userrel(:,j);
        [rell,id]=sort(rel,'descend');
        newtrainuser=[];
        for k=1:trainnum
            cid=id(k,1);
            tmp=trainuser(find(trainuser(:,1)==cid),:);
            newtrainuser=[newtrainuser;tmp];
        end
        
        tp=find(newtestset(:,1)==j);
        curritem=newtestset(tp,2);
        relist=zeros(itemnum,1);
        for b=1:itemnum
            pos=find(curritem(:,1)==b);
            [r,c]=size(pos);
            if r==0
                pos=find(newtrainuser(:,2)==b);
                [row,col]=size(pos);
                if row==0
                    relist(b,1)=0;
                else
                    row=min(20,row);
                    pos=pos(1:row,1);
                    uid=newtrainuser(pos,1);
                    temp=find(rel(uid,1)>=0.1);
                    [row,col]=size(temp);
                    if row==0
                        relist(b,1)=0;
                        continue;
                    end
                    pos=pos(1:row,1);
                    val=newtrainuser(pos,3);
                    uid=newtrainuser(pos,1);
                    
                    cu=sum(rel(uid,1).*(val-trainuseravg(uid,1)));
                    cd=sum(abs(rel(uid,1)));
                    relist(b,1)=testuseravg(j,1)+cu/cd;
                end
            else
                relist(b,1)=0;
            end
        end
        
        pred(i,j)=relist(target,1);
        [newlist,idx]=sort(relist,'descend');
        pos=find(idx(1:10,1)==target);
        [r,c]=size(pos);
        if r==1
            ht1=ht1+1;
        end
        
        pos=find(idx(1:20,1)==target);
        [r,c]=size(pos);
        if r==1
            ht2=ht2+1;
        end
        
        pos=find(idx(1:40,1)==target);
        [r,c]=size(pos);
        if r==1
            ht3=ht3+1;
        end
        
        pos=find(idx(1:60,1)==target);
        [r,c]=size(pos);
        if r==1
            ht4=ht4+1;
        end
    end
end

ps4=mean(mean(abs(pred-prev)));
ret(2,1)=ps4;
hitratio(2,1)=ht1;
hitratio(2,2)=ht2;
hitratio(2,3)=ht3;
hitratio(2,4)=ht4;

ht1=0;
ht2=0; 
ht3=0;
ht4=0;
%prediction target item
pred=zeros(itemn,testnum);
%m: the number of target item;n: the number of target user
for i=1:itemn
    trainuser=re6;
    
    for j=1:trainnum
        tmp=find(trainuser(:,1)==j);
        trainuseravg(j,1)=mean(trainuser(tmp,3));
    end
    
    %delete target in the test set
    newtestset=testset(find(testset(:,2)~=target),:);
    
    for j=1:testnum
        tmp=find(newtestset(:,1)==j);
        testuseravg(j,1)=mean(newtestset(tmp,3));
    end

    %new correlation matrix, n: number of target user
    userrel=zeros(trainnum,testnum);
    for j=1:testnum  
        itemx=newtestset(find(newtestset(:,1)==j),:);
        for k=1:trainnum
            itemy=trainuser(find(trainuser(:,1)==k),:);
            [t,ia,ib]=intersect(itemx(:,2),itemy(:,2));
            [r,c]=size(t);

            if(r==0)
                continue;
            end

%             if r<5
%                 userrel(k,j)=0;
%                 continue;
%             end
            
            avg1=mean(itemx(ia,3));
            avg2=mean(itemy(ib,3));
            itx=itemx(ia,3)-avg1;
            ity=itemy(ib,3)-avg2;
            sum1=sum( itx.*ity );
            sum2=sum( itx.^2 );
            sum3=sum( ity.^2 );
            
            if sum1==0
                csum=0;
            else
                csum=sum1/((sum2^0.5)*(sum3^0.5));
            end
            userrel(k,j)=csum;
        end
    end
    
    for j=1:testnum
        rel=userrel(:,j);
        [rell,id]=sort(rel,'descend');
        newtrainuser=[];
        for k=1:trainnum
            cid=id(k,1);
            tmp=trainuser(find(trainuser(:,1)==cid),:);
            newtrainuser=[newtrainuser;tmp];
        end
        
        tp=find(newtestset(:,1)==j);
        curritem=newtestset(tp,2);
        relist=zeros(itemnum,1);
        for b=1:itemnum
            pos=find(curritem(:,1)==b);
            [r,c]=size(pos);
            if r==0
                pos=find(newtrainuser(:,2)==b);
                [row,col]=size(pos);
                if row==0
                    relist(b,1)=0;
                else
                    row=min(20,row);
                    pos=pos(1:row,1);
                    uid=newtrainuser(pos,1);
                    temp=find(rel(uid,1)>=0.1);
                    [row,col]=size(temp);
                    if row==0
                        relist(b,1)=0;
                        continue;
                    end
                    pos=pos(1:row,1);
                    val=newtrainuser(pos,3);
                    uid=newtrainuser(pos,1);
                    
                    cu=sum(rel(uid,1).*(val-trainuseravg(uid,1)));
                    cd=sum(abs(rel(uid,1)));
                    relist(b,1)=testuseravg(j,1)+cu/cd;
                end
            else
                relist(b,1)=0;
            end
        end
        
        pred(i,j)=relist(target,1);
        [newlist,idx]=sort(relist,'descend');
        pos=find(idx(1:10,1)==target);
        [r,c]=size(pos);
        if r==1
            ht1=ht1+1;
        end
        
        pos=find(idx(1:20,1)==target);
        [r,c]=size(pos);
        if r==1
            ht2=ht2+1;
        end
        
        pos=find(idx(1:40,1)==target);
        [r,c]=size(pos);
        if r==1
            ht3=ht3+1;
        end
        
        pos=find(idx(1:60,1)==target);
        [r,c]=size(pos);
        if r==1
            ht4=ht4+1;
        end
    end
end

ps6=mean(mean(abs(pred-prev)));
ret(3,1)=ps6;
hitratio(3,1)=ht1;
hitratio(3,2)=ht2;
hitratio(3,3)=ht3;
hitratio(3,4)=ht4;

ht1=0;
ht2=0; 
ht3=0;
ht4=0;
%prediction target item
pred=zeros(itemn,testnum);
%m: the number of target item;n: the number of target user
for i=1:itemn
    trainuser=re8;
    
    for j=1:trainnum
        tmp=find(trainuser(:,1)==j);
        trainuseravg(j,1)=mean(trainuser(tmp,3));
    end
    
    %delete target in the test set
    newtestset=testset(find(testset(:,2)~=target),:);
    
    for j=1:testnum
        tmp=find(newtestset(:,1)==j);
        testuseravg(j,1)=mean(newtestset(tmp,3));
    end

    %new correlation matrix, n: number of target user
    userrel=zeros(trainnum,testnum);
    for j=1:testnum  
        itemx=newtestset(find(newtestset(:,1)==j),:);
        for k=1:trainnum
            itemy=trainuser(find(trainuser(:,1)==k),:);
            [t,ia,ib]=intersect(itemx(:,2),itemy(:,2));
            [r,c]=size(t);

            if(r==0)
                continue;
            end

%             if r<5
%                 userrel(k,j)=0;
%                 continue;
%             end
            
            avg1=mean(itemx(ia,3));
            avg2=mean(itemy(ib,3));
            itx=itemx(ia,3)-avg1;
            ity=itemy(ib,3)-avg2;
            sum1=sum( itx.*ity );
            sum2=sum( itx.^2 );
            sum3=sum( ity.^2 );
            
            if sum1==0
                csum=0;
            else
                csum=sum1/((sum2^0.5)*(sum3^0.5));
            end
            userrel(k,j)=csum;
        end
    end
    
    for j=1:testnum
        rel=userrel(:,j);
        [rell,id]=sort(rel,'descend');
        newtrainuser=[];
        for k=1:trainnum
            cid=id(k,1);
            tmp=trainuser(find(trainuser(:,1)==cid),:);
            newtrainuser=[newtrainuser;tmp];
        end
        
        tp=find(newtestset(:,1)==j);
        curritem=newtestset(tp,2);
        relist=zeros(itemnum,1);
        for b=1:itemnum
            pos=find(curritem(:,1)==b);
            [r,c]=size(pos);
            if r==0
                pos=find(newtrainuser(:,2)==b);
                [row,col]=size(pos);
                if row==0
                    relist(b,1)=0;
                else
                    row=min(20,row);
                    pos=pos(1:row,1);
                    uid=newtrainuser(pos,1);
                    temp=find(rel(uid,1)>=0.1);
                    [row,col]=size(temp);
                    if row==0
                        relist(b,1)=0;
                        continue;
                    end
                    pos=pos(1:row,1);
                    val=newtrainuser(pos,3);
                    uid=newtrainuser(pos,1);
                    
                    cu=sum(rel(uid,1).*(val-trainuseravg(uid,1)));
                    cd=sum(abs(rel(uid,1)));
                    relist(b,1)=testuseravg(j,1)+cu/cd;
                end
            else
                relist(b,1)=0;
            end
        end
        
        pred(i,j)=relist(target,1);
        [newlist,idx]=sort(relist,'descend');
        pos=find(idx(1:10,1)==target);
        [r,c]=size(pos);
        if r==1
            ht1=ht1+1;
        end
        
        pos=find(idx(1:20,1)==target);
        [r,c]=size(pos);
        if r==1
            ht2=ht2+1;
        end
        
        pos=find(idx(1:40,1)==target);
        [r,c]=size(pos);
        if r==1
            ht3=ht3+1;
        end
        
        pos=find(idx(1:60,1)==target);
        [r,c]=size(pos);
        if r==1
            ht4=ht4+1;
        end
    end
end

ps8=mean(mean(abs(pred-prev)));
ret(4,1)=ps8;
hitratio(4,1)=ht1;
hitratio(4,2)=ht2;
hitratio(4,3)=ht3;
hitratio(4,4)=ht4;