function [retval,B,A]=gatest(X,itemavg,itemcnt,attacksize)

usersim=zeros(attacksize,attacksize);
cnt=0;
B=0;
usernum=943;
itemnum=1682;
for i=1:attacksize-1
    itemx=find(X(i,:)>0)';
    for j=i+1:attacksize
        itemy=find(X(j,:)>0)';
        t=intersect(itemx,itemy);
        [r,~]=size(t);

        if(r==0)
           continue;
        end
         
        avg1=mean(X(i,t));
        avg2=mean(X(j,t));
        itx=X(i,t)-avg1;
        ity=X(j,t)-avg2;
        sum1=sum( itx.*ity );
        sum2=sum( itx.^2 );
        sum3=sum( ity.^2 );
            
        if sum1==0
             csum=0;
        else
             csum=sum1/((sum2^0.5)*(sum3^0.5));
        end
        usersim(i,j)=csum;
        B=B+csum;
        cnt=cnt+1;
    end
end

B=B/cnt;

A=0;
for i=1:attacksize
    pos=find(X(i,:)>0);
    [r,c]=size(pos);
    if c==0
        retval=100;
        return;
    end
    
    K=(5-abs(X(i,pos)'-itemavg(pos,1)))/5;
    A=A+mean(itemcnt(pos,1).*K);
end

A=A/attacksize;
B=abs(B);
retval=B-A;