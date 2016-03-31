%function mygafun()
options = gaoptimset('PopulationType','bitString'); 
options = gaoptimset(options,'PlotFcns',{@gaplotbestf,@gaplotscores,@gaplotbestindiv,@gaplotstopping});
len=94*1682*3;
popsize=20;
Y=zeros(popsize,len);

target=470;
attacksize=94;
usernum=943;
itemnum=1682;
fillersize=floor(1682*0.05);

[user_id,movie_id,rating_id,timestamp]=textread('ml-100k/u.data','%d%d%d%d');
users=[user_id,movie_id,rating_id,timestamp];

totavg=mean(users(:,3));
totstd=std(users(:,3));

users=sortrows(users,1);
movies=sortrows(users,2);

itemavg=zeros(itemnum,1);
itemcnt=zeros(itemnum,1);
itemstd=zeros(itemnum,1);

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

%[retval,B,A]=gatest(X,itemavg,itemcnt,attacksize);

num0=[0 0 0];
num1=[0 0 1];
num2=[0 1 0];
num3=[0 1 1];
num4=[1 0 0];
num5=[1 0 1];

for i=1:popsize
    [bandwagonattack]=bandwagon_attack2(target,attacksize,usernum,itemnum,itemcnt,fillersize,totavg,totstd);
    [r,c]=size(bandwagonattack);
    T=zeros(attacksize,1682*3);
    perlen=r/attacksize;
    cnt=1;
    y=[];
    for j=1:attacksize
        for k=1:perlen
            it=bandwagonattack(cnt,2);
            val=bandwagonattack(cnt,3);
            if val==1
                  num=num1;
            elseif val==2
                  num=num2;
            elseif val==3
                   num=num3;
            elseif val==4;
                    num=num4;
            elseif val==5
                    num=num5;
            end
            sta=(it-1)*3+1;
            ter=it*3;
            T(j,sta:ter)=num;
            cnt=cnt+1;
        end
        y=[y T(j,:)];
    end
    
    Y(i,:)=y;
end

[retval,B,A]=fitness(Y(1,:));
options.PopulationSize=popsize;
options.InitialPopulation=Y;
%options.MigrationFraction=0.2;
FitnessFcn = @fitness;
GenomeLength = 94*1682*3; % 32 bit representation (might have to increase)
options.Generations=100;
[x,fval,exitflag,output] = ga(FitnessFcn,GenomeLength,options);

[~,n]=size(x);
X=zeros(attacksize,itemnum);

row=1;
col=1;

for i=1:3:n
    tmp=0;
    tmp = tmp + 4*x(i);
    tmp = tmp + 2*x(i+1);
    tmp = tmp + 1*x(i+2);
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

% save dat X;
% load dat;
[prec,ret,hitratio,ret2,hitratio2] = prep_new(X);
