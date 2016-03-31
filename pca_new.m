function [prec2,prec4,prec6,prec8,re2,re4,re6,re8] = pca_new(trainuser,bandwagonattack2,bandwagonattack6,averageattack,randomattack);

usernum=943;
itemnum=1682;
as=floor(943*0.1);
users=trainuser;
[prec2,id2,re2]=pcacal(users,usernum,itemnum,bandwagonattack2,as);
[prec4,id4,re4]=pcacal(users,usernum,itemnum,bandwagonattack6,as);
[prec6,id6,re6]=pcacal(users,usernum,itemnum,averageattack,as);
[prec8,id8,re8]=pcacal(users,usernum,itemnum,randomattack,as);
        
        

