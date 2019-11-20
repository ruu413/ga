import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BackGammon:
    def __init__(self,table=None):
        if table is None:
            self.table = np.array([
                0,#プレイヤ2のあまり[0]
                -2,0,0,0,0,5,#[1-6]
                0,3,0,0,0,-5,#[7-12]
                5,0,0,0,-3,0,#[13-18]
                -5,0,0,0,0,2,#[19-24]
                0#プレイヤ1のあまり[25]
            ])
        else:
            self.table = np.copy(table)
    def dice(self):
        return np.random.randint(1,6,2)
    def isGoalablePlayer(self,table):
        return np.all(table[7:26]<=0)
    def isGoal(self):
        return np.all(self.table[1:26]<=0)
    def changePlayer(self):
        return BackGammon(np.flip(np.copy(-self.table)))
    def playPlayerRetNext(self,dice):
        if(self.table[25]>=2):
            nexttable=np.copy(self.table)
            if nexttable[25-dice[0]]>=0:
                nexttable[25-dice[0]]+=1
                nexttable[25]-=1
            elif nexttable[25-dice[0]]==-1:
                nexttable[25-dice[0]]=1
                nexttable[25]-=1
                nexttable[0]-=1
            if nexttable[25-dice[1]]>=0:
                nexttable[25-dice[1]]+=1
                nexttable[25]-=1
            elif nexttable[25-dice[1]]==-1:
                nexttable[25-dice[1]]=1
                nexttable[25]-=1
                nexttable[0]-=1
            return np.unique(np.copy([nexttable]),axis=0)
        elif self.table[25]==1:
            nexttables=[]
            if self.table[25-dice[0]]>-2:
                table=np.copy(self.table)
                if table[25-dice[0]]>=0:
                    table[25-dice[0]]+=1
                    table[25]=dice[1]
                elif table[25-dice[0]]==-1:
                    table[25-dice[0]]=1
                    table[25]=dice[1]#diceの値を入れとく
                    table[0]-=1
                nexttables.append(table)
            if self.table[25-dice[1]]>-2&dice[0]!=dice[1]:
                table=np.copy(self.table)
                if table[25-dice[1]]>=0:
                    table[25-dice[1]]+=1
                    table[25]=dice[0]
                elif table[25-dice[1]]==-1:
                    table[25-dice[1]]=1
                    table[25]=dice[0]
                    table[0]-=1
                nexttables.append(table)
            if len(nexttables) == 0:
                return np.copy(np.array([self.table]))
            ret=[]
            for table in nexttables:
                ids = np.where(self.table[1:25]>0)[0]+1
                for i in ids:
                    dice = table[25]
                    table_=np.copy(table)
                    table_[25]=0
                    if i-dice<1:
                        continue
                    if table_[i-dice]<=-2:
                        continue
                    elif table_[i-dice]==-1:
                        table_[i]-=1
                        table_[i-dice]=1
                        table_[0]-=1
                    else:
                        table_[i]-=1
                        table_[i-dice]+=1
                    ret.append(table_)
            if len(ret) ==0:
                return np.copy(np.array([self.table]))
            return np.unique(np.copy(ret),axis=0)
        ret=[]
        ids = np.where(self.table[1:25]>0)[0]+1
        for i in ids:
            table = np.copy(self.table)
            if i-dice[0]<1:
                if self.isGoalablePlayer(table):
                    table[i]-=1
                else:
                    continue
            else:
                if table[i-dice[0]] <= -2:
                    continue
                elif table[i-dice[0]]== -1:
                    table[i]-=1
                    table[i-dice[0]]=1
                    table[0]-=1
                else:
                    table[i]-=1
                    table[i-dice[0]]+=1
            ids_ = np.where(table[1:25]>0)[0]+1
            if ids_.size==0:
                ret.append(table)
                continue
            for j in ids_:
                
                table_ = np.copy(table)
                if j-dice[1]<1:
                    if self.isGoalablePlayer(table_):
                        table_[j]-=1
                    else:
                        continue
                else:
                    if table_[j-dice[1]] <= -2:
                        continue
                    elif table_[j-dice[1]]== -1:
                        table_[j]-=1
                        table_[j-dice[1]]=1
                        table_[0]-=1
                    else:
                        table_[j]-=1
                        table_[j-dice[1]]+=1
                ret.append(table_)
        if len(ret) == 0:
            return np.copy(np.array([self.table]))
        return np.unique(np.copy(ret),axis=0)
                

    

class EvalModel(nn.Module):
    def __init__(self):
        super(EvalModel,self).__init__()
        self.fc1 = nn.Linear(26,3)
        #weight 10*10 bias 10
        self.fc2 = nn.Linear(3,3)
        self.fc3 = nn.Linear(3,1)
        #weight 1*10 bias 1
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def setweight(self,gene):
        fc1w = gene[0:78].reshape((3,26))
        model.fc1.weight=torch.nn.Parameter(torch.from_numpy(fc1w))
        fc2w = gene[78:87].reshape((3,3))
        model.fc2.weight=torch.nn.Parameter(torch.from_numpy(fc2w))
        fc3w = gene[87:90].reshape((1,3))
        model.fc3.weight=torch.nn.Parameter(torch.from_numpy(fc3w))
        fc1b = gene[90:93].reshape((1,3))
        model.fc1.bias=torch.nn.Parameter(torch.from_numpy(fc1b))
        fc2b = gene[93:96].reshape((1,3))
        model.fc2.bias=torch.nn.Parameter(torch.from_numpy(fc2b))
        fc3b = gene[96:97].reshape((1,1))
        model.fc3.bias=torch.nn.Parameter(torch.from_numpy(fc3b))
class Gene:
    def __init__(self,gene):
        self.gene = np.array(gene)
        self.evaluate=0
        self.s=0.3
        self.size=self.gene.size
    def blx(self,gen):
        vec=np.vstack((self.gene,gen.gene))
        min=np.min(vec,axis=0)
        max=np.max(vec,axis=0)
        mean=(max+min)/2
        norm = np.random.normal(0,1,self.size)
        d=max-min
        gene=norm*(d/2)+mean
        return Gene(gene)


class GeneticAlgo:
    def __init__(self,size,paramsize):
        self.size=size
        self.paramsize=paramsize
        self.genes = [Gene(np.random.random(paramsize)) for i in range(size)]

    def evaluate(self):
        for gene in self.genes:
            gene.evaluate=-(gene.gene[0]-0.7)*(gene.gene[0]-0.7)
    def challengerandom(self):
        num = 0
        model1=EvalModel()
        ii=0
        for i in range(100):
            model1.setweight(((self.genes[i].gene)))
            b=BackGammon()
            print(i)
            while True:
                
                if(b.table is None):
                    break
                t = b.playPlayerRetNext(b.dice())
                b=BackGammon(t[np.random.randint(0,len(t))])
                if b.isGoal():
                    break
                #print(b.table)
                
                
                b=b.changePlayer()
                
                maxt = None
                maxev = -np.inf
                if(b.table is None):
                    break
                nextt = b.playPlayerRetNext(b.dice())
                maxt=None
                #for table in nextt:
                #    ev=model1(torch.from_numpy(table.astype(np.float32)))
                #    
                #    if maxev < float(ev):
                #        maxt = table
                #        maxev = ev
                
                b=BackGammon(nextt[np.random.randint(0,len(nextt))])
                b=BackGammon(maxt)
                
                if b.isGoal() :
                    num += 1
                    break
                b=b.changePlayer()

                
                #print(b.table)
                
        print("num win")
        print(num)

    
    def challengerandom2(self):
        num = 0
        model1=EvalModel()
        ii=0
        model1.setweight(((self.genes[0].gene)))
        for i in range(100):
            b=BackGammon()
            print(i)
            while True:
                if(b.table is None):
                    break
                t = b.playPlayerRetNext(b.dice())
                b=BackGammon(t[np.random.randint(0,len(t))])
                if b.isGoal():
                    print(b.table)
                    break
                
                
                
                b=b.changePlayer()
                if(b.table is None):
                    break
                nextt = b.playPlayerRetNext(b.dice())
                maxt=None
                maxev = -np.inf
                for table in nextt:
                    ev=model1(torch.from_numpy(table.astype(np.float32)))
                    
                    if maxev < float(ev):
                        maxt = table
                        maxev = float(ev)
                
                #b=BackGammon(nextt[np.random.randint(0,len(nextt))])
                b=BackGammon(maxt)
                
                #print(b.table)
                
                if b.isGoal() :
                    num += 1
                    break
                b=b.changePlayer()
                
                #print(b.table)
                
        print("num win")
        print(num)
    def select(self):
        #self.genes.sort(reverse=True,key=lambda gen: gen.evaluate)
        #self.genes = self.genes[0:20]
        newgenes=[]
        model1=EvalModel()
        model2=EvalModel()
        for i in range(50):
            model1.setweight(((self.genes[i].gene)))
            model2.setweight((self.genes[i+1].gene))
            b=BackGammon()
            print("n")
            print(i)
            while True:
                if(b.table is None):
                    break
                maxt = None
                maxev = -np.inf
                for table in b.playPlayerRetNext(b.dice()):
                    ev=model1(torch.from_numpy(table.astype(np.float32)))
                    if maxev < float(ev):
                        maxt = table
                        maxev = float(ev)
                b=BackGammon(maxt)
                #print(b.table)
                if b.isGoal() :
                    newgenes.append(self.genes[i])
                    break
                
                b=b.changePlayer()
                if(b.table is None):
                    break
                maxt = None
                maxev = -np.inf
                for table in b.playPlayerRetNext(b.dice()):
                    ev=model2(torch.from_numpy(table.astype(np.float32)))
                    if maxev < float(ev):
                        maxt = table
                        maxev = float(ev)
                b=BackGammon(maxt)
                if b.isGoal() :
                    newgenes.append(self.genes[i+1])
                    break
                b=b.changePlayer()
        
        self.genes=newgenes
        newgenes=[]
        for i in range(25):
            print("m")
            print(i)
            model1.setweight(((self.genes[i].gene)))
            model2.setweight(((self.genes[i+1].gene)))
            b=BackGammon()
            while True:
                if(b.table is None):
                    break
                maxt = None
                maxev = -np.inf
                for table in b.playPlayerRetNext(b.dice()):
                    ev=model1(torch.from_numpy(table.astype(np.float32)))
                    if maxev < float(ev):
                        maxt = table
                        maxev = ev
                b=BackGammon(maxt)
                if b.isGoal() :
                    newgenes.append(self.genes[i])
                    break
                
                b=b.changePlayer()
                if(b.table is None):
                    break
                maxt = None
                maxev = -np.inf
                for table in b.playPlayerRetNext(b.dice()):
                    ev=model2(torch.from_numpy(table.astype(np.float32)))
                    if maxev < float(ev):
                        maxt = table
                        maxev = ev
                b=BackGammon(maxt)
                if b.isGoal() :
                    newgenes.append(self.genes[i+1])
                    break
                b=b.changePlayer()
            
        self.genes=newgenes                
    def kousa(self):
        for _ in range(75):
            i = np.random.randint(20)
            j = np.random.randint(19)
            if(i<=j):
                j+=1
            self.genes.append(self.genes[i].blx(self.genes[j]))
            random.shuffle(self.genes)



a = Gene([1,2])
b = Gene([3,4])
print(a.blx(b).gene)
c =[Gene([i,i]) for i in range(10)]
d = GeneticAlgo(100,97)
model=EvalModel()
print(model(torch.from_numpy(np.zeros((1,26),np.float32))))
d.challengerandom2()
for i in range(1000):
    print(i)
#    d.evaluate()
    d.select()
    #print(d.genes[0].gene)
    if(i%10==0):
        d.challengerandom2()
    d.kousa()
#d.evaluate()
#d.select()
model.setweight(d.genes[0].gene.astype(np.float32))
print(model(torch.from_numpy(np.zeros((1,26),np.float32))))
#t=torch.from_numpy(np.zeros((1,10)))
#print(model(t))
#=torch.nn.Parameter(torch.from_numpy(np.zeros((10,10),np.float32)))