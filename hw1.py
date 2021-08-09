import numpy as np 
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torch import nn 
import random


###########################################################data solving
train = np.loadtxt('/home/zhangjr/Desktop/kaggle/ml2021spring-hw1/covid.train.csv',delimiter=',',skiprows=1)

mean = np.mean(train[:,41:-1],axis=0)
std = np.std(train[:,41:-1],axis=0)

train[:,41:-1] = (train[:,41:-1]-mean)/std

random.shuffle(train)


index=list(range(1,41))
index.extend([58,76])

x_train_data = train[0:2160,index]
y_train_data = train[0:2160,-1]

x_test_data = train[2160:,index]
y_test_data = train[2160:,-1]

print(x_train_data.shape)
print(y_train_data.shape)

print(x_test_data.shape)
print(y_test_data.shape)




###################################################dataloader building
x_train_data = torch.from_numpy(x_train_data)
y_train_data = torch.from_numpy(y_train_data)

x_test_data = torch.from_numpy(x_test_data)
y_test_data = torch.from_numpy(y_test_data)
#print(x_data.size())
#print(y_data.size())



deal_train_dataset = TensorDataset(x_train_data,y_train_data)
deal_test_dataset = TensorDataset(x_test_data,y_test_data)

bs = 54


train_dataloader = DataLoader(dataset=deal_train_dataset,batch_size=bs,shuffle=True)
print(len(train_dataloader))

test_dataloader = DataLoader(dataset=deal_test_dataset,batch_size=bs,shuffle=True)
print(len(test_dataloader))





########################################################device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(42,21),
            nn.Sigmoid(),
            nn.Linear(21,1)
        )
    
    def forward(self,x):
        y_hat = self.linear_relu_stack(x)

        return y_hat



model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(),lr=0.005)


def train_model(dataloader,model,loss_fn,optimizer):
    s=0
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)

        pred = model(X.float())
        loss = loss_fn(pred,y.float().reshape(-1,1))
        
        s += loss*len(X)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("train:",s/len(dataloader.dataset))

def test_model(dataloader,model):
    s = 0 
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X.float())
            loss = loss_fn(pred,y.float().reshape(-1,1))
            s += loss*len(X)
    
        print("test:",s/len(dataloader.dataset))
    



epochs = 2000
for t in range(epochs):
    print("epoch",t+1)
    train_model(train_dataloader,model,loss_fn,optimizer)
    test_model(test_dataloader,model)

print("Done")


#save model
torch.save(model.state_dict(),'model.pth')
print("saved pytorch model state to model.pth")








#load model 
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))




sub = np.loadtxt('/home/zhangjr/Desktop/kaggle/ml2021spring-hw1/covid.test.csv',delimiter=',',skiprows=1)

sub[:,41:] = (sub[:,41:]-mean)/std





x_sub = sub[:,index]
x_sub = torch.from_numpy(x_sub)

import pandas as pd 



model.eval()
with torch.no_grad():
    pred = model(x_sub.float())
    pred = pred.numpy()
    

    #print(pred)
    pd_data = pd.DataFrame(pred, columns=['tested_positive'])
    pd_data.to_csv('/home/zhangjr/Desktop/kaggle/ml2021spring-hw1/pd_data.csv')
    
