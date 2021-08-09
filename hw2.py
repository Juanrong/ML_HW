import numpy as np 
import random
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch import nn 
import pandas as pd 


#read data
train_data = np.load('/home/zhangjr/Desktop/kaggle/ml2021spring-hw2/train_11.npy',encoding='latin1')
train_label = np.load('/home/zhangjr/Desktop/kaggle/ml2021spring-hw2/train_label_11.npy',encoding='latin1')
train_label = np.array(train_label,dtype="int")




#it is not necessary to convert into one-hot vector

##################################################################
#class_count = len(np.unique(train_label))                       #
#def one_hot(x,class_count):                                     #
#
#    return torch.eye(class_count)[x,:]                          #
#train_label = np.array(train_label,dtype="float")               #
#train_label = one_hot(train_label,class_count)                  #
##################################################################


#seperate train into for_train and for_validation
index = list(range(len(train_data)))
random.shuffle(index)


for_train_data = train_data[index[0:979932]]
#for_train_data = (for_train_data-np.mean(for_train_data,axis=0))/np.std(for_train_data,axis=0)
for_train_data = torch.from_numpy(for_train_data)
for_train_label = train_label[index[0:979932]]
for_train_label = torch.from_numpy(for_train_label)

for_validation_data = train_data[index[979932:]]
#for_validationn_data = (for_validation_data-np.mean(for_validation_data,axis=0))/np.std(for_validation_data,axis=0)
for_validation_data = torch.from_numpy(for_validation_data)
for_validation_label = train_label[index[979932:]]
for_validation_label = torch.from_numpy(for_validation_label)



#create dataloader

deal_train_dataset = TensorDataset(for_train_data,for_train_label)
deal_validation_dataset = TensorDataset(for_validation_data,for_validation_label)

bs=500

train_dataloader = DataLoader(dataset=deal_train_dataset,batch_size=bs,shuffle=True)
validation_dataloader = DataLoader(dataset=deal_validation_dataset,batch_size=bs)

device = "cuda" if torch.cuda.is_available() else "cpu"

#define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(429,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,39)
        )
    
    def forward(self,x):
        
        return self.linear_relu_stack(x)


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.005)

epochs = 50

acc=[]

for epoch in range(epochs):
    size = len(train_dataloader.dataset)
    for batch,(X,y) in enumerate(train_dataloader):
        X,y = X.to(device),y.to(device)

        pred = model(X.float())
        loss = loss_fn(pred,y.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        '''
        if batch%100==0:
            print("epoch:",epoch+1,"loss:",loss.item(),batch*len(X))
        '''


    model.eval()
    validation_loss,correct = 0,0
    with torch.no_grad():
        for X,y in validation_dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X.float())
            validation_loss += loss_fn(pred,y.long()).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
        validation_loss /= len(validation_dataloader)
        correct /= len(validation_dataloader.dataset)
        acc.append(correct)
        print("validation error:",validation_loss,"accuarcy:",correct)

np.savetxt('/home/zhangjr/Desktop/kaggle/ml2021spring-hw2/acc.txt',acc)

torch.save(model.state_dict(),'model.pth')
print("saved pytorch model state to model.pth")

#load model 
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

test_data = np.load('/home/zhangjr/Desktop/kaggle/ml2021spring-hw2/test_11.npy',encoding='latin1')
#test_data = (test_data-np.mean(test_data,axis=0))/np.std(test_data)
test_data = torch.from_numpy(test_data)

model.eval()
with torch.no_grad():
    pred = model(test_data.float())
    pred = pred.numpy()

    final = pred.argmax(1)
    pd_data = pd.DataFrame(final, columns=['Class'])
    pd_data.to_csv('/home/zhangjr/Desktop/kaggle/ml2021spring-hw2/pd_data.csv')

