
# coding: utf-8

# In[ ]:



# # coding: utf-8

# # In[1]:


import numpy as np
import h5py
import time
import copy
from random import randint
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# In[2]:


batch_size = 128

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR100(root='./',
                                            train=True,
                                            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                          transforms.RandomVerticalFlip(),
                                                                          transforms.ColorJitter(brightness=0.4),
                                                                          transforms.ToTensor()]),
                                                                         
                                            download=True) 

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) 

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader) 
# Mini-batch images and labels.
images, labels = data_iter.next() 


# In[3]:





# In[4]:


#number of hidden units

#Model architecture

class BasicBlock(nn.Module):


    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 32    

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout2d(p = 0.2)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv1_drop(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, kernel_size=4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1) 

def ResNet18():
    return ResNet(BasicBlock, [2,4,4,2])




model = ResNet18()


model.cuda()


# In[5]:


#Stochastic gradient descent optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 80


model.train()

train_loss = []


# In[6]:


#Train Model
for epoch in range(num_epochs):
    train_accu = []
    
    for images, labels in train_loader:
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        
        #PyTorch "accumulates gradients", so we need to set the stored gradients to zero when there’s a new batch of data.
        optimizer.zero_grad()
        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        output = model(data)
        #The objective function is the negative log-likelihood function.
        loss = F.nll_loss(output, target)
        #This calculates the gradients (via backpropagation)
        loss.backward()
        train_loss.append(loss.data[0])
        #The parameters for the model are updated using stochastic gradient descent.
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        #Calculate accuracy on the training set.
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch)


# # Save and load the entire model.
# torch.save(model, 'model.ckpt')
# model = torch.load('model.ckpt')     
    
# In[ ]:


# Download and construct CIFAR-10 dataset.
test_dataset = torchvision.datasets.CIFAR100(root='./',
                                            train=False,
                                            transform=transforms.Compose([transforms.ToTensor()
                                                                          ]),
                                            download=True) 

# Data loader (this provides queues and threads in a very simple way).
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False) 

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(test_loader) 
# Mini-batch images and labels.
images, labels = data_iter.next() 


# # In[ ]:

#Calculate accuracy of trained model on the Test Set
model.eval()
test_accu = []
for images, labels in test_loader:
    data, target = Variable(images).cuda(), Variable(labels).cuda()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
    test_accu.append(accuracy)
accuracy_test = np.mean(test_accu)
print(accuracy_test)







# In[6]:




