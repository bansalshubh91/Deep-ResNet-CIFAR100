
# coding: utf-8

# In[3]:



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
import torchvision.models as models
from torchvision.models.resnet import model_urls


# In[2]:


batch_size = 128

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR100(root='./',
                                            train=True,
                                            transform=transforms.Compose([transforms.Resize(256),
                                                                          transforms.CenterCrop(224),
                                                                          transforms.RandomHorizontalFlip(),
                                                                          transforms.RandomVerticalFlip(),
                                                                          transforms.ColorJitter(brightness=0.4),
                                                                          transforms.ToTensor()
                                                                          ]),
                                            download=True) 

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) 




# In[3]:





# In[4]:


#number of hidden units

#Model architecture

model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)



model.cuda()


# In[5]:


#Stochastic gradient descent optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50


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
        loss = F.cross_entropy(output, target)
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
                                            transform=transforms.Compose([transforms.Resize(256),
                                                                          transforms.CenterCrop(224),
                                                                          transforms.ToTensor()
                                                                          ]),
                                            download=True) 

# Data loader (this provides queues and threads in a very simple way).
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False) 




# # In[ ]:

#Calculate accuracy of trained model on the Test Set
model.eval()
test_accu = []
for images, labels in test_loader:
    data, target = Variable(images).cuda(), Variable(labels).cuda()
    optimizer.zero_grad()
    output = model(data)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
    test_accu.append(accuracy)
accuracy_test = np.mean(test_accu)
print(accuracy_test)






