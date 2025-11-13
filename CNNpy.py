import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BATCH_SIZE = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 
pipeline = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307, ), (0.3081, ))  
])
train_set = datasets.MNIST('data',train=True,download=True,transform=pipeline)
test_set = datasets.MNIST('data',train=False,download=True,transform=pipeline)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
with open('data/MNIST/raw/t10k-images-idx3-ubyte','rb') as f:
    image_data = f.read(16 + 784)[16:]
img_np = np.frombuffer(image_data, dtype=np.uint8).reshape(28, 28)
cv2.imwrite('digit.jpg',img_np)

class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
 
    def forward(self,x):
        input_size = x.size(0) 
        x = self.conv1(x)      
        x = F.relu(x)         
        x = F.max_pool2d(x,2,2)     
 
        x = self.conv2(x)      
        x = F.relu(x)
 
        x = x.view(input_size,-1)   
 
        x = self.fc1(x)         
        x = F.relu(x)
 
        x = self.fc2(x)        
 
        output = F.log_softmax(x, dim=1)    
        return output
 
 
model = Digit().to(DEVICE)
optimizer = optim.Adam(model.parameters())
 

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.max(1, keepdim=True)[1]
        loss.backward()
        optimizer.step()
        if batch_idx % 3000 == 0:
            print('Train Epoch : {} \t Loss : {:.6f}'.format(epoch, loss.item()))
 

def test_mode(model, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0.0
    with torch.no_grad():  
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            # pred = torch.max(output, dim=1)
            # pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('Test —— Average Loss: {:.4f}, Accuracy : {:.3f} %\n'.format(test_loss, 100.0*correct/len(test_loader.dataset)))
 
    return 100.0 * correct / len(test_loader.dataset)
 
acc_list_test = []
for epoch in range(1,EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    acc = test_mode(model, DEVICE, test_loader)
    acc_list_test.append(acc)
 
plt.figure(dpi=100)  
plt.plot(acc_list_test, marker='o', linestyle='--', color='b')
plt.title('Test Accuracy Trend')
plt.grid(True)
