#AlexNet & MNIST


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

#定义网络结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #AlexCONV1(3,96, k=11,s=4,p=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#AlexCONV2(96, 256,k=5,s=1,p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)#AlexPool2(k=3,s=2)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#AlexCONV3(256,384,k=3,s=1,p=1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)#AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256*3*3, 1024)  #AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512) #AlexFC6(4096,4096)
        self.fc8 = nn.Linear(512, 10)  #AlexFC6(4096,1000)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)#Alex: x = x.view(-1, 256*6*6)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x

#transform
transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),


])

transform1 = transforms.Compose([
                    transforms.ToTensor()
])

# 加载数据
trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True,num_workers=0)# windows下num_workers设置为0，不然有bug

testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform1)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)

#net
net = AlexNet()

#损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()

#优化器 这里用SGD
optimizer = optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

print("Start Training!")

num_epochs = 200 #训练次数

for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f'%(epoch+1, (i+1)*100, loss.item()))

print("Finished Traning")


#保存训练模型
torch.save(net, 'MNIST.pkl')
net = torch.load('MNIST.pkl')
#开始识别
with torch.no_grad():
    #在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images:{}%'.format(100 * correct / total)) #输出识别准确率
