# 导入包
# basic
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

#resize功能
from scipy import misc

# pytorch
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 设备设置
# torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行，pytorch-cpu版本不需要运行这句
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.1

# cifar10 分类索引
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 数据增广方法
transform = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机裁剪至32x32
    transforms.RandomCrop(32),
    # 转换至Tensor
    transforms.ToTensor(),
    #  归一化
#     transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))
    ])

# cifar10路径
cifar10Path = 'C://Users/35947/Desktop/Homework/data/'

#  训练数据集
train_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
                                             train=True,
                                             transform=transform,
                                             download=False)

# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
                                            train=False,
                                            transform=transform)

# 生成数据加载器
# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 查看数据,取一组batch
data_iter = iter(test_loader)

images, labels = next(data_iter)

# 取batch中的一张图像
idx = 15
image = images[idx].numpy()
image = np.transpose(image, (1,2,0))
fig, ax = plt.subplots()
ax.imshow(image)
plt.show()
classes[labels[idx].numpy()]


# 搭建卷积神经网络模型
# 两个卷积层
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            #  批归一化
            nn.BatchNorm2d(16),
            # ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(8 * 8 * 32, num_classes)

    # 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# 实例化一个模型，并迁移至gpu
model = ConvNet(num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 注意模型在GPU中，数据也要搬到GPU中
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 设置为评估模式
model.eval()

# 节省计算资源，不去计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

#  保存模型
torch.save(model.state_dict(), 'model.ckpt')

# 查看数据,取一组batch
data_iter = iter(test_loader)
images, labels = next(data_iter)

# 取batch中的一张图像，显示图像和真实标签
idx = 10
image = images[idx].numpy()
image = np.transpose(image, (1,2,0))
plt.imshow(image)
classes[labels[idx].numpy()]

# 转换为（B,C,H,W）大小
imagebatch = image.reshape(-1,3,32,32)

# 转换为torch tensor
image_tensor = torch.from_numpy(imagebatch)

# 调用模型进行评估
model.eval()
output = model(image_tensor.to(device))
_, predicted = torch.max(output.data, 1)
pre = predicted.cpu().numpy()
print(pre) # 查看预测结果ID
print(classes[pre[0]])


