from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import torch
import random
from tqdm import tqdm
# dataSet 处理
class ImgDataSet(Dataset):
    # 类构造器
    def __init__(self,main_folder,transform = None):
        self.main_folder = main_folder
        self.transform = transform
        self.images = []
        self.labels = []
        # pass

        # 遍历所有的文件夹，创建数据集
        for label in os.listdir(main_folder):
            label_folder = os.path.join(main_folder,label)  # 不同label图片的文件夹路径

            if os.path.isdir(label_folder):
                for image in os.listdir(label_folder):
                    image_path = os.path.join(label_folder,image)
                    self.images.append(image_path)
                    self.labels.append(int(label))


    # 获取到一个包含数据和标签的元组
    # 同时会动态加载图片，所以只需要存储图片路径即可
    def __getitem__(self, index):

        img_path = self.images[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]  # 获取标签索引
        label = torch.tensor(label, dtype=torch.long)  # 转换为长整型 Tensor
        return image, label  # 确保返回的是 (image, label)
    # 获取数据集长度
    def __len__(self):
        return self.images.__len__()
        # pass

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        # 计算测试集比例
        test_ratio = 1 - train_ratio - val_ratio

        # 随机打乱数据
        indices = list(range(len(self.images)))
        random.shuffle(indices)

        train_size = int(len(self.images) * train_ratio)
        val_size = int(len(self.images) * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return (
            ImgDataSetSubset(self, train_indices),
            ImgDataSetSubset(self, val_indices),
            ImgDataSetSubset(self, test_indices),
        )

class ImgDataSetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

main_folder = 'data/img'

dataset = ImgDataSet(main_folder=main_folder,transform=transform)

print(len(dataset))

train_dataset,val_dataset,test_dataset = dataset.split_dataset()
print(len(train_dataset), len(val_dataset), len(test_dataset))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)  # 展平
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 假设 dataset 是你已经创建好的 ImgDataSet 实例
train_dataset, val_dataset, test_dataset = dataset.split_dataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 模型、损失函数和优化器
model = SimpleCNN(num_classes=4).to(device)  # 将模型转移到 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    # 使用 tqdm 包装 train_loader
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        images, labels = images.to(device), labels.to(device)  # 将数据转移到 GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 验证模型
model.eval()
with torch.no_grad():
    total, correct = 0, 0
    # 使用 tqdm 包装 val_loader
    for images, labels in tqdm(val_loader, desc='Validation', unit='batch'):
        images, labels = images.to(device), labels.to(device)  # 将数据转移到 GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')