from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        label = self.labels[idx]
        return image, label

# 加载 CLIP 模型和处理器
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# 指定本地文件夹路径
model_save_path = "./clip_model"  # 存储模型的文件夹路径
processor_save_path = "./clip_processor"  # 存储处理器的文件夹路径

# 从本地文件夹加载模型和处理器
model = CLIPModel.from_pretrained(model_save_path)
processor = CLIPProcessor.from_pretrained(processor_save_path)

from sklearn.model_selection import train_test_split
import os
from PIL import Image
import torch
from torchvision import transforms

def build_dataset_from_folder(folder_path, processor):
    imgs = []
    labels = []

    cnt = 0

    # 定义图片预处理步骤
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为Tensor类型
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

    # 遍历文件夹中的子文件夹（假设每个子文件夹代表一个标签）
    for label, folder in enumerate(os.listdir(folder_path)):
        folder_path_full = os.path.join(folder_path, folder)

        # 确保是文件夹
        if os.path.isdir(folder_path_full):
            with open('log.txt', 'a') as f:
                print(f"Processing folder: {folder}", file=f)
            print(f"Processing folder: {folder}")

            # 遍历该文件夹中的所有图片文件
            for img in os.listdir(folder_path_full):
                img_path = os.path.join(folder_path_full, img)

                # 加载图片并进行预处理
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image)

                img_tensor = (img_tensor + 1) / 2

                imgs.append(img_tensor)
                labels.append(int(label))

                cnt += 1
                if cnt == 2000:
                    cnt = 0
                    break

    # 将数据集拆分为训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.2, random_state=42, shuffle=True
    )

    # 将数据传入CustomDataset（你可能需要自己定义CustomDataset类来处理）
    train_dataset = CustomDataset(train_texts, train_labels, processor)
    test_dataset = CustomDataset(test_texts, test_labels, processor)

    return train_dataset, test_dataset


# 文件夹路径
folder_path = r"data"  # 请替换为实际的文件夹路径

# 创建数据集
train_dataset , test_dataset= build_dataset_from_folder(folder_path, processor)

with open('log.txt','a') as f:
    print(len(train_dataset),file=f)
    print(len(test_dataset),file=f)

# 创建自定义数据集
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 创建分类头部（线性层），假设有 N 个类别
num_classes = 3  # 假设类别数量是标签数
classification_head = nn.Linear(model.config.projection_dim, num_classes)

# 将模型和分类头部放到设备（GPU 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
classification_head.to(device)

# 设置优化器
optimizer = torch.optim.Adam(list(model.parameters()) + list(classification_head.parameters()), lr=1e-5)



epochs = 3
for epoch in range(epochs):
    model.train()
    classification_head.train()
    total_loss = 0
    for images, labels in tqdm(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)  # 确保标签是Tensor类型

        # 将数字标签转换为文本标签（如果你已经有对应的文本标签，可以跳过这步）
        text_labels = [f"Class {label.item()}" for label in labels]  # 示例，按需修改

        # 处理输入数据（这里传递文本和图像）
        inputs = processor(text=text_labels, images=images, return_tensors="pt", padding=True).to(device)

        # 获取 CLIP 输出
        outputs = model(**inputs)
        image_features = outputs.image_embeds  # 图像特征

        # 使用分类头部进行分类
        logits_per_image = classification_head(image_features)

        # 计算损失（例如交叉熵损失）
        loss = nn.CrossEntropyLoss()(logits_per_image, labels)
        total_loss += loss.item()

        # 反向传播并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with open('log.txt', 'a') as f:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}", file=f)

# 测试
from sklearn.metrics import accuracy_score

model.eval()  # 切换到评估模式

# 创建 DataLoader 用于测试集
from torch.utils.data import DataLoader

# 定义 batch size
batch_size = 16
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 评估模型
predictions = []
true_labels = []

for images, labels in tqdm(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)  # 确保标签是Tensor类型

        # 将数字标签转换为文本标签（如果你已经有对应的文本标签，可以跳过这步）
        text_labels = [f"Class {label.item()}" for label in labels]  # 示例，按需修改

        true_labels.append(text_labels)
        # 处理输入数据（这里传递文本和图像）
        inputs = processor(text=text_labels, images=images, return_tensors="pt", padding=True).to(device)

        # 获取 CLIP 输出
        outputs = model(**inputs)
        image_features = outputs.image_embeds  # 图像特征

        # 使用分类头部进行分类
        logits_per_image = classification_head(image_features)

        predictions.append(logits_per_image)


accuracy = accuracy_score(true_labels, predictions)
with open('log.txt','a') as f:
    print(f"Test Accuracy: {accuracy * 100:.2f}%",file=f)
