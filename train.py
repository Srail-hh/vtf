import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from models import MyModel_cat, MyModel_conv, MyModel_cat_dropout, MyModel_conv_dropout
from datasets import CustomDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 训练

# 数据加载器
transform = None  # You can define your own transformations here if needed
train_dataset = CustomDataset('data/train', transform=transform)
val_dataset = CustomDataset('data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#PATH='state_dict_model_cat.pth'
#PATH='state_dict_model_conv.pth'
#PATH='state_dict_model_cat_dropout.pth'
PATH='state_dict_model_conv_dropout.pth'
# 定义模型、损失函数和优化器
model = MyModel_conv_dropout(num_classes=63)  # 假设有63个类别
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 训练
model.to(device)
model.train()  # 设置模型为训练模式
num_epochs = 200
best_accuracy = 0.0  # 初始化最佳准确率

for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs_touch, inputs_vision, labels in train_loader:
        inputs_touch = inputs_touch.to(device)
        inputs_vision = inputs_vision.to(device)
        labels = labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs_touch, inputs_vision)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计预测准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 统计损失
        total_loss += loss.item()

    # 输出本轮训练的损失和准确率
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {(correct/total)*100:.2f}%")
    
    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for touch_image, vision_image, label in val_loader:
            touch_image, vision_image, label = touch_image.to(device), vision_image.to(device), label.to(device)
            outputs = model(touch_image, vision_image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
    
    # 如果当前验证准确率高于最佳准确率，则保存模型权重
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), PATH)
        print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
       
    # 每20个epoch保存一次checkpoint
    if (epoch + 1) % 20 == 0:
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
    
    model.train()  # 切回训练模式
  
print("Finished Training")
