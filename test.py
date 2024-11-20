import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from models import MyModel_cat, MyModel_conv, MyModel_cat_dropout, MyModel_conv_dropout
from datasets import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 训练

# 数据加载器
# 这里是你之前创建的 data_loader
transform = None  # You can define your own transformations here if needed
test_dataset = CustomDataset('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#PATH='state_dict_model_cat.pth'
#PATH='state_dict_model_conv.pth'
#PATH='state_dict_model_cat_dropout.pth'
PATH='state_dict_model_conv_dropout.pth'

# 在测试集上评估模型
model=MyModel_conv_dropout(num_classes=63)
model.load_state_dict(torch.load(PATH))
model.to(device)
#载入保存的模型参数
model.eval()
#不启用 BatchNormalization 和 Dropout
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for touch_image, vision_image, label in test_loader:
        touch_image, vision_image, label = touch_image.to(device), vision_image.to(device), label.to(device)
        outputs = model(touch_image, vision_image)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 在测试集上评估模型
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for touch_image, vision_image, label in test_loader:
        touch_image, vision_image, label = touch_image.to(device), vision_image.to(device), label.to(device)
        outputs = model(touch_image, vision_image)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(label.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(30, 24))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(63), yticklabels=range(63))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')