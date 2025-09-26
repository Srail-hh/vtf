import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 导入模型和数据集模块
from DiFFmodels import (
    WiseDiffModule, DualModel,
    ResNet18FeatureExtractor, GroupAttention
)
from datasets import CustomDataset

# =====================
# 配置与超参数
# =====================
MODEL_PATH = 'best_model.pth'
TEST_DATA_PATH = 'data/test'
BATCH_SIZE = 32
NUM_CLASSES = 6
CONF_MATRIX_SAVE_PATH = 'Confusion_Matrix.png'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================
# 加载测试数据
# =====================
# 如有需要可定义 transform
transform = None
test_dataset = CustomDataset(TEST_DATA_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# 初始化模型
# =====================
model = DualModel(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# =====================
# 模型推理与评估
# =====================
total = 0
correct = 0
predictions = []
true_labels = []

with torch.no_grad():
    for touch_img, vision_img, labels in test_loader:
        touch_img, vision_img, labels = touch_img.to(DEVICE), vision_img.to(DEVICE), labels.to(DEVICE)

        y_vis, y_real, y_ir = model(touch_img, vision_img)
        _, preds = torch.max(y_real.data, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# =====================
# 输出准确率
# =====================
accuracy = 100.0 * correct / total
recall = recall_score(true_labels, predictions, average='macro')
print(f"[INFO] Test Accuracy: {accuracy:.2f}%")
print(f"[INFO] Test recall: {recall:.2f}%")

# =====================
# 混淆矩阵可视化
# =====================
conf_matrix = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.arange(NUM_CLASSES),
    yticklabels=np.arange(NUM_CLASSES)
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(CONF_MATRIX_SAVE_PATH)
print(f"[INFO] Confusion matrix saved to '{CONF_MATRIX_SAVE_PATH}'")
