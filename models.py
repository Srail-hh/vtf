import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        self.mask = torch.ones(x.shape, device=x.device)
        if not self.training:
            return x
        else:
            self.mask = torch.bernoulli((1 - self.p) * torch.ones(x.shape, device=x.device))
            return x * self.mask

    def get_mask(self):
        return self.mask


class MyModel_cat_dropout(nn.Module):
    def __init__(self, num_classes, lambda_val=0.5):
        super(MyModel_cat_dropout, self).__init__()
        self.touch_cnn = models.resnet18(pretrained=True)
        self.vision_cnn = models.resnet18(pretrained=True)
        
        # 添加新的全连接层
        self.fc = nn.Linear(1000 * 2, num_classes)
        self.dropout = CustomDropout(0.5)  # 初始dropout率设为0.5
        self.lambda_val = lambda_val
        self.relu = nn.ReLU()
        self.flag = 0

    def forward(self, x_touch, x_vision):
        x_touch = self.touch_cnn(x_touch)
        x_vision = self.vision_cnn(x_vision)
        x = torch.cat((x_touch, x_vision), dim=1)
        if self.flag:
            self.lambda_val = torch.max(self.Y, 1).values.mean()
            # 计算并更新Dropout率
            self.zi = F.softmax((self.fc.weight.data@self.dropout.mask.T),dim=0).mean()  # 利用softmax来模拟加权和的归一化
            p = (1-self.lambda_val)*torch.exp(-self.zi)
            self.flag = 1
        else:
            p = (1-self.lambda_val)

        self.dropout.p = p  # 更新Dropout层的概率

        x = self.dropout(x)
        x = self.fc(x)
        self.Y = F.softmax(x,dim=1)
        return x

# 模型定义
class MyModel_cat(nn.Module):
    def __init__(self, num_classes):
        super(MyModel_cat, self).__init__()
        self.touch_cnn = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18 作为触摸图像的 CNN
        self.vision_cnn = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18 作为视觉图像的 CNN
        self.fc = nn.Linear(1000 * 2, num_classes)  # 全连接层将两个 CNN 的输出连接并进行分类

    def forward(self, x_touch, x_vision):
        x_touch = self.touch_cnn(x_touch)
        x_vision = self.vision_cnn(x_vision)
        x = torch.cat((x_touch, x_vision), dim=1)  # 拼接两个 CNN 的输出
        x = self.fc(x)
        return x
    
# 模型定义
class MyModel_conv(nn.Module):
    def __init__(self, num_classes):
        super(MyModel_conv, self).__init__()
        self.touch_cnn = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18 作为触摸图像的 CNN
        self.vision_cnn = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18 作为视觉图像的 CNN
        self.fc = nn.Linear(1000, num_classes)  # 全连接层将两个 CNN 的输出连接并进行分类
        self.W = nn.Parameter(torch.rand(9))
        self.conv = nn.Conv2d(2, 1, 1, padding=0)

    def forward(self, x_touch, x_vision):
        x_touch = self.touch_cnn(x_touch)
        x_vision = self.vision_cnn(x_vision)

        T = torch.reshape(x_touch, [-1, 50,20])
        V = torch.reshape(x_vision, [-1, 50,20])
        Y = torch.zeros(48,18)

        diff_T = []
        diff_V = []
        sigma = 2
        for i in range(3):
            for j in range(3):
                diff_T.append(T[:,i:50-2+i,j:20-2+j]-T[:,1:49,1:19])
                diff_V.append(V[:,i:50-2+i,j:20-2+j]-V[:,1:49,1:19])
        T_temp = torch.stack(diff_T, dim=3)
        V_temp = torch.stack(diff_V, dim=3)

        Y = (torch.exp(-T_temp**2/sigma**2)*V_temp)@self.W
        Y1 = F.pad(Y, [1,1,1,1])

        F1 = torch.stack([V,Y1], dim=3).permute(0,3,1,2)

        F2 = self.conv(F1)

        # x = torch.cat((x_touch, x_vision), dim=1)  # 拼接两个 CNN 的输出
        x = self.fc(F2.reshape(F2.shape[0],1000))
        return x
    
# 模型定义
class MyModel_conv_dropout(nn.Module):
    def __init__(self, num_classes, lambda_val = 0.5):
        super(MyModel_conv_dropout, self).__init__()
        self.touch_cnn = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18 作为触摸图像的 CNN
        self.vision_cnn = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18 作为视觉图像的 CNN
        self.fc = nn.Linear(1000, num_classes)  # 全连接层将两个 CNN 的输出连接并进行分类
        self.fc_T = nn.Linear(1000, 1000)  # 全连接层将两个 CNN 的输出连接并进行分类
        self.fc_V = nn.Linear(1000, 1000)  # 全连接层将两个 CNN 的输出连接并进行分类
        self.relu = nn.ReLU()
        self.W = nn.Parameter(torch.rand(9))
        self.conv = nn.Conv2d(2, 1, 1, padding=0)
        self.dropout = CustomDropout(p=0.5)  # dropout训练
        self.lambda_val = lambda_val
        self.flag = 0

    def forward(self, x_touch, x_vision):
        x_touch = self.touch_cnn(x_touch)
        x_vision = self.vision_cnn(x_vision)

        x_touch = self.dropout(self.fc_T(x_touch))
        x_vision = self.dropout(self.fc_T(x_vision))

        T = torch.reshape(x_touch, [-1, 50,20])
        V = torch.reshape(x_vision, [-1, 50,20])
        Y = torch.zeros(48,18)

        diff_T = []
        diff_V = []
        sigma = 2
        for i in range(3):
            for j in range(3):
                diff_T.append(T[:,i:50-2+i,j:20-2+j]-T[:,1:49,1:19])
                diff_V.append(V[:,i:50-2+i,j:20-2+j]-V[:,1:49,1:19])
        T_temp = torch.stack(diff_T, dim=3)
        V_temp = torch.stack(diff_V, dim=3)

        Y = (torch.exp(-T_temp**2/sigma**2)*V_temp)@self.W
        Y1 = F.pad(Y, [1,1,1,1])

        F1 = torch.stack([V,Y1], dim=3).permute(0,3,1,2)

        F2 = self.conv(F1)
        x = F2.reshape(F2.shape[0],1000)

        # x = torch.cat((x_touch, x_vision), dim=1)  # 拼接两个 CNN 的输出
        if self.flag:
            self.lambda_val = torch.max(self.Y, 1).values.mean()
            # 计算并更新Dropout率
            self.zi = F.softmax((self.fc.weight.data@self.dropout.mask.T),dim=0).mean()  # 利用softmax来模拟加权和的归一化
            p = (1-self.lambda_val)*torch.exp(-self.zi)
            self.flag = 1
        else:
            p = (1-self.lambda_val)

        self.dropout.p = p  # 更新Dropout层的概率

        x = self.dropout(x)
        x = self.fc(x)
        self.Y = F.softmax(x,dim=1)
        return x