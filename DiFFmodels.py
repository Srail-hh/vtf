import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # 输出(B,512,7,7)

    def forward(self, x):
        return self.features(x)

class WiseDiffModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.rand(9))  # 9维可学习权重
        self.sigma = 2  # 高斯核宽度
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)

    def forward(self, T, V):
        batchsize, channels, height, width = T.shape
        assert height == 7 and width == 7, "空间维度必须为7x7"
        diff_T = []  
        diff_V = []
        T_orig = F.pad(T,  pad=(1, 1, 1, 1), mode='reflect')
        V_orig = F.pad(V,  pad=(1, 1, 1, 1), mode='reflect')
        T_reshaped = T_orig.view(-1, 9, 9)  
        V_reshaped = V_orig.view(-1, 9, 9)
 
        diff_T = []  
        diff_V = []  
        for i in range(3):  
            for j in range(3):  
        # 计算差分，注意空间维度的边界  
                diff_T.append(T_reshaped[:, i:7+i, j:7+j] - T_reshaped[:, 1:8, 1:8])  
                diff_V.append(V_reshaped[:, i:7+i, j:7+j] - V_reshaped[:, 1:8, 1:8])    
  
# 堆叠差分结果，得到形状为 [batchsize*512, height, width, 9]  
        T_temp = torch.stack(diff_T, dim=3)  
        V_temp = torch.stack(diff_V, dim=3)  
  
# 应用高斯加权和权重矩阵  
        Y_reshaped = (torch.exp(-T_temp ** 2 / self.sigma ** 2) * V_temp) @ self.W  
  
# 将结果重塑回原始形状 [batchsize, 512, ...]  
        Y = Y_reshaped.view(T.shape[0], T.shape[1], 7, 7)
        F1 =  torch.cat([V, Y], dim=1)  # (B,1024, 7, 7)


        x = self.conv(F1) # (B,512, 7, 7)
        return x

        

# -----------------------------------------------------------------------------
# 分组注意力模块
# -----------------------------------------------------------------------------

class GroupAttention(nn.Module):
    def __init__(self, groups: int = 32, group_channels: int = 16):
        super().__init__()
        assert groups * group_channels == 512, "groups×group_channels 必须等于 512"
        self.groups = groups
        self.group_channels = group_channels
        self.conv = nn.Conv2d(group_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,512,7,7)
        B, C, H, W = x.shape
        assert C == self.groups * self.group_channels
        att_maps = []
        for g in range(self.groups):
            chunk = x[:, g*self.group_channels:(g+1)*self.group_channels]
            att_maps.append(torch.sigmoid(self.conv(chunk)))  # (B,1,7,7)
        return torch.cat(att_maps, dim=1)  # (B,groups,7,7)


# -----------------------------------------------------------------------------
#  双模态分类网络
# -----------------------------------------------------------------------------

class DualModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 两个模态的 ResNet18 backbone
        self.visual_backbone = ResNet18FeatureExtractor()
        self.ir_backbone = ResNet18FeatureExtractor()
        # 融合
        self.fusion_gc = WiseDiffModule()
        # 注意力
        self.visual_att = GroupAttention()
        self.ir_att = GroupAttention()
        # 分类器 (输入 512 维全局池化向量)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )


    def forward(self, visual_img: torch.Tensor, ir_img: torch.Tensor,):
        # 1) 提取特征
        vis_feat = self.visual_backbone(visual_img)   # (B,512,7,7)
        ir_feat = self.ir_backbone(ir_img) # (B,512,7,7)
        # 2) 特征融合
        fused = self.fusion_gc(vis_feat, ir_feat)    # (B,512,7,7)
        # 3) 注意力
        vis_att = self.visual_att(vis_feat)           # (B,32,7,7)
        ir_att = self.ir_att(ir_feat)          # (B,32,7,7)
        fake_att = torch.rand_like(vis_att)
        # 4) 加权注意力
        B = fused.size(0)
        fused_g = fused.view(B, 32, 16, 7, 7)
        att_ir = (fused_g * (vis_att - fake_att).unsqueeze(2)).view(B, 512, 7, 7)
        att_vis = (fused_g * (ir_att - fake_att).unsqueeze(2)).view(B, 512, 7, 7)
        # helper: 全局均值池
        pool = lambda x: F.adaptive_avg_pool2d(x,1).view(x.size(0), -1)
        vis_pool = pool(att_vis)
        ir_pool = pool(att_ir)
        y_vis = self.classifier(vis_pool)
        y_ir = self.classifier(ir_pool)
# 计算置信度（softmax后的最大概率值）
        vis_conf = torch.softmax(y_vis, dim=1).max(dim=1)[0]
        ir_conf = torch.softmax(y_ir, dim=1).max(dim=1)[0]
        
        # 5) 置信度加权融合
        total_conf = vis_conf + ir_conf + 1e-8  # 防止除以零
        weight_vis = vis_conf / total_conf
        weight_ir = ir_conf / total_conf
       
        # 扩展权重维度以匹配特征图
        weight_vis = weight_vis.view(vis_att.size(0), 1, 1, 1)
        weight_ir = weight_ir.view(ir_att.size(0), 1, 1, 1)
        att_fused = (fused_g * (weight_vis * vis_att + weight_ir * ir_att).unsqueeze(2)).view(B, 512, 7, 7)

        real_pool = pool(att_fused)
        y_real = self.classifier(real_pool)
        return y_vis, y_real, y_ir
        

