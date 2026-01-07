"""
EEG Encoder Model for Two-Stage Domain Generalization
"""
import torch
import torch.nn as nn


class EEG_Encoder(nn.Module):
    """
    EEG情绪识别编码器

    输入: [batch, 310] DE特征
    输出: [batch, 3] 情绪类别概率
    """
    def __init__(self, input_dim=310, hidden_dim=128, num_classes=3, dropout=0.5):
        super(EEG_Encoder, self).__init__()

        # Backbone: 特征提取器
        self.backbone = nn.Sequential(
            # Block 1: 输入层
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2: 隐藏层1
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier: 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: [batch, 310] DE特征

        Returns:
            logits: [batch, 3] 未归一化的logits
            features: [batch, 128] 特征向量
        """
        # 特征提取
        features = self.backbone(x)  # [batch, 128]

        # 分类
        logits = self.classifier(features)  # [batch, 3]

        return logits, features

    def predict(self, x):
        """
        预测（推理时使用）

        Args:
            x: [batch, 310]

        Returns:
            predictions: [batch] 类别索引
        """
        logits, _ = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions


def freeze_backbone(model, freeze_ratio=0.7):
    """
    冻结backbone的前部分层

    Args:
        model: EEG_Encoder
        freeze_ratio: 冻结比例 (0-1)

    Returns:
        model: 冻结后的模型
    """
    # 获取backbone的所有层
    backbone_layers = list(model.backbone.children())
    num_layers = len(backbone_layers)
    num_freeze = int(num_layers * freeze_ratio)

    # 冻结前num_freeze层
    for i in range(num_freeze):
        for param in backbone_layers[i].parameters():
            param.requires_grad = False

    # 后面的层保持可训练
    for i in range(num_freeze, num_layers):
        for param in backbone_layers[i].parameters():
            param.requires_grad = True

    # Classifier完全可训练
    for param in model.classifier.parameters():
        param.requires_grad = True

    print(f"Frozen {num_freeze}/{num_layers} layers of backbone")

    return model


if __name__ == '__main__':
    # 测试模型
    model = EEG_Encoder()
    x = torch.randn(32, 310)  # batch=32, feature=310
    logits, features = model(x)
    print(f"Input: {x.shape}")
    print(f"Features: {features.shape}")
    print(f"Logits: {logits.shape}")

    # 测试预测
    preds = model.predict(x)
    print(f"Predictions: {preds.shape}")

    # 测试冻结
    print("\n=== Testing freeze ===")
    model = freeze_backbone(model, freeze_ratio=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params} ({100*trainable_params/total_params:.1f}%)")
