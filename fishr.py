"""
Fishr (Fisher Invariance Regularization) Loss
基于ICLR 2024: "Fishr: Invariant Risk Minimization for Out-of-Distribution Generalization"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FishrLoss(nn.Module):
    """
    Fishr损失函数

    目标：让不同域上的梯度方向一致
    原理：如果不同被试的梯度一致 → 学到了不变特征
    """
    def __init__(self):
        super(FishrLoss, self).__init__()

    def forward(self, model, domain_batches, criterion):
        """
        计算Fishr损失

        Args:
            model: 神经网络
            domain_batches: list of m tuples (data, labels, domain_id)
            criterion: 损失函数（如CrossEntropy）

        Returns:
            fishr_loss: 标量
        """
        m = len(domain_batches)
        if m < 2:
            return torch.tensor(0.0)

        # 获取模型所在设备
        device = next(model.parameters()).device

        # 1. 计算每个域的梯度
        domain_gradients = []

        for data, labels, domain_id in domain_batches:
            # 确保数据在正确的设备上
            data = data.to(device)
            labels = labels.to(device)

            # 前向传播
            logits, features = model(data)
            loss = criterion(logits, labels)

            # 反向传播（计算梯度）
            model.zero_grad()
            loss.backward(retain_graph=True)

            # 提取梯度
            grads = []
            for param in model.parameters():
                if param.grad is not None and param.requires_grad:
                    grads.append(param.grad.detach().flatten())

            if len(grads) > 0:
                grad_vector = torch.cat(grads)
                domain_gradients.append(grad_vector)

        if len(domain_gradients) < 2:
            return torch.tensor(0.0)

        # 2. 堆叠成矩阵 [m, num_params]
        G = torch.stack(domain_gradients)

        # 3. 计算平均梯度
        G_bar = G.mean(dim=0, keepdim=True)  # [1, num_params]

        # 4. Fishr损失：每个域梯度与平均梯度的距离
        fishr_loss = ((G - G_bar) ** 2).sum() / m

        return fishr_loss


if __name__ == '__main__':
    # 测试Fishr损失
    from model import EEG_Encoder

    # 创建模型
    model = EEG_Encoder()
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss()
    fishr_loss_fn = FishrLoss()

    # 模拟4个域的batch
    domain_batches = []
    for i in range(4):
        data = torch.randn(16, 310).to(device)
        labels = torch.randint(0, 3, (16, 3)).float().to(device)
        domain_batches.append((data, labels, i))

    # 计算Fishr损失
    fishr_loss = fishr_loss_fn(model, domain_batches, criterion)
    print(f"Fishr Loss: {fishr_loss.item():.4f}")
