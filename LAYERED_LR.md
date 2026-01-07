# 分层学习率配置说明

## 概述

为了更精细地控制DGCNN在Stage 2的训练，我们实现了**分层学习率**（Layered Learning Rate）策略。

## 背景

DGCNN在Stage 2使用策略2（脑结构冻结）时，只有约1.4K参数可训练（占总参数0.14%）：
- GraphConv.weight: 640参数
- B1ReLU.bias: 64参数
- fc2: 768参数

不同层的作用不同，因此应该使用不同的学习率。

## 分层学习率策略

### 原则
- **越低层越保守**：特征提取层（GraphConv）小心调整
- **越高层越激进**：分类头（fc2）快速适应
- **偏置参数可以用更大学习率**：B1ReLU.bias可以快速调整

### 推荐配置

```python
config = {
    'use_layered_lr': True,  # 启用分层学习率
    'lr_graphconv': 0.5,     # GraphConv: 0.5x (相对于Stage 1的lr)
    'lr_brelu': 1.0,         # B1ReLU: 1.0x
    'lr_fc2': 1.0,           # fc2: 1.0x
}
```

**实际学习率**（假设Stage 1的lr=1e-3）：
- GraphConv: 5e-4（保守，避免破坏特征提取能力）
- B1ReLU: 1e-3（适中，快速调整激活偏置）
- fc2: 1e-3（激进，快速适应新类别边界）

## 使用方法

### 1. 在main.py中启用

```python
config = {
    'model_type': 'dgcnn',
    'use_layered_lr': True,  # 启用分层学习率
    'lr_graphconv': 0.5,
    'lr_brelu': 1.0,
    'lr_fc2': 1.0,
    # ... 其他配置
}
```

### 2. 禁用分层学习率（使用统一学习率）

```python
config = {
    'model_type': 'dgcnn',
    'use_layered_lr': False,  # 禁用，使用统一学习率
    'lr_ratio': 0.5,  # 所有层使用0.5x
    # ... 其他配置
}
```

### 3. MLP模型

MLP模型不支持分层学习率（因为freeze_backbone的实现不同），会自动使用统一学习率：
```python
config = {
    'model_type': 'mlp',
    'use_layered_lr': True,  # 这个设置会被忽略
    'lr_ratio': 0.5,  # MLP会使用这个统一学习率
}
```

## 不同学习率策略对比

| 策略 | 配置 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **统一学习率（保守）** | lr_ratio=0.5 | 小数据集，源域和目标域差异大 | 安全，不会破坏特征 | 收敛慢 |
| **统一学习率（适中）** | lr_ratio=1.0 | 中等数据集 | 平衡 | 可能初期不稳定 |
| **分层学习率（推荐）** | use_layered_lr=True | DGCNN，源域和目标域有一定差异 | 精细控制，符合微调最佳实践 | 需要调参 |
| **分层学习率（激进）** | lr_graphconv=1.0 | 大数据集，源域和目标域差异小 | 收敛快 | 可能过拟合 |

## 调参建议

### GraphConv学习率（lr_graphconv）
- **0.3-0.5**：保守，适用于源域和目标域差异大的情况
- **0.5-1.0**：适中，一般情况
- **1.0-2.0**：激进，适用于源域和目标域相似的情况

### B1ReLU学习率（lr_brelu）
- **0.5-1.0**：保守
- **1.0-2.0**：适中（推荐）
- **2.0-5.0**：激进

### fc2学习率（lr_fc2）
- **0.5-1.0**：保守
- **1.0-2.0**：适中（推荐）
- **2.0-5.0**：激进（适用于快速适应新类别边界）

## 实现细节

代码在`train_stage2.py`的优化器部分：

```python
if config.get('use_layered_lr', False) and isinstance(model, EEG_DGCNN):
    # 分层学习率
    optimizer = optim.Adam([
        {'params': model.graph_convs.parameters(), 'lr': base_lr * config['lr_graphconv']},
        {'params': model.b_relus.parameters(), 'lr': base_lr * config['lr_brelu']},
        {'params': model.fc2.parameters(), 'lr': base_lr * config['lr_fc2']}
    ], weight_decay=config['weight_decay'])
else:
    # 统一学习率
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr * config['lr_ratio'],
        weight_decay=config['weight_decay']
    )
```

## 实验建议

建议进行以下对比实验：

1. **基线**：统一学习率（lr_ratio=0.5）
2. **推荐配置**：分层学习率（0.5, 1.0, 1.0）
3. **激进配置**：分层学习率（1.0, 2.0, 2.0）

观察哪种配置在LOSO实验上表现最好。
