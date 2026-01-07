# 模型选择说明

## 概述

框架现在支持多种模型架构，可以通过配置文件轻松切换：
- **MLP**: 多层感知机（原有架构）
- **DGCNN**: Dynamical Graph Convolutional Neural Network

---

## 快速开始

### 使用MLP模型（默认）

```python
config = {
    'model_type': 'mlp',  # 选择MLP
    'hidden_dim': 128,
    'dropout': 0.5,
    ...
}
```

### 使用DGCNN模型

```python
config = {
    'model_type': 'dgcnn',  # 选择DGCNN
    'k': 2,  # Chebyshev多项式阶数
    'layers': [64],  # GCN层输出通道数
    'dropout': 0.5,
    ...
}
```

---

## 模型架构对比

### 1. EEG_MLP（多层感知机）

**架构**：
```
Input: [batch, 310] 展平的DE特征
  ↓
Backbone:
  Linear(310 → 256) + BN + ReLU + Dropout
  Linear(256 → 128) + BN + ReLU + Dropout
  ↓
Features: [batch, 128]
  ↓
Classifier:
  Linear(128 → 64) + BN + ReLU + Dropout
  Linear(64 → 3)
  ↓
Output: [batch, 3] logits
```

**参数量**：
- Backbone: ~113K
- Classifier: ~8.6K
- Total: ~121K

**优点**：
- 简单高效，训练快速
- 参数量适中，不易过拟合
- 适合快速原型开发

**缺点**：
- 没有利用EEG的脑区连接信息
- 对输入特征质量依赖较高

**适用场景**：
- 快速建立baseline
- 计算资源有限
- 需要快速迭代

---

### 2. EEG_DGCNN（图卷积神经网络）

**架构**：
```
Input: [batch, 310] 或 [batch, 62, 5]
  - 62个电极 × 5个频段

  ↓ reshape为 [batch, 62, 5]

Graph Convolution Layers:
  GraphConv(k=2, 5 → 64) + B1ReLU + Dropout
  ↓
Features: [batch, 62, 64]

  ↓ Flatten
Features: [batch, 62×64=3968]

  ↓
Classifier:
  Linear(3968 → 256) + Dropout
  Linear(256 → 3)

  ↓
Output: [batch, 3] logits
```

**关键组件**：
1. **可学习邻接矩阵** `adj`: [62, 62]
   - 学习脑区之间的连接强度
   - 初始化为随机值，训练时更新

2. **拉普拉斯矩阵** `lap = I - D^(-1/2) * A * D^(-1/2)`
   - 用于图卷积操作
   - 每个forward都重新计算

3. **Chebyshev多项式**（k=2）
   - T_0(x) = I
   - T_1(x) = L * x
   - 用于扩展感受野

4. **B1ReLU激活函数**
   - `ReLU(bias + x)`
   - 每个通道独立学习偏置

**参数量**：
- GraphConv: ~1.4K (k=2, 5→64)
- Adjacency: 3.8K (62×62)
- Classifier: ~1M (3968→256→3)
- Total: ~1M

**优点**：
- 利用EEG的脑区连接信息
- 可学习脑区之间的连接模式
- 在EEG任务上表现更好

**缺点**：
- 参数量较大（主要是分类器）
- 训练时间较长
- 需要更多调参

**适用场景**：
- 追求更好的性能
- 有足够的计算资源
- 需要利用脑区连接信息

---

## 配置参数详解

### MLP专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_dim` | 128 | Backbone输出特征维度 |
| `dropout` | 0.5 | Dropout比率 |
| `input_dim` | 310 | 输入维度（固定） |

### DGCNN专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_electrodes` | 62 | 电极数量（固定） |
| `in_channels` | 5 | 每个电极的特征维度（固定） |
| `k` | 2 | Chebyshev多项式阶数（1-5） |
| `layers` | [64] | GCN层输出通道数 |
| `dropout` | 0.5 | Dropout比率 |

**k的影响**：
- k=1: 只考虑当前节点（局部）
- k=2: 考虑1阶邻居（推荐）
- k=3-5: 考虑更大感受野（可能过拟合）

**layers的影响**：
- `[64]`: 单层GCN（推荐）
- `[64, 128]`: 两层GCN（参数更多）
- `[128]`: 更宽的单层（表达能力更强）

---

## 使用示例

### 1. 单独训练Stage 1（MLP）

```bash
python train_stage1.py
```

**配置** (train_stage1.py):
```python
config = {
    'model_type': 'mlp',
    'hidden_dim': 128,
    'dropout': 0.5,
    ...
}
```

### 2. 单独训练Stage 1（DGCNN）

修改 `train_stage1.py`:
```python
config = {
    'model_type': 'dgcnn',  # 改为dgcnn
    'k': 2,
    'layers': [64],
    'dropout': 0.5,
    ...
}
```

然后运行：
```bash
python train_stage1.py
```

### 3. 完整LOSO实验（MLP）

修改 `main.py`:
```python
config = {
    'model_type': 'mlp',  # 选择mlp
    ...
}
```

运行：
```bash
python main.py
```

### 4. 完整LOSO实验（DGCNN）

修改 `main.py`:
```python
config = {
    'model_type': 'dgcnn',  # 选择dgcnn
    'k': 2,
    'layers': [64],
    ...
}
```

运行：
```bash
python main.py
```

---

## 性能对比（预期）

| 模型 | 参数量 | 训练时间 | 预期准确率 | 稳定性 |
|------|--------|----------|-----------|--------|
| MLP | 121K | 快 | 80-85% | 高 |
| DGCNN | ~1M | 慢2-3倍 | 85-90% | 中等 |

**说明**：
- DGCNN参数量主要集中在分类器（3968→256→3）
- 可以减小DGCNN分类器来加速：
  ```python
  # 修改models.py中的DGCNN分类器
  self.fc = nn.Linear(num_electrodes * layers[-1], 128)  # 256→128
  self.fc2 = nn.Linear(128, num_classes)
  ```

---

## 添加新模型

### 步骤1: 在models.py中定义新模型

```python
class EEG_YourModel(nn.Module):
    def __init__(self, **kwargs):
        super(EEG_YourModel, self).__init__()
        # 定义模型架构
        ...

    def forward(self, x):
        """
        Args:
            x: [batch, 310] 输入

        Returns:
            logits: [batch, 3]
            features: [batch, feature_dim]
        """
        # 前向传播
        ...
        return logits, features
```

### 步骤2: 在create_model中注册

```python
def create_model(model_type='mlp', **kwargs):
    if model_type.lower() == 'mlp':
        model = EEG_MLP(...)
    elif model_type.lower() == 'dgcnn':
        model = EEG_DGCNN(...)
    elif model_type.lower() == 'your_model':  # 新增
        model = EEG_YourModel(...)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
```

### 步骤3: 更新freeze_backbone（如果需要）

```python
def freeze_backbone(model, freeze_ratio=0.7):
    if isinstance(model, EEG_YourModel):
        # 实现冻结逻辑
        ...
    return model
```

### 步骤4: 在config中添加参数

```python
config = {
    'model_type': 'your_model',
    'your_param1': value1,
    'your_param2': value2,
    ...
}
```

### 步骤5: 在train_stage1.py和train_stage2.py中添加配置

```python
elif model_type.lower() == 'your_model':
    model = create_model(
        model_type='your_model',
        param1=config.get('your_param1', default_value),
        param2=config.get('your_param2', default_value),
        ...
    ).to(device)
```

---

## 测试新模型

```bash
# 测试模型是否正常工作
python models.py

# 应该看到：
# Testing EEG_MLP
# Testing EEG_DGCNN
# Testing create_model factory
# Testing freeze_backbone
# All tests passed!
```

---

## 常见问题

### Q1: DGCNN训练很慢怎么办？

**A**: 减小分类器维度：
```python
# 修改models.py
self.fc = nn.Linear(num_electrodes * layers[-1], 64)  # 256→64
```

### Q2: DGCNN显存不足怎么办？

**A**: 减小batch_size或使用更少的GCN层：
```python
config = {
    'model_type': 'dgcnn',
    'layers': [32],  # [64] → [32]
    ...
}

# 或者在train_stage1.py中减小batch_size
train_loader = Data.DataLoader(..., batch_size=32)  # 64→32
```

### Q3: 如何对比不同模型？

**A**: 运行多次实验，记录结果：
```python
# MLP
config = {'model_type': 'mlp', ...}
python main.py

# DGCNN
config = {'model_type': 'dgcnn', ...}
python main.py

# 比较results/目录下的结果文件
```

### Q4: checkpoint可以跨模型加载吗？

**A**: 不可以！不同模型的架构不同，权重不兼容。

### Q5: 如何调整DGCNN的邻接矩阵初始化？

**A**: 修改models.py中的init_weights：
```python
# 使用不同的初始化
nn.init.kaiming_uniform_(self.adj)  # 改为Kaiming初始化
# 或使用预定义的邻接矩阵
self.adj.data = torch.from_numpy(predefined_adj).float()
```

---

## 总结

现在框架支持：
- ✅ 灵活的模型选择（MLP/DGCNN）
- ✅ 统一的接口（create_model工厂函数）
- ✅ 方便扩展（添加新模型只需几步）
- ✅ 配置文件驱动（无需修改代码）

推荐流程：
1. 先用MLP建立baseline（快速）
2. 再用DGCNN提升性能（慢但效果好）
3. 根据需求选择合适的模型
