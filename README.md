# Two-Stage Domain Generalization for EEG Emotion Recognition

基于Fishr的两阶段域泛化框架，用于EEG情绪识别的跨被试泛化。

## 📁 文件结构

```
two_stage_dg/
├── data_loader.py      # 数据加载模块
├── model.py            # EEG编码器模型
├── fishr.py            # Fishr损失函数
├── train_stage1.py     # Stage 1训练（ERM预训练）
├── train_stage2.py     # Stage 2训练（DG微调）
├── main.py             # 主脚本（完整LOSO实验）
└── README.md           # 本文件
```

## 🎯 核心思想

### Stage 1: ERM预训练
- **目标**：让模型学会"情绪分类"
- **损失**：仅使用CrossEntropy
- **数据**：混合所有源被试（不分域）
- **输出**：能分类但跨被试不稳定的模型

### Stage 2: DG微调
- **目标**：学习跨被试泛化
- **损失**：CrossEntropy + λ·Fishr
- **数据**：按被试采样（每轮采样4个被试）
- **输出**：跨被试稳定的模型

## 🚀 快速开始

### 1. 测试单个模块

```bash
# 测试数据加载
python data_loader.py

# 测试模型
python model.py

# 测试Fishr损失
python fishr.py
```

### 2. 快速测试（单个被试）

```bash
# Stage 1测试（test_id=0）
python train_stage1.py

# Stage 2测试（test_id=0，需要先运行Stage 1）
python train_stage2.py
```

### 3. 完整LOSO实验（15个被试）

```bash
# 运行完整实验（会花费较长时间）
python main.py
```

## ⚙️ 配置说明

### main.py中的关键配置

```python
config = {
    # 数据
    'session': 1,  # SEED的session编号 (1, 2, 3)

    # 模型
    'hidden_dim': 128,   # 隐藏层维度
    'dropout': 0.5,      # Dropout比率

    # Stage 1
    'lr': 1e-3,              # 学习率
    'weight_decay': 1e-4,    # L2正则
    'epochs_stage1': 60,     # 训练轮数
    'patience': 15,          # 早停耐心值

    # Stage 2
    'lr_ratio': 0.5,        # Stage 2学习率比率（0.5x）
    'epochs_stage2': 50,     # 训练轮数
    'freeze_ratio': 0.7,    # 冻结backbone比例（70%）
    'm': 4,                  # 每次采样域数量
    'lambda_fishr_max': 0.3, # Fishr最大权重
    'warmup_epochs': 20,     # Fishr权重warm-up轮数

    # 其他
    'seed': 2024,  # 随机种子
}
```

## 📊 预期输出

### Stage 1输出
```
Epoch 1/60: Train Loss=1.0234, Train Acc=45.23%, Val Loss=0.9876, Val Acc=52.31%
  → Saved best model (Val Acc=52.31%)
...
Best Val Acc: 88.45%
```

### Stage 2输出
```
Initial Test Acc (Stage 1): 76.32%
Epoch 1/50: Loss=0.7234, CE=0.6123, Fishr=0.0369, Train Acc=78.45%, Test Acc=79.12%, λ=0.000
...
Best Test Acc: 84.56%
Improvement: 8.24%
```

### 最终LOSO结果
```
Per-subject accuracies:
  Subject  1: 85.23%
  Subject  2: 83.45%
  ...
  Subject 15: 86.78%

Average over 15 subjects:
  Mean: 84.56%
  Std:  3.21%
  Min:  78.92%
  Max:  89.34%
```

## 🔧 超参数调优建议

### 如果Stage 1不收敛（<80%）
- 增加训练轮数：`epochs_stage1: 60 → 80`
- 降低学习率：`lr: 1e-3 → 5e-4`
- 增加Dropout：`dropout: 0.5 → 0.6`

### 如果Stage 2过拟合（Stage 2 < Stage 1）
- 增加冻结比例：`freeze_ratio: 0.7 → 0.8`
- 降低Fishr权重：`lambda_fishr_max: 0.3 → 0.2`
- 减少训练轮数：`epochs_stage2: 50 → 30`

### 如果Stage 2欠拟合（提升很小<2%）
- 降低冻结比例：`freeze_ratio: 0.7 → 0.5`
- 增加Fishr权重：`lambda_fishr_max: 0.3 → 0.5`
- 延长warmup：`warmup_epochs: 20 → 30`

## 📝 依赖库

```bash
pip install torch numpy scikit-learn scipy tqdm
```

## 💡 关键设计

### 按被试归一化
- 每个被试独立计算z-score
- 避免测试集数据泄露

### 按域采样（Stage 2）
- 每次迭代采样4个被试
- Fishr需要区分不同域的梯度

### 渐进式Fishr权重
- 前20个epoch: 0 → 0.3
- 避免初期约束噪声特征

### 部分冻结
- 冻结backbone前70%
- 只微调高层特征

## ⏱️ 训练时间估计

单GPU (RTX 3060)：
- Stage 1: ~10分钟/被试
- Stage 2: ~15分钟/被试
- 完整LOSO: ~6小时

## 🐛 常见问题

### 1. 找不到数据文件
```
FileNotFoundError: H:\SEED\feature_for_net_session1_LDS_de
```
**解决**：修改`data_loader.py`中的`data_path`为你的实际路径

### 2. CUDA out of memory
**解决**：降低batch_size
- Stage 1: `batch_size: 64 → 32`
- Stage 2: `batch_size: 16 → 8`

### 3. Fishr损失为0
**原因**：只有1个域的batch
**解决**：确保`m >= 2`

## 📚 参考文献

- Fishr: "Fishr: Invariant Risk Minimization for Out-of-Distribution Generalization" (ICLR 2024)
- SEED Dataset: https://bcmi.sjtu.edu.cn/home/seed/

## 📧 联系

如有问题，请检查：
1. 数据路径是否正确
2. 数据格式是否为310维DE特征
3. GPU内存是否充足
