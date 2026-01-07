# Stage 1 修复总结

## 修复的问题

### 1. ✅ Checkpoint保存位置错误
**问题**：权重文件保存在错误的目录（可能保存在数据目录而非项目目录）

**修复**：
- 在 `train_stage1.py` 中添加 `PROJECT_ROOT` 变量，指向脚本所在目录
- 使用 `os.path.join(PROJECT_ROOT, 'checkpoints', ...)` 构建绝对路径
- 同时修复 `train_stage2.py` 中的checkpoint加载路径

**修改文件**：
- `train_stage1.py` (lines 15-16, 140-145)
- `train_stage2.py` (lines 16-17, 83-90)

### 2. ✅ Trial级别的数据划分
**问题**：之前随机shuffle所有样本后划分，导致同一个trial的数据可能同时出现在训练集和验证集中

**修复**：
- 修改 `prepare_stage1_data()` 函数，按trial边界划分数据
- SEED数据集每个被试有15个trial（对应15个video）
- 将每个被试的数据按trial切分，然后随机分配trials到训练集和验证集
- 确保同一个trial的所有样本要么全部在训练集，要么全部在验证集

**修改文件**：
- `data_loader.py` (lines 91-181)

**修改逻辑**：
```python
# 之前：随机shuffle所有样本后划分
all_data = np.concatenate(source_data, axis=0)
indices = np.random.permutation(len(all_data))
split_idx = int(len(all_data) * 0.8)

# 现在：按trial划分
video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
all_trials = []
for each subject:
    for trial_id, num_samples in enumerate(video_time):
        extract trial data
        append to all_trials

randomly split trials (80%-20%)
concatenate train trials
concatenate val trials
```

### 3. ✅ 配置参数统一
**问题**：`train_stage1.py` 和 `train_stage2.py` 使用 `config['epochs']`，但 `main.py` 使用 `config['epochs_stage1']` 和 `config['epochs_stage2']`

**修复**：
- 统一使用 `config['epochs_stage1']` 和 `config['epochs_stage2']`

**修改文件**：
- `train_stage1.py` (lines 60, 72, 79, 131, 165)
- `train_stage2.py` (lines 111, 127, 143, 191)

## 验证方法

### 测试trial划分：
```bash
cd E:\code\20240804SDA_DDA_Pairwise2\two_stage_dg
python test_trial_split.py
```

### 重新训练Stage 1：
```bash
python train_stage1.py
```

检查checkpoint是否保存在正确位置：
```
E:\code\20240804SDA_DDA_Pairwise2\two_stage_dg\checkpoints\stage1_testid0_best.pth
```

### 运行完整流程：
```bash
python train_stage1.py  # Stage 1训练
python train_stage2.py  # Stage 2训练（应该能正确加载Stage 1权重）
```

## 数据泄露的说明

**为什么需要trial级别划分？**

在脑电情绪识别任务中：
- 每个trial对应一个video刺激（约1分钟）
- 同一个trial内的多个时间样本具有高度相关性
- 如果同一个trial的样本既在训练集又在测试集，会导致：
  - 模型"看到"测试集的相似样本
  - 高估模型性能
  - 无法真实反映跨被试泛化能力

**修复后**：
- 每个trial作为独立单位
- 同一个trial的所有样本保证只在训练集或只在测试集
- 更真实的泛化性能评估

## 检查点保存路径

**修复前**：
```python
save_dir = 'checkpoints'  # 相对路径，可能保存到错误位置
torch.save(model.state_dict(), f'{save_dir}/stage1_testid{test_id}_best.pth')
```

**修复后**：
```python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
save_path = os.path.join(save_dir, f'stage1_testid{test_id}_best.pth')
torch.save(model.state_dict(), save_path)
```

现在checkpoint会确保保存在项目目录下：
```
E:\code\20240804SDA_DDA_Pairwise2\two_stage_dg\checkpoints\
```
