# 严格被试划分：Stage 1和Stage 2使用相同的验证被试

## 问题

之前的实现中，Stage 1和Stage 2各自独立地划分验证集，导致**数据泄露**：

```
修改前：
Stage 1:
  - 训练：14个源被试的所有数据（包括被试4, 12, 13）
  - 验证：14个源被试的部分trials

Stage 2:
  - 训练：11个源被试（如被试0,1,2,3,5,6,7,8,9,10,11）
  - 验证：3个源被试（如被试4, 12, 13）

问题：Stage 2的验证被试（4, 12, 13）在Stage 1中已经"见过"了！
```

## 解决方案

**统一被试划分**：Stage 1和Stage 2使用相同的训练/验证被试划分

```
修改后：
1. 先划分源被试：
   - 训练被试：11个（如被试0,1,2,3,5,6,7,8,9,10,11）
   - 验证被试：3个（如被试4, 12, 13）

2. Stage 1:
   - 训练：11个训练被试的所有trials
   - 验证：11个训练被试的部分trials
   - ✅ 从未见过验证被试（4, 12, 13）

3. Stage 2:
   - 训练：11个训练被试（按被试采样）
   - 验证：3个验证被试（4, 12, 13）
   - ✅ 验证被试在Stage 1中从未见过

4. 最终测试：
   - 目标被试：1个全新的被试（如被试0）
   - ✅ 目标被试在两个阶段都从未见过
```

---

## 代码修改

### 1. data_loader.py

#### prepare_stage1_data()

**新增参数**：`val_subject_indices`

```python
def prepare_stage1_data(source_data, source_labels, val_subject_indices=None):
    """
    如果提供了val_subject_indices，则只使用训练被试
    """
    if val_subject_indices is not None:
        # 排除验证被试
        train_subject_indices = [i for i in range(len(source_data))
                                if i not in val_subject_indices]
        source_data = [source_data[i] for i in train_subject_indices]
        source_labels = [source_labels[i] for i in train_subject_indices]

    # 然后按trial划分训练/验证集
    ...
```

#### prepare_stage2_data()

**新增参数**：`val_subject_indices`

```python
def prepare_stage2_data(source_data, source_labels, val_ratio=0.2,
                       val_subject_indices=None):
    """
    如果提供了val_subject_indices，则使用这些被试作为验证集
    否则随机划分
    """
    if val_subject_indices is None:
        # 随机划分验证被试
        ...
    else:
        # 使用指定的验证被试
        train_subject_indices = [i for i in range(len(source_data))
                                if i not in val_subject_indices]
        val_subject_indices = val_subject_indices

    return domain_loaders, val_loader, val_subject_indices
```

### 2. main.py

**修改流程**：先划分被试，再传递给两个阶段

```python
# 1. 加载数据
source_data, source_labels, target_data, target_labels = load_seed_data(...)

# 2. 先划分源被试
_, _, val_subject_indices = prepare_stage2_data(
    source_data, source_labels, val_ratio=0.2
)

# 3. 传递给Stage 1（排除验证被试）
model_stage1, best_val_acc = train_stage1(
    source_data, source_labels, test_id, config,
    val_subject_indices=val_subject_indices  # ← 新增
)

# 4. 传递给Stage 2（使用验证被试验证）
model_stage2, best_test_acc = train_stage2(
    source_data, source_labels, target_data, target_labels,
    test_id, config,
    val_subject_indices=val_subject_indices  # ← 新增
)
```

### 3. train_stage1.py 和 train_stage2.py

**新增参数**：`val_subject_indices`

```python
def train_stage1(source_data, source_labels, test_id, config,
                val_subject_indices=None):
    train_loader, val_loader = prepare_stage1_data(
        source_data, source_labels,
        val_subject_indices=val_subject_indices  # ← 传递
    )
    ...

def train_stage2(source_data, source_labels, target_data, target_labels,
                test_id, config, val_subject_indices=None):
    domain_loaders, val_loader, _ = prepare_stage2_data(
        source_data, source_labels,
        val_subject_indices=val_subject_indices  # ← 传递
    )
    ...
```

---

## 完整数据流

### LOSO实验（test_id=0为例）

```
15个被试：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

步骤1：选择目标被试
  目标被试：0（1个）

步骤2：划分剩余14个源被试
  源被试：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]（14个）

  随机划分（固定随机种子）：
  - 训练被试：[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13]（11个）
  - 验证被试：[4, 12, 14]（3个）

步骤3：Stage 1
  - 训练：11个训练被试的trials（按trial划分80%-20%）
    - Train: 11个被试 × 12个trial = 132个trial
    - Val: 11个被试 × 3个trial = 33个trial
  - ✅ 从未见过验证被试[4, 12, 14]

步骤4：Stage 2
  - 训练：11个训练被试（按被试采样）
  - 验证：3个验证被试[4, 12, 14]的所有数据
  - ✅ 验证被试在Stage 1中从未见过

步骤5：最终测试
  - 测试：目标被试[0]的所有数据
  - ✅ 目标被试在两个阶段都从未见过
```

---

## 验证方法

运行 `python main.py`，查看输出：

```
================================================================================
SUBJECT SPLIT
================================================================================
Stage 2 Subject split: 11 subjects for train, 3 subjects for val
  Train subjects: [ 1  2  3  5  6  7  8  9 10 11 13]
  Val subjects: [ 4 12 14]

Validation subjects indices (will be excluded from Stage 1): [4, 12, 14]

================================================================================
STAGE 1: ERM Pre-training (using training subjects only)
================================================================================
Stage 1: Using 11 training subjects (excluding val subjects [4, 12, 14])
Total trials from 11 subjects: 165
...

================================================================================
STAGE 2: DG Fine-tuning (using training subjects for train, val subjects for validation)
================================================================================
Stage 2 Subject split: 11 subjects for train, 3 subjects for val
  Train subjects: [ 1  2  3  5  6  7  8  9 10 11 13]
  Val subjects: [ 4 12 14]
...
```

**关键检查点**：
- ✅ Stage 1使用的被试数 = 11（排除3个验证被试）
- ✅ Stage 1和Stage 2的训练被试相同
- ✅ Stage 1和Stage 2的验证被试相同
- ✅ Stage 2的验证被试在Stage 1中从未出现

---

## 优势

### 1. 避免数据泄露
- Stage 1从未见过Stage 2的验证被试
- 真实反映跨被试泛化能力

### 2. 评估一致性
- Stage 1的验证集性能和Stage 2的验证集性能来自相同的被试
- 可以公平比较两个阶段

### 3. 符合实际应用
- 训练时使用一组被试
- 验证时使用另一组被试（从未见过）
- 测试时使用全新的被试（从未见过）

---

## FAQ

### Q1: 为什么不在Stage 1也用被试验证？

**A**:
- **Stage 1目标**：学习情绪分类能力
  - 验证集只要能评估分类能力即可
  - 按trial划分增加验证集多样性

- **Stage 2目标**：学习跨被试泛化
  - 验证集必须模拟目标域（全新被试）
  - 因此验证集必须是完整的被试

### Q2: Stage 1的验证集还有意义吗？

**A**: 有意义！
- Stage 1的验证集用于**选择最佳epoch**
- 从11个训练被试的trials中划分，确保不会过拟合
- Stage 1的验证集性能反映的是**分类能力**，不是泛化能力

### Q3: 为什么不直接用Stage 2的验证被试作为Stage 1的验证集？

**A**: 可以，但不是必须的：
- **当前方案**：Stage 1按trial划分（在训练被试内部）
  - 优点：更多验证数据，训练被试利用率高
  - 缺点：Stage 1验证集性能不能预测跨被试性能

- **替代方案**：Stage 1也使用验证被试
  - 优点：Stage 1验证集性能可以预测跨被试性能
  - 缺点：Stage 1训练数据减少，可能影响学习

**当前方案更适合**，因为Stage 1的目标是学习分类，不是泛化。

---

## 总结

这次修改确保了：
1. ✅ Stage 1和Stage 2使用相同的被试划分
2. ✅ Stage 2的验证被试在Stage 1中从未出现
3. ✅ 避免数据泄露，真实反映跨被试泛化能力
4. ✅ 符合严格的域泛化设定

**核心思想**：被试级别的一致性 > trial级别的划分方式
