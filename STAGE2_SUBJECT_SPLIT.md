# Stage 2 数据划分说明：按被试而非Trial

## 重要修改

Stage 2的验证集**必须按被试划分**，而不是按trial划分。

---

## 为什么按被试划分？

### Stage 2的目标
学习**跨被试泛化**，即：
- 训练被试：被试1, 2, 3, ...
- 测试被试：全新的被试N

### 验证集的作用
验证集应该模拟目标域（全新被试），因此：
- ✅ **正确**：验证集 = 完整的几个被试（如被试12, 13, 14）
- ❌ **错误**：验证集 = 所有被试的部分trials混合

---

## 两种划分方式对比

### 方式A：按被试划分 ✅（当前实现）

```python
# 14个源被试
# 随机选择：11个被试用于训练，3个被试用于验证

train_subjects = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]  # 11个被试
val_subjects = [4, 12, 13]  # 3个完整被试

domain_loaders = [loader for subj in train_subjects]  # 11个domain loaders
val_loader = merge(val_subjects)  # 3个被试的所有数据
```

**优点**：
- ✅ 验证集是完整的被试，模拟真实的跨被试场景
- ✅ 验证集性能可以预测目标域性能
- ✅ 符合域泛化的设定

### 方式B：按Trial划分 ❌（之前的错误实现）

```python
# 14个源被试，每个被试的trials被拆分
# 每个被试：12个trials训练，3个trials验证

for each subject:
    split trials into 12 train + 3 val

val_loader = merge(all subjects' val trials)  # 混合所有被试的验证trials
```

**缺点**：
- ❌ 同一个被试的数据既在训练集又在验证集
- ❌ 无法验证跨被试泛化能力（验证集和训练集来自相同的被试）
- ❌ 验证集性能不能反映目标域性能

---

## 实际例子

### 假设14个源被试（索引0-13）

**按被试划分**（val_ratio=0.2）：
```
Train subjects (11个): [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
  → 用于训练，每个被试一个domain loader

Val subjects (3个): [4, 12, 13]
  → 用于验证，3个被试的所有数据合并
  → 验证集样本数 ≈ 3被试 × 3396样本/被试 = 10188样本
```

**验证集是完整的被试**，可以真正测试跨被试泛化。

---

## 代码实现

### prepare_stage2_data()

```python
def prepare_stage2_data(source_data, source_labels, val_ratio=0.2):
    num_subjects = len(source_data)  # 14个源被试
    num_val_subjects = max(1, int(num_subjects * val_ratio))  # 至少1个，如2个

    # 随机选择验证被试
    subject_indices = np.random.permutation(num_subjects)
    train_subject_indices = subject_indices[:-num_val_subjects]  # 12个
    val_subject_indices = subject_indices[-num_val_subjects:]   # 2个

    print(f"Train subjects: {train_subject_indices}")
    print(f"Val subjects: {val_subject_indices}")

    # 训练集：每个训练被试一个loader
    domain_loaders = []
    for subj_idx in train_subject_indices:
        loader = create_loader(source_data[subj_idx], source_labels[subj_idx])
        domain_loaders.append(loader)

    # 验证集：合并所有验证被试
    val_data = np.concatenate([source_data[i] for i in val_subject_indices])
    val_labels = np.concatenate([source_labels[i] for i in val_subject_indices])
    val_loader = create_loader(val_data, val_labels)

    return domain_loaders, val_loader
```

---

## 输出示例

运行Stage 2时会看到：

```
=== Preparing Stage 2 Data ===
Stage 2 Subject split: 12 subjects for train, 2 subjects for val
  Train subjects: [ 0  1  2  3  4  5  6  7  9 10 11 13]
  Val subjects: [ 8 12]
Stage 2: 12 domain loaders for train, Val=6792 samples from 2 subjects

=== Stage 2 Training Start (Strict DG Setting) ===
Model selection: based on source validation set (not target domain)
```

---

## 训练过程

```
每个epoch：
  1. 从12个训练被试中采样4个被试
  2. 计算CE loss + Fishr loss
  3. 更新模型

  4. 在2个验证被试上评估
  5. 如果验证集准确率提升，保存模型

训练结束：
  1. 加载验证集上最好的模型
  2. 在目标被试（test_id）上测试一次 → 这才是真实泛化性能
```

---

## 数据划分总结

### Stage 1：混合所有源被试，按Trial划分
```
14个源被试 × 15个trial/被试 = 210个trials
  → 随机划分：168个trials训练，42个trials验证
  → 目的：学习情绪分类能力
```

### Stage 2：按被试划分
```
14个源被试
  → 随机划分：11-12个被试训练，2-3个被试验证
  → 目的：学习跨被试泛化能力
```

### 最终测试：目标被试
```
1个目标被试（从未见过）
  → 只在最后测试一次
  → 这才是真实的跨被试泛化性能
```

---

## 常见问题

### Q1: 为什么Stage 1按trial，Stage 2按被试？

**A**:
- **Stage 1目标**：学习情绪分类（不需要跨被试）
  - 验证集只要能评估分类能力即可，按trial划分增加数据多样性

- **Stage 2目标**：学习跨被试泛化
  - 验证集必须模拟目标域（全新被试）
  - 因此验证集必须是完整的被试

### Q2: 验证被试数量多少合适？

**A**:
- `val_ratio=0.2` → 14个被试中约2-3个验证
- 太少（1个）：不稳定，随机性大
- 太多（5个）：训练被试太少，影响学习
- **推荐**：2-3个验证被试（约20%）

### Q3: 每次运行验证被试都不同？

**A**:
- 是的，因为 `np.random.permutation` 每次随机
- 如果要固定，设置随机种子：
  ```python
  np.random.seed(2024)
  ```

---

## 总结

**关键点**：
- Stage 2验证集 = 完整的几个被试（不是trial混合）
- 这样才能真正测试跨被试泛化能力
- 验证集性能可以预测目标域性能

**修改后的优势**：
- ✅ 符合域泛化设定
- ✅ 验证集模拟目标域
- ✅ 结果更可靠
