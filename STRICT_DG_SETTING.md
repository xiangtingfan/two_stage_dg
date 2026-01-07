# 方案A：严格域泛化设定

## 修改说明

将Stage 2从"在目标域上选择最佳模型"修改为"在源域验证集上选择最佳模型"，符合严格的域泛化设定。

---

## 修改前后对比

### 修改前（方案B - Oracle Setting）❌

```python
# 每个epoch都在目标域上评估
for epoch in range(epochs):
    train_model()
    test_acc = evaluate_on_target_domain(model)  # 在目标域上测试

    if test_acc > best_test_acc:  # 选择目标域上最好的模型
        save_checkpoint()

# 问题：这相当于在目标域上调参，会高估性能
```

**缺点**：
- ❌ 违反域泛化假设（训练时看不到目标域）
- ❌ 高估模型泛化性能
- ❌ 实际应用中无法复现（不知道哪个epoch最好）

---

### 修改后（方案A - Strict DG Setting）✅

```python
# 从源被试划分验证集
domain_loaders, val_loader = prepare_stage2_data(source_data, source_labels)

for epoch in range(epochs):
    train_model()
    val_acc = evaluate_on_source_validation(model)  # 在源域验证集上评估

    if val_acc > best_val_acc:  # 选择验证集上最好的模型
        save_checkpoint()

# 只在最后测试一次目标域
best_model = load_best_model()
final_test_acc = evaluate_on_target_domain(best_model)  # 仅测试一次
```

**优点**：
- ✅ 符合域泛化假设（训练时只看源域）
- ✅ 真实反映泛化性能
- ✅ 实际应用中可以复现

---

## 关键修改

### 1. data_loader.py - prepare_stage2_data()

**修改前**：
```python
def prepare_stage2_data(source_data, source_labels):
    domain_loaders = [...]
    return domain_loaders
```

**修改后**：
```python
def prepare_stage2_data(source_data, source_labels, val_ratio=0.2):
    # 按trial划分，80%训练，20%验证
    domain_loaders = [...]  # 训练集的domain loaders
    val_loader = [...]      # 源域验证集
    return domain_loaders, val_loader
```

**数据划分逻辑**：
- 14个源被试 × 15个trial/被试 = 210个trials
- 随机划分：168个trials (80%) → 训练集，42个trials (20%) → 验证集
- 确保trial级别不泄露

### 2. train_stage2.py - 训练循环

**修改前**：
```python
best_test_acc = 0

for epoch in range(epochs):
    train()
    test_acc = loso_evaluate(model, target_data, target_labels)  # 目标域

    if test_acc > best_test_acc:  # 在目标域上选
        best_test_acc = test_acc
        save_model()

return model, best_test_acc  # 返回目标域上的最好结果
```

**修改后**：
```python
best_val_acc = 0
best_model_state = None

for epoch in range(epochs):
    train()
    val_acc = evaluate_on_source_validation(model)  # 源域验证集

    if val_acc > best_val_acc:  # 在源域验证集上选
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        save_model()

# 加载最佳模型，只在目标域上测试一次
model.load_state_dict(best_model_state)
final_test_acc = loso_evaluate(model, target_data, target_labels)

return model, final_test_acc  # 返回目标域上的真实泛化性能
```

### 3. 输出信息变化

**修改前**：
```
Epoch 1/50: Loss=0.7234, CE=0.6123, Fishr=0.0369, Train Acc=78.45%, Test Acc=79.12%, λ=0.000
  → Saved best model (Test Acc=79.12%)

Stage 2 Best Test Acc: 84.56%
```

**修改后**：
```
Stage 2 Training Start (Strict DG Setting)
Model selection: based on source validation set (not target domain)

Epoch 1/50: Train Loss=0.7234, CE=0.6123, Fishr=0.0369, Train Acc=78.45%, Val Loss=0.8234, Val Acc=75.32%, λ=0.000
  → Saved best model (Val Acc=75.32%)

Loading best model (Val Acc=77.89%)...

=== Final Results (Strict DG Setting) ===
Stage 1 Test Acc: 72.34%
Stage 2 Best Val Acc: 77.89%
Stage 2 Final Test Acc: 76.12%  ← 真实泛化性能
Improvement: 3.78%
```

---

## 对两种方案的理解

### 方案A（Strict DG）- 当前实现
- **场景**：真实的跨被试应用
- **假设**：训练时看不到目标被试
- **评估**：在源域验证集选模型 → 目标域上测试一次
- **优点**：结果真实可靠
- **适用**：论文发表、实际部署

### 方案B（Oracle Setting）- 修改前
- **场景**：理论上限研究
- **假设**：可以看到目标被试来调参
- **评估**：在目标域上选最好的epoch
- **缺点**：高估性能，实际无法达到
- **适用**：作为参考上限，但不应作为主要结果

---

## 实验结果预期

修改后，Stage 2的提升可能会**变小**，这是因为：

**修改前（Oracle）**：
```
Stage 1 Test Acc: 72%
Stage 2 Test Acc (best epoch): 80%  ← 在50个epoch中选最好的
Improvement: +8%
```

**修改后（Strict DG）**：
```
Stage 1 Test Acc: 72%
Stage 2 Test Acc (selected by val): 75%  ← 验证集选的模型在目标域上
Improvement: +3%
```

**注意**：
- 虽然提升变小了，但这才是**真实的**泛化能力
- 方案B的80%是虚高的，实际应用中达不到
- 论文中应该使用方案A的结果

---

## 修改的文件

1. ✅ `data_loader.py` - `prepare_stage2_data()` 返回验证集
2. ✅ `train_stage2.py` - 使用验证集选择模型
3. ✅ `quick_test.py` - 适配新接口

---

## 验证方法

### 运行Stage 2训练：
```bash
cd E:\code\20240804SDA_DDA_Pairwise2\two_stage_dg
python train_stage2.py
```

观察输出：
- 应该看到 "Strict DG Setting"
- 每个epoch显示 "Val Acc" 而不是 "Test Acc"
- 最后显示 "Final Test Acc"（一次性测试）

---

## FAQ

### Q1: 为什么验证集准确率比测试集低？
**A**: 正常情况！验证集从源被试划分，测试集是目标被试。
- Val Acc: 源被试的一部分
- Test Acc: 全新的被试
- 测试集可能比验证集简单或复杂，都是可能的

### Q2: 可以用早停吗？
**A**: 可以！在 `train_stage2.py` 中添加早停逻辑（基于验证集）：
```python
if val_acc > best_val_acc:
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 20:
        break
```

### Q3: Stage 2的验证集和Stage 1的验证集是一样的吗？
**A**: 不一定！由于随机shuffle不同，两次划分的trials可能不同。
- 如果要完全一致，需要设置相同的随机种子
- 但通常不需要一致，它们是独立的实验

---

## 总结

这次修改将框架从"Oracle Setting"改为"Strict DG Setting"，更符合域泛化的真实场景，结果更加可靠和公平。
