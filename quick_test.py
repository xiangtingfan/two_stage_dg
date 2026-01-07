"""
Quick Test Script
快速测试所有模块是否能正常工作
"""
import torch
import numpy as np
import os

print("="*80)
print("Two-Stage DG Framework - Quick Test")
print("="*80)

# ========== 1. 测试数据加载 ==========
print("\n[1/5] Testing data loader...")
try:
    from data_loader import load_seed_data, prepare_stage1_data, prepare_stage2_data

    # 加载一个小测试（test_id=0）
    source_data, source_labels, target_data, target_labels = load_seed_data(
        test_id=0,
        session=1
    )

    print(f"✓ Loaded {len(source_data)} source subjects")
    print(f"✓ Source data shapes: {[d.shape for d in source_data[:3]]}...")

    # 测试Stage 1数据准备
    train_loader, val_loader = prepare_stage1_data(source_data, source_labels, val_subject_indices=None)
    print(f"✓ Stage 1: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    # 测试Stage 2数据准备
    domain_loaders, val_loader, val_subject_indices = prepare_stage2_data(source_data, source_labels)
    print(f"✓ Stage 2: {len(domain_loaders)} domain loaders, Val batches={len(val_loader)}")
    print(f"  Validation subjects: {val_subject_indices}")

except Exception as e:
    print(f"✗ Data loader failed: {e}")
    exit(1)

# ========== 2. 测试模型 ==========
print("\n[2/5] Testing model...")
try:
    from model import EEG_Encoder, freeze_backbone

    model = EEG_Encoder()
    x = torch.randn(16, 310)
    logits, features = model(x)

    print(f"✓ Model forward pass successful")
    print(f"  Input: {x.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Logits: {logits.shape}")

    # 测试预测
    preds = model.predict(x)
    print(f"✓ Predictions: {preds.shape}")

    # 测试冻结
    model_frozen = freeze_backbone(model, freeze_ratio=0.5)
    total_params = sum(p.numel() for p in model_frozen.parameters())
    trainable_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"✓ Freeze: {trainable_params}/{total_params} params trainable ({100*trainable_params/total_params:.1f}%)")

except Exception as e:
    print(f"✗ Model test failed: {e}")
    exit(1)

# ========== 3. 测试Fishr损失 ==========
print("\n[3/5] Testing Fishr loss...")
try:
    from fishr import FishrLoss

    fishr_loss_fn = FishrLoss()
    criterion = torch.nn.CrossEntropyLoss()

    # 模拟4个域的batch
    domain_batches = []
    for i in range(4):
        data = torch.randn(8, 310)
        labels = torch.randint(0, 3, (8, 3)).float()
        domain_batches.append((data, labels, i))

    fishr_loss = fishr_loss_fn(model, domain_batches, criterion)
    print(f"✓ Fishr loss computed: {fishr_loss.item():.4f}")

except Exception as e:
    print(f"✗ Fishr loss test failed: {e}")
    exit(1)

# ========== 4. 测试训练流程（单步）==========
print("\n[4/5] Testing training step...")
try:
    from data_loader import sample_multi_domain_batch

    # Stage 1单步测试
    model = EEG_Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    data, labels = next(iter(train_loader))
    logits, features = model(data)
    loss = criterion(logits, labels.argmax(dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Stage 1 training step successful: loss={loss.item():.4f}")

    # Stage 2单步测试
    domain_batches = sample_multi_domain_batch(domain_loaders, m=4)
    all_data = torch.cat([b[0] for b in domain_batches], dim=0)
    all_labels = torch.cat([b[1] for b in domain_batches], dim=0)

    logits, features = model(all_data)
    ce_loss = criterion(logits, all_labels.argmax(dim=1))
    fishr_loss = fishr_loss_fn(model, domain_batches, criterion)
    total_loss = ce_loss + 0.1 * fishr_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 测试验证集评估
    model.eval()
    val_data, val_labels = next(iter(val_loader))
    val_logits, _ = model(val_data)
    val_preds = val_logits.argmax(dim=1)
    val_labels_idx = val_labels.argmax(dim=1)
    val_acc = (val_preds == val_labels_idx).float().mean().item()

    print(f"✓ Stage 2 training step successful:")
    print(f"  CE loss: {ce_loss.item():.4f}")
    print(f"  Fishr loss: {fishr_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Val acc: {val_acc:.2%}")

except Exception as e:
    print(f"✗ Training step test failed: {e}")
    exit(1)

# ========== 5. 测试GPU兼容性 ==========
print("\n[5/5] Testing GPU compatibility...")
if torch.cuda.is_available():
    try:
        device = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

        # 测试GPU上的模型
        model = EEG_Encoder().to(device)
        x = torch.randn(16, 310).to(device)
        logits, features = model(x)
        print(f"✓ Model on GPU: input={x.shape}, output={logits.shape}")

    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("⚠ CUDA not available, will use CPU")

# ========== 总结 ==========
print("\n" + "="*80)
print("All tests passed! ✓")
print("="*80)
print("\nNext steps:")
print("1. Run Stage 1: python train_stage1.py")
print("2. Run Stage 2: python train_stage2.py")
print("3. Run full LOSO: python main.py")
print("\nOr read README.md for more details.")
print("="*80)
