"""
Stage 2: DG Fine-tuning with Fishr
目标：学习跨被试泛化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

from model import EEG_Encoder, freeze_backbone
from fishr import FishrLoss
from data_loader import load_seed_data, prepare_stage2_data, sample_multi_domain_batch

# 获取项目根目录（本脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def loso_evaluate(model, target_data, target_labels, device):
    """
    LOSO评估

    Args:
        model: 模型
        target_data: [n_samples, 310]
        target_labels: [n_samples, 3]
        device: 设备

    Returns:
        accuracy: 准确率
    """
    model.eval()

    data = torch.from_numpy(target_data).float().to(device)
    labels = torch.from_numpy(target_labels).float().to(device)

    with torch.no_grad():
        logits, features = model(data)
        preds = logits.argmax(dim=1)
        labels_idx = labels.argmax(dim=1)
        correct = (preds == labels_idx).sum().item()
        total = labels.size(0)

    accuracy = correct / total
    return accuracy


def train_stage2(source_data, source_labels, target_data, target_labels,
                 test_id, config, val_subject_indices=None):
    """
    Stage 2训练

    Args:
        source_data: list of [n_samples, 310]
        source_labels: list of [n_samples, 3]
        target_data: [n_samples, 310]
        target_labels: [n_samples, 3]
        test_id: 测试被试ID
        config: 训练配置
        val_subject_indices: list of int - 验证被试的索引（如果提供，则使用这些被试验证）

    Returns:
        model: 训练好的模型
        best_test_acc: 最佳测试准确率
    """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 准备数据（按域）
    print("\n=== Preparing Stage 2 Data ===")
    domain_loaders, val_loader, _ = prepare_stage2_data(
        source_data=source_data,
        source_labels=source_labels,
        val_ratio=0.2,
        val_subject_indices=val_subject_indices  # 使用main.py中划分的验证被试
    )

    # 创建模型
    print("\n=== Building Model ===")
    model = EEG_Encoder(
        input_dim=310,
        hidden_dim=config['hidden_dim'],
        num_classes=3,
        dropout=config['dropout']
    ).to(device)

    # 加载Stage 1权重（使用项目根目录的绝对路径）
    stage1_path = os.path.join(PROJECT_ROOT, f"checkpoints/stage1_testid{test_id}_best.pth")
    if os.path.exists(stage1_path):
        model.load_state_dict(torch.load(stage1_path), strict=True)
        print(f"Loaded Stage 1 weights from {stage1_path}")
    else:
        print(f"Warning: Stage 1 weights not found at {stage1_path}")
        print("Training from scratch...")

    # 冻结backbone
    model = freeze_backbone(model, freeze_ratio=config['freeze_ratio'])

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # 更小的学习率
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'] * config['lr_ratio'],  # 0.5x
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs_stage2'],
        eta_min=1e-6
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    fishr_loss_fn = FishrLoss()

    # 训练循环
    print("\n=== Stage 2 Training Start (Strict DG Setting) ===")
    print("Model selection: based on source validation set (not target domain)")
    best_val_acc = 0
    best_model_state = None

    # 初始评估（Stage 1的性能）
    initial_test_acc = loso_evaluate(model, target_data, target_labels, device)
    print(f"Initial Test Acc (Stage 1): {initial_test_acc:.2%}")

    for epoch in range(config['epochs_stage2']):
        model.train()

        # Fishr权重warm-up
        lambda_fishr = min(config['lambda_fishr_max'],
                          config['lambda_fishr_max'] * epoch / config['warmup_epochs'])

        train_loss = 0.0
        train_ce_loss = 0.0
        train_fishr_loss = 0.0
        train_correct = 0
        train_total = 0

        # 每个epoch采样多次
        num_iterations = 100

        for iter_idx in tqdm(range(num_iterations), desc=f"Epoch {epoch+1}/{config['epochs_stage2']}"):
            # 1. 按域采样batch
            domain_batches = sample_multi_domain_batch(domain_loaders, m=config['m'])

            # 2. 展平所有样本
            all_data = []
            all_labels = []
            for data, labels, _ in domain_batches:
                all_data.append(data)
                all_labels.append(labels)
            all_data = torch.cat(all_data, dim=0).to(device)
            all_labels = torch.cat(all_labels, dim=0).to(device)

            # 3. 前向传播
            logits, features = model(all_data)
            ce_loss = criterion(logits, all_labels)

            # 4. 计算Fishr损失
            fishr_loss = fishr_loss_fn(model, domain_batches, criterion)

            # 5. 总损失
            total_loss = ce_loss + lambda_fishr * fishr_loss

            # 6. 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 统计
            train_loss += total_loss.item()
            train_ce_loss += ce_loss.item()
            train_fishr_loss += fishr_loss.item()
            preds = logits.argmax(dim=1)
            labels_idx = all_labels.argmax(dim=1)
            train_correct += (preds == labels_idx).sum().item()
            train_total += all_labels.size(0)

        scheduler.step()

        # 计算训练准确率
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / num_iterations
        avg_ce_loss = train_ce_loss / num_iterations
        avg_fishr_loss = train_fishr_loss / num_iterations

        # ========== 在源域验证集上评估（用于模型选择）==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)

                logits, features = model(data)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                labels_idx = labels.argmax(dim=1)
                val_correct += (preds == labels_idx).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs_stage2']}: "
              f"Train Loss={avg_train_loss:.4f}, CE={avg_ce_loss:.4f}, Fishr={avg_fishr_loss:.4f}, "
              f"Train Acc={train_acc:.2%}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2%}, λ={lambda_fishr:.3f}")

        # 保存最佳模型（基于验证集）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'stage2_testid{test_id}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path} (Val Acc={val_acc:.2%})")

    # ========== 加载最佳模型，在目标域上测试（仅一次）==========
    print(f"\n=== Stage 2 Complete ===")
    print(f"Loading best model (Val Acc={best_val_acc:.2%})...")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_test_acc = loso_evaluate(model, target_data, target_labels, device)

    print(f"\n=== Final Results (Strict DG Setting) ===")
    print(f"Stage 1 Test Acc: {initial_test_acc:.2%}")
    print(f"Stage 2 Best Val Acc: {best_val_acc:.2%}")
    print(f"Stage 2 Final Test Acc: {final_test_acc:.2%}")
    print(f"Improvement: {final_test_acc - initial_test_acc:.2%}")

    return model, final_test_acc


if __name__ == '__main__':
    # 配置
    config = {
        'hidden_dim': 128,
        'dropout': 0.5,
        'lr': 1e-3,
        'lr_ratio': 0.5,  # Stage 2学习率是Stage 1的0.5x
        'weight_decay': 1e-4,
        'epochs_stage2': 50,  # 修改为epochs_stage2以匹配main.py
        'freeze_ratio': 0.7,  # 冻结70%的backbone
        'm': 4,  # 每次采样4个域
        'lambda_fishr_max': 0.3,  # Fishr最大权重
        'warmup_epochs': 20,  # Fishr权重warm-up
        'session': 1,
    }

    # 固定随机种子
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

    # 测试一个被试
    test_id = 0

    # 加载数据
    print("=== Loading Data ===")
    source_data, source_labels, target_data, target_labels = load_seed_data(
        test_id=test_id,
        session=config['session']
    )

    # 训练
    model, best_test_acc = train_stage2(
        source_data=source_data,
        source_labels=source_labels,
        target_data=target_data,
        target_labels=target_labels,
        test_id=test_id,
        config=config
    )

    print("\nDone!")
