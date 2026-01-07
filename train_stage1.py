"""
Stage 1: ERM Pre-training
目标：让模型学会"情绪分类"
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

from models import create_model, freeze_backbone
from data_loader import load_seed_data, prepare_stage1_data

# 获取项目根目录（本脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def train_stage1(source_data, source_labels, test_id, config, val_subject_indices=None):
    """
    Stage 1训练

    Args:
        source_data: list of [n_samples, 310]
        source_labels: list of [n_samples, 3]
        test_id: 测试被试ID
        config: 训练配置
        val_subject_indices: list of int - 验证被试的索引（从源被试中排除）

    Returns:
        model: 训练好的模型
        best_val_acc: 最佳验证准确率
    """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 准备数据
    print("\n=== Preparing Stage 1 Data ===")
    train_loader, val_loader = prepare_stage1_data(
        source_data=source_data,
        source_labels=source_labels,
        val_subject_indices=val_subject_indices  # 排除验证被试
    )

    # 创建模型
    print("\n=== Building Model ===")
    model_type = config.get('model_type', 'mlp')  # 默认使用MLP
    print(f"Model type: {model_type}")

    if model_type.lower() == 'mlp':
        model = create_model(
            model_type='mlp',
            input_dim=310,
            hidden_dim=config['hidden_dim'],
            num_classes=3,
            dropout=config['dropout']
        ).to(device)
    elif model_type.lower() == 'dgcnn':
        model = create_model(
            model_type='dgcnn',
            num_electrodes=62,
            in_channels=5,
            num_classes=3,
            k=config.get('k', 2),
            layers=config.get('layers', [64]),
            dropout=config['dropout']
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs_stage1'],
        eta_min=1e-5
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print("\n=== Stage 1 Training Start ===")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config['epochs_stage1']):
        # ========== 训练 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs_stage1']}")
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)

            # 前向传播
            logits, features = model(data)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            labels_idx = labels.argmax(dim=1)
            train_correct += (preds == labels_idx).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        # 计算训练准确率
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ========== 验证 ==========
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

        print(f"Epoch {epoch+1}/{config['epochs_stage1']}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2%}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2%}")

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # 保存最佳模型（使用项目根目录的绝对路径）
            save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'stage1_testid{test_id}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path} (Val Acc={val_acc:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n=== Stage 1 Complete ===")
    print(f"Best Val Acc: {best_val_acc:.2%}")

    return model, best_val_acc


if __name__ == '__main__':
    # 配置
    config = {
        'model_type': 'mlp',  # 'mlp' 或 'dgcnn'

        # MLP参数
        'hidden_dim': 128,
        'dropout': 0.5,

        # DGCNN参数（仅当model_type='dgcnn'时使用）
        'k': 2,
        'layers': [64],

        # 训练参数
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs_stage1': 60,
        'patience': 15,
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
    model, best_val_acc = train_stage1(
        source_data=source_data,
        source_labels=source_labels,
        test_id=test_id,
        config=config
    )

    print("\nDone!")
