"""
Two-Stage Domain Generalization for EEG Emotion Recognition
主训练脚本：完整的LOSO实验
"""
import torch
import numpy as np
import os
from datetime import datetime

from data_loader import load_seed_data, prepare_stage2_data
from train_stage1 import train_stage1
from train_stage2 import train_stage2


def loso_experiment(config):
    """
    Leave-One-Subject-Out实验

    Args:
        config: 实验配置
    """
    # 设置随机种子
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # 存储所有结果
    all_results = []

    # LOSO循环：15个被试轮流作为测试集
    for test_id in range(15):
        print("\n" + "="*80)
        print(f"TEST SUBJECT: {test_id+1}/15")
        print("="*80)

        # 加载数据
        source_data, source_labels, target_data, target_labels = load_seed_data(
            test_id=test_id,
            session=config['session']
        )

        # ========== 划分源被试：训练被试 vs 验证被试 ==========
        # 先划分，确保Stage 1和Stage 2使用相同的验证被试
        print("\n" + "="*80)
        print("SUBJECT SPLIT")
        print("="*80)

        # 随机划分源被试为训练被试和验证被试
        np.random.seed(config['seed'])  # 固定随机种子，确保可重复
        _, _, val_subject_indices = prepare_stage2_data(
            source_data=source_data,
            source_labels=source_labels,
            val_ratio=0.2  # 20%的源被试作为验证集
        )
        print(f"\nValidation subjects indices (will be excluded from Stage 1): {val_subject_indices}")

        # ========== Stage 1: ERM预训练 ==========
        print("\n" + "="*80)
        print("STAGE 1: ERM Pre-training (using training subjects only)")
        print("="*80)

        model_stage1, best_val_acc = train_stage1(
            source_data=source_data,
            source_labels=source_labels,
            test_id=test_id,
            config=config,
            val_subject_indices=val_subject_indices  # 传递验证被试索引
        )

        # ========== Stage 2: DG微调 ==========
        print("\n" + "="*80)
        print("STAGE 2: DG Fine-tuning (using training subjects for train, val subjects for validation)")
        print("="*80)

        model_stage2, best_test_acc = train_stage2(
            source_data=source_data,
            source_labels=source_labels,
            target_data=target_data,
            target_labels=target_labels,
            test_id=test_id,
            config=config,
            val_subject_indices=val_subject_indices  # 传递验证被试索引
        )

        # 记录结果
        result = {
            'test_id': test_id + 1,
            'stage1_val_acc': best_val_acc,
            'stage2_test_acc': best_test_acc
        }
        all_results.append(result)

        print(f"\nSubject {test_id+1} Result: {best_test_acc:.2%}")

    # ========== 统计所有结果 ==========
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    # 提取准确率
    accuracies = [r['stage2_test_acc'] for r in all_results]

    # 统计
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    print(f"\nPer-subject accuracies:")
    for i, r in enumerate(all_results):
        print(f"  Subject {r['test_id']:2d}: {r['stage2_test_acc']:.2%}")

    print(f"\nAverage over 15 subjects:")
    print(f"  Mean: {mean_acc:.2%}")
    print(f"  Std:  {std_acc:.2%}")
    print(f"  Min:  {min_acc:.2%}")
    print(f"  Max:  {max_acc:.2%}")

    # 保存结果
    results_file = os.path.join(save_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write("Two-Stage Domain Generalization Results\n")
        f.write(f"Session: {config['session']}\n")
        f.write(f"Date: {timestamp}\n\n")

        f.write("Per-subject accuracies:\n")
        for r in all_results:
            f.write(f"  Subject {r['test_id']:2d}: {r['stage2_test_acc']:.2%}\n")

        f.write(f"\nAverage over 15 subjects:\n")
        f.write(f"  Mean: {mean_acc:.2%}\n")
        f.write(f"  Std:  {std_acc:.2%}\n")
        f.write(f"  Min:  {min_acc:.2%}\n")
        f.write(f"  Max:  {max_acc:.2%}\n")

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == '__main__':
    # 完整配置
    config = {
        # 数据
        'session': 1,  # 1, 2, or 3

        # 模型
        'hidden_dim': 128,
        'dropout': 0.5,

        # Stage 1
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs_stage1': 60,
        'patience': 15,

        # Stage 2
        'lr_ratio': 0.5,  # Stage 2学习率是Stage 1的0.5x
        'epochs_stage2': 50,
        'freeze_ratio': 0.7,  # 冻结70%的backbone
        'm': 4,  # 每次采样4个域
        'lambda_fishr_max': 0.3,  # Fishr最大权重
        'warmup_epochs': 20,  # Fishr权重warm-up

        # 其他
        'seed': 2024,
    }

    # 打印配置
    print("="*80)
    print("Two-Stage Domain Generalization for EEG Emotion Recognition")
    print("="*80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n" + "="*80)

    # 运行实验
    results = loso_experiment(config)

    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
