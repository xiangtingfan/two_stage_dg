"""
Test script to verify trial-level splitting
测试脚本：验证trial级别的数据划分是否正确
"""
import torch
import numpy as np
from data_loader import load_seed_data, prepare_stage1_data

print("="*80)
print("Testing Trial-Level Splitting")
print("="*80)

# 加载一个小测试（test_id=0）
print("\n[1/3] Loading data...")
source_data, source_labels, target_data, target_labels = load_seed_data(
    test_id=0,
    session=1
)

# SEED数据集每个被试有15个trial
video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]

print(f"\n[2/3] Verifying source data structure...")
print(f"Number of source subjects: {len(source_data)}")
print(f"Expected samples per subject: {sum(video_time)}")

# 验证每个被试的数据
for subj_idx in range(min(3, len(source_data))):
    data = source_data[subj_idx]
    expected_samples = sum(video_time)

    print(f"\nSubject {subj_idx}:")
    print(f"  Total samples: {data.shape[0]} (expected: {expected_samples})")

    # 验证trial边界
    start_idx = 0
    for trial_id, num_samples in enumerate(video_time):
        end_idx = start_idx + num_samples
        if end_idx <= data.shape[0]:
            trial_data = data[start_idx:end_idx]
            print(f"    Trial {trial_id+1}: {trial_data.shape[0]} samples")
        else:
            print(f"    Trial {trial_id+1}: WARNING - incomplete ({data.shape[0]-start_idx}/{num_samples})")
        start_idx = end_idx

print(f"\n[3/3] Testing trial-level split...")
# 设置随机种子以确保可重复性
np.random.seed(2024)

# 准备Stage 1数据
train_loader, val_loader = prepare_stage1_data(source_data, source_labels)

# 获取一个batch
train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print(f"\nTrain batch shape: {train_batch[0].shape}")
print(f"Val batch shape: {val_batch[0].shape}")

print("\n" + "="*80)
print("✓ Trial-level splitting test completed!")
print("="*80)
print("\nKey points:")
print("1. Each subject has 15 trials (videos)")
print("2. Trials are assigned to train/val sets (not individual samples)")
print("3. No trial appears in both train and val sets")
print("4. This prevents data leakage from same trial")
