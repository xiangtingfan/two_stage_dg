"""
Data Loader for Two-Stage Domain Generalization
基于load_data3.py改编，适配两阶段训练
"""
import os
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
import torch
import torch.utils.data as Data


def load_seed_data(test_id, session=1, data_path='H:/SEED/feature_for_net_session{}_LDS_de'):
    """
    加载SEED数据集

    Args:
        test_id: 测试被试ID (0-14)
        session: session编号 (1, 2, 3)
        data_path: 数据路径模板

    Returns:
        source_data: list of [n_samples, 310] - 14个源被试的数据
        source_labels: list of [n_samples, 3] - 14个源被试的标签
        target_data: [n_samples, 310] - 测试被试的数据
        target_labels: [n_samples, 3] - 测试被试的标签
    """
    path = data_path.format(session)
    # 移除 os.chdir(path)，避免改变工作目录

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]

    source_data = []
    source_labels = []
    target_data = None
    target_labels = None

    index = 0
    for info in os.listdir(path):
        # 过滤：只处理.mat文件
        if not info.endswith('.mat'):
            continue

        # 使用绝对路径
        info_ = os.path.join(path, info)

        # 加载数据
        if session == 1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        elif session == 2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]

        # 归一化（按被试）
        feature = min_max_scaler.fit_transform(feature).astype('float32')

        # 标签转one-hot
        one_hot_label_mat = np.zeros((len(label), 3))
        for i in range(len(label)):
            if label[i] == -1:  # negative
                one_hot_label_mat[i, :] = [1, 0, 0]
            elif label[i] == 0:  # neutral
                one_hot_label_mat[i, :] = [0, 1, 0]
            elif label[i] == 1:  # positive
                one_hot_label_mat[i, :] = [0, 0, 1]

        # 分割源域和目标域
        if index != test_id:
            # 源域
            source_data.append(feature)
            source_labels.append(one_hot_label_mat)
        else:
            # 目标域
            target_data = feature
            target_labels = one_hot_label_mat

        index += 1

    print(f"Loaded {len(source_data)} source subjects, 1 target subject (test_id={test_id})")
    print(f"Source samples: {[d.shape[0] for d in source_data]}")
    print(f"Target samples: {target_data.shape[0]}")

    return source_data, source_labels, target_data, target_labels


def prepare_stage1_data(source_data, source_labels):
    """
    Stage 1数据准备：混合所有源被试

    重要：按trial级别划分，确保同一个trial的数据不会同时出现在训练集和验证集

    Returns:
        train_loader: DataLoader (混合所有源被试)
        val_loader: DataLoader (从源被试中划分20%作为验证集)
    """
    # SEED数据集每个被试有15个trial (video)
    video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]

    # 将每个被试的数据按trial划分
    all_trials_data = []
    all_trials_labels = []

    for subj_idx, (data, labels) in enumerate(zip(source_data, source_labels)):
        start_idx = 0
        for trial_id, num_samples in enumerate(video_time):
            end_idx = start_idx + num_samples

            # 提取当前trial的数据和标签
            trial_data = data[start_idx:end_idx]
            trial_labels = labels[start_idx:end_idx]

            # 记录trial来源信息
            all_trials_data.append(trial_data)
            all_trials_labels.append(trial_labels)

            start_idx = end_idx

    print(f"Total trials from {len(source_data)} subjects: {len(all_trials_data)}")

    # 随机打乱trials
    num_trials = len(all_trials_data)
    trial_indices = np.random.permutation(num_trials)

    # 划分训练集和验证集 (80%-20%的trials)
    split_idx = int(num_trials * 0.8)
    train_trial_indices = trial_indices[:split_idx]
    val_trial_indices = trial_indices[split_idx:]

    print(f"Trial split: {len(train_trial_indices)} trials for train, {len(val_trial_indices)} trials for val")

    # 合并训练trials
    train_data_list = [all_trials_data[i] for i in train_trial_indices]
    train_labels_list = [all_trials_labels[i] for i in train_trial_indices]
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    # 合并验证trials
    val_data_list = [all_trials_data[i] for i in val_trial_indices]
    val_labels_list = [all_trials_labels[i] for i in val_trial_indices]
    val_data = np.concatenate(val_data_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)

    # 在sample级别shuffle（train set）
    train_indices = np.random.permutation(len(train_data))
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]

    # 转为Tensor
    train_dataset = Data.TensorDataset(
        torch.from_numpy(train_data),
        torch.from_numpy(train_labels)
    )
    val_dataset = Data.TensorDataset(
        torch.from_numpy(val_data),
        torch.from_numpy(val_labels)
    )

    # DataLoader
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print(f"Stage 1: Train={len(train_data)} samples, Val={len(val_data)} samples")

    return train_loader, val_loader


def prepare_stage2_data(source_data, source_labels, val_ratio=0.2):
    """
    Stage 2数据准备：按被试组织（每个被试独立）
    同时划分验证集（按被试划分，而不是trial）

    重要：验证集是完整的被试，这样才能真正测试跨被试泛化

    Args:
        source_data: list of [n_samples, 310]
        source_labels: list of [n_samples, 3]
        val_ratio: 验证集比例（用于选择验证被试数量）

    Returns:
        domain_loaders: list of DataLoader - 每个训练被试一个loader
        val_loader: DataLoader - 验证被试的合并数据
    """
    num_subjects = len(source_data)
    num_val_subjects = max(1, int(num_subjects * val_ratio))  # 至少保留1个被试验证
    num_train_subjects = num_subjects - num_val_subjects

    # 随机选择哪些被试作为验证集
    subject_indices = np.random.permutation(num_subjects)
    train_subject_indices = subject_indices[:num_train_subjects]
    val_subject_indices = subject_indices[num_train_subjects:]

    print(f"Stage 2 Subject split: {len(train_subject_indices)} subjects for train, {len(val_subject_indices)} subjects for val")
    print(f"  Train subjects: {train_subject_indices}")
    print(f"  Val subjects: {val_subject_indices}")

    # 构建验证集（合并所有验证被试的数据）
    val_data_list = [source_data[i] for i in val_subject_indices]
    val_labels_list = [source_labels[i] for i in val_subject_indices]
    val_data = np.concatenate(val_data_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)

    # 构建验证集loader
    val_dataset = Data.TensorDataset(
        torch.from_numpy(val_data),
        torch.from_numpy(val_labels)
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 构建训练集的domain loaders（每个训练被试一个loader）
    domain_loaders = []
    for subj_idx in train_subject_indices:
        data = source_data[subj_idx]
        labels = source_labels[subj_idx]

        dataset = Data.TensorDataset(
            torch.from_numpy(data),
            torch.from_numpy(labels)
        )
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=16,  # 每个被试batch小一点
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        domain_loaders.append(loader)

    print(f"Stage 2: {len(domain_loaders)} domain loaders for train, Val={len(val_data)} samples from {len(val_subject_indices)} subjects")

    return domain_loaders, val_loader


def sample_multi_domain_batch(domain_loaders, m=4):
    """
    Stage 2训练时采样m个域的batch

    Args:
        domain_loaders: list of DataLoader
        m: 采样域数量

    Returns:
        batches: list of (data, labels, domain_id)
    """
    # 随机选择m个域
    num_domains = len(domain_loaders)
    selected_domains = np.random.choice(num_domains, size=m, replace=False)

    batches = []
    for domain_id in selected_domains:
        data, labels = next(iter(domain_loaders[domain_id]))
        batches.append((data, labels, domain_id))

    return batches
