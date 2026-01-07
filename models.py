"""
Unified Model Interface for Two-Stage DG
支持多种模型架构：MLP, DGCNN等
"""
import torch
import torch.nn as nn


class EEG_MLP(nn.Module):
    """
    MLP模型（原有架构）
    输入: [batch, 310] 展平的DE特征
    输出: [batch, 3] logits + [batch, 128] features
    """
    def __init__(self, input_dim=310, hidden_dim=128, num_classes=3, dropout=0.5):
        super(EEG_MLP, self).__init__()

        # Backbone: 特征提取器
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier: 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 310] 展平的DE特征

        Returns:
            logits: [batch, 3]
            features: [batch, 128]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class GraphConv(nn.Module):
    """
    基于Chebyshev多项式的图卷积
    """
    def __init__(self, k, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(self, x, lap):
        """计算Chebyshev多项式"""
        batch, ele_channel, in_channel = x.shape
        t = torch.ones(batch, ele_channel, in_channel).to(x.device)

        if self.k == 1:
            return t.unsqueeze(1)
        if self.k == 2:
            return torch.cat((t.unsqueeze(1), torch.matmul(lap, x).unsqueeze(1)), dim=1)
        elif self.k > 2:
            tk_minus_one = x
            tk = torch.matmul(lap, x)
            t = torch.cat((t.unsqueeze(1), tk_minus_one.unsqueeze(1), tk.unsqueeze(1)), dim=1)

            for i in range(3, self.k):
                tk_minus_two, tk_minus_one = tk_minus_one, tk
                tk = 2 * torch.matmul(lap, tk_minus_one) - tk_minus_two
                t = torch.cat((t, tk.unsqueeze(1)), dim=1)
            return t

    def forward(self, x, lap):
        """
        Args:
            x: [batch, num_electrodes, in_channels]
            lap: 拉普拉斯矩阵 [num_electrodes, num_electrodes]

        Returns:
            out: [batch, num_electrodes, out_channels]
        """
        # 计算Chebyshev多项式
        cp = self.chebyshev_polynomial(x, lap)  # [batch, k, num_electrodes, in_channels]
        cp = cp.permute(0, 2, 3, 1)  # [batch, num_electrodes, in_channels, k]
        cp = cp.flatten(start_dim=2)  # [batch, num_electrodes, in_channels * k]

        # 滤波操作
        out = torch.matmul(cp, self.weight)  # [batch, num_electrodes, out_channels]
        return out


class B1ReLU(nn.Module):
    """带偏置的ReLU激活函数"""
    def __init__(self, bias_shape):
        super(B1ReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.relu(self.bias + x)


def laplacian(adj):
    """
    计算邻接矩阵的拉普拉斯矩阵

    Args:
        adj: [num_electrodes, num_electrodes] 邻接矩阵

    Returns:
        lap: [num_electrodes, num_electrodes] 归一化拉普拉斯矩阵
    """
    # 计算度矩阵
    d = torch.sum(adj, dim=1)
    d_re = 1 / torch.sqrt(d + 1e-5)
    d_matrix = torch.diag_embed(d_re)

    # 计算拉普拉斯矩阵: L = I - D^(-1/2) * A * D^(-1/2)
    lap = torch.eye(d_matrix.shape[0], device=adj.device) - torch.matmul(torch.matmul(d_matrix, adj), d_matrix)
    return lap


class EEG_DGCNN(nn.Module):
    """
    DGCNN模型
    输入: [batch, 62, 5] DE特征（62个电极，5个频段）
    输出: [batch, 3] logits + [batch, num_electrodes*layers[-1]] features
    """
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3,
                 k=2, layers=None, dropout=0.5):
        """
        Args:
            num_electrodes: 电极数量（62）
            in_channels: 每个电极的特征维度（5个频段）
            num_classes: 类别数（3）
            k: Chebyshev多项式的阶数
            layers: 每层GCN的输出通道数
            dropout: Dropout比率
        """
        super(EEG_DGCNN, self).__init__()

        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.k = k

        # 设置默认层数
        if layers is None:
            if num_electrodes == 62:
                layers = [64]
            elif num_electrodes == 32:
                layers = [128]
            else:
                layers = [64]

        self.layers = layers

        # 图卷积层
        self.graph_convs = nn.ModuleList()
        self.graph_convs.append(GraphConv(k, in_channels, layers[0]))
        for i in range(len(layers) - 1):
            self.graph_convs.append(GraphConv(k, layers[i], layers[i + 1]))

        # B1ReLU激活函数
        self.b_relus = nn.ModuleList()
        for i in range(len(layers)):
            self.b_relus.append(B1ReLU(layers[i]))

        # 可学习的邻接矩阵
        self.adj = nn.Parameter(torch.Tensor(num_electrodes, num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))

        # 分类器
        self.fc = nn.Linear(num_electrodes * layers[-1], 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Args:
            x: [batch, 62, 5] 或者 [batch, 310]

        Returns:
            logits: [batch, 3]
            features: [batch, num_electrodes * layers[-1]]
        """
        # 如果输入是展平的[batch, 310]，reshape为[batch, 62, 5]
        if x.dim() == 2 and x.size(1) == 310:
            x = x.view(x.size(0), 62, 5)

        # 计算邻接矩阵和拉普拉斯矩阵
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)

        # 图卷积层
        for i in range(len(self.layers)):
            x = self.graph_convs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)

        # 展平特征
        features = x.reshape(x.shape[0], -1)
        x = self.dropout(features)

        # 分类
        x = self.fc(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, features


def create_model(model_type='mlp', **kwargs):
    """
    创建模型的工厂函数

    Args:
        model_type: 'mlp' 或 'dgcnn'
        **kwargs: 模型特定参数

    Returns:
        model: 对应的模型实例
    """
    if model_type.lower() == 'mlp':
        model = EEG_MLP(
            input_dim=kwargs.get('input_dim', 310),
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_classes=kwargs.get('num_classes', 3),
            dropout=kwargs.get('dropout', 0.5)
        )
    elif model_type.lower() == 'dgcnn':
        model = EEG_DGCNN(
            num_electrodes=kwargs.get('num_electrodes', 62),
            in_channels=kwargs.get('in_channels', 5),
            num_classes=kwargs.get('num_classes', 3),
            k=kwargs.get('k', 2),
            layers=kwargs.get('layers', [64]),
            dropout=kwargs.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'mlp' or 'dgcnn'.")

    return model


def freeze_backbone(model, freeze_ratio=0.7):
    """
    冻结backbone的前部分层

    Args:
        model: EEG模型
        freeze_ratio: 冻结比例 (0-1)

    Returns:
        model: 冻结后的模型
    """
    if isinstance(model, EEG_MLP):
        # MLP冻结
        backbone_layers = list(model.backbone.children())
        num_layers = len(backbone_layers)
        num_freeze = int(num_layers * freeze_ratio)

        for i in range(num_freeze):
            for param in backbone_layers[i].parameters():
                param.requires_grad = False

        for i in range(num_freeze, num_layers):
            for param in backbone_layers[i].parameters():
                param.requires_grad = True

        # Classifier完全可训练
        for param in model.classifier.parameters():
            param.requires_grad = True

        print(f"Frozen {num_freeze}/{num_layers} layers of MLP backbone")

    elif isinstance(model, EEG_DGCNN):
        # DGCNN Stage 2 冻结策略：
        # 冻结：adj, adj_bias, fc
        # 训练：GraphConv.weight, B1ReLU.bias, fc2

        print("DGCNN Stage 2 Freeze Strategy:")
        print("  Frozen: adj, adj_bias, fc (classifier layer 1)")
        print("  Trainable: GraphConv.weight, B1ReLU.bias, fc2 (classifier layer 2)")

        # 1. 冻结邻接矩阵和偏置
        model.adj.requires_grad = False
        model.adj_bias.requires_grad = False
        print(f"  ✓ Frozen adjacency matrix (adj: {model.adj.numel():,} params)")
        print(f"  ✓ Frozen adjacency bias (adj_bias: {model.adj_bias.numel():,} params)")

        # 2. 图卷积层的权重可训练
        for conv in model.graph_convs:
            conv.weight.requires_grad = True  # GraphConv的权重
        graph_conv_params = sum(p.numel() for conv in model.graph_convs for p in conv.parameters() if p.requires_grad)
        print(f"  ✓ Trainable GraphConv weights: {graph_conv_params:,} params")

        # 3. B1ReLU的偏置可训练
        for b_relu in model.b_relus:
            b_relu.bias.requires_grad = True  # B1ReLU的偏置
        b_relu_params = sum(p.numel() for b_relu in model.b_relus for p in b_relu.parameters() if p.requires_grad)
        print(f"  ✓ Trainable B1ReLU biases: {b_relu_params:,} params")

        # 4. 冻结分类器第一层（fc）
        for param in model.fc.parameters():
            param.requires_grad = False
        print(f"  ✓ Frozen fc (3968→256): {sum(p.numel() for p in model.fc.parameters()):,} params")

        # 5. 分类器第二层可训练（fc2）
        for param in model.fc2.parameters():
            param.requires_grad = True
        print(f"  ✓ Trainable fc2 (256→3): {sum(p.numel() for p in model.fc2.parameters()):,} params")

    # 统计总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal: {total_params:,} params, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    return model


if __name__ == '__main__':
    # 测试MLP
    print("="*80)
    print("Testing EEG_MLP")
    print("="*80)
    model_mlp = EEG_MLP()
    x_mlp = torch.randn(32, 310)
    logits_mlp, features_mlp = model_mlp(x_mlp)
    print(f"MLP Input: {x_mlp.shape}")
    print(f"MLP Features: {features_mlp.shape}")
    print(f"MLP Logits: {logits_mlp.shape}")

    # 测试DGCNN
    print("\n" + "="*80)
    print("Testing EEG_DGCNN")
    print("="*80)
    model_dgcnn = EEG_DGCNN()

    # 测试展平输入
    x_flat = torch.randn(32, 310)
    logits_dgcnn1, features_dgcnn1 = model_dgcnn(x_flat)
    print(f"DGCNN Input (flat): {x_flat.shape}")
    print(f"DGCNN Features: {features_dgcnn1.shape}")
    print(f"DGCNN Logits: {logits_dgcnn1.shape}")

    # 测试reshape输入
    x_reshaped = torch.randn(32, 62, 5)
    logits_dgcnn2, features_dgcnn2 = model_dgcnn(x_reshaped)
    print(f"DGCNN Input (reshaped): {x_reshaped.shape}")
    print(f"DGCNN Features: {features_dgcnn2.shape}")
    print(f"DGCNN Logits: {logits_dgcnn2.shape}")

    # 测试工厂函数
    print("\n" + "="*80)
    print("Testing create_model factory")
    print("="*80)
    model1 = create_model('mlp', hidden_dim=128, dropout=0.5)
    model2 = create_model('dgcnn', k=2, layers=[64], dropout=0.5)
    print(f"Created MLP model: {type(model1).__name__}")
    print(f"Created DGCNN model: {type(model2).__name__}")

    # 测试冻结
    print("\n" + "="*80)
    print("Testing freeze_backbone")
    print("="*80)
    model_mlp_frozen = freeze_backbone(model_mlp, freeze_ratio=0.5)
    model_dgcnn_frozen = freeze_backbone(model_dgcnn, freeze_ratio=0.5)

    total_mlp = sum(p.numel() for p in model_mlp_frozen.parameters())
    trainable_mlp = sum(p.numel() for p in model_mlp_frozen.parameters() if p.requires_grad)
    print(f"MLP Total params: {total_mlp}")
    print(f"MLP Trainable params: {trainable_mlp} ({100*trainable_mlp/total_mlp:.1f}%)")

    total_dgcnn = sum(p.numel() for p in model_dgcnn_frozen.parameters())
    trainable_dgcnn = sum(p.numel() for p in model_dgcnn_frozen.parameters() if p.requires_grad)
    print(f"DGCNN Total params: {total_dgcnn}")
    print(f"DGCNN Trainable params: {trainable_dgcnn} ({100*trainable_dgcnn/total_dgcnn:.1f}%)")

    print("\nAll tests passed!")
