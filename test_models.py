"""
测试新的模型接口
"""
import torch
print("Testing model imports...")

try:
    from models import EEG_MLP, EEG_DGCNN, create_model, freeze_backbone
    print("✓ Imports successful")

    # 测试MLP
    print("\nTesting MLP...")
    model_mlp = create_model('mlp', hidden_dim=128, dropout=0.5)
    x = torch.randn(4, 310)
    logits, features = model_mlp(x)
    print(f"  MLP Input: {x.shape}")
    print(f"  MLP Features: {features.shape}")
    print(f"  MLP Logits: {logits.shape}")
    print("  ✓ MLP works!")

    # 测试DGCNN
    print("\nTesting DGCNN...")
    model_dgcnn = create_model('dgcnn', k=2, layers=[64], dropout=0.5)
    logits_dgcnn, features_dgcnn = model_dgcnn(x)
    print(f"  DGCNN Input: {x.shape}")
    print(f"  DGCNN Features: {features_dgcnn.shape}")
    print(f"  DGCNN Logits: {logits_dgcnn.shape}")
    print("  ✓ DGCNN works!")

    # 测试冻结
    print("\nTesting freeze...")
    model_frozen = freeze_backbone(model_mlp, freeze_ratio=0.5)
    total = sum(p.numel() for p in model_frozen.parameters())
    trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"  MLP params: {total}")
    print(f"  MLP trainable: {trainable} ({100*trainable/total:.1f}%)")
    print("  ✓ Freeze works!")

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
