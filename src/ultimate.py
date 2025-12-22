import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

# 定义数据预处理和加载
def get_data_loaders(batch_size=128):
    # 增强数据预处理（添加随机旋转和小幅度平移）
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 测试集不需要数据增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ================ 终极版本模型定义 ================

class ResidualBlock(nn.Module):
    """ResNet风格的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)  # 空间dropout
        
        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class UltimateCNNResNet(nn.Module):
    """
    终极版本CNN模型架构说明：
    
    1. 三层CNN架构：
       - 初始卷积层：32个3x3卷积核，填充1，BatchNorm，ReLU激活
       - 最大池化：2x2窗口，步长2
       
    2. 残差块层（使用ResNet shortcut）：
       - 残差块1：输入32通道，输出64通道，步长1（保持尺寸）
       - 残差块2：输入64通道，输出128通道，步长2（下采样）
       - 残差块3：输入128通道，输出256通道，步长2（下采样）
    
    3. Dropout策略：
       - 空间Dropout：在残差块中使用Dropout2d
       - 标准Dropout：在全连接层前使用Dropout
    
    4. 卷积核参数选择：
       - 使用3x3卷积核（标准大小）
       - 逐步增加通道数：32→64→128→256
       - 使用BatchNorm加速训练并提高稳定性
    
    5. 最终分类层：
       - 全局平均池化
       - 两层全连接层，中间有Dropout
       - 输出10个类别
    """
    def __init__(self, dropout_rate=0.3):
        super(UltimateCNNResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 残差块层
        self.res_block1 = ResidualBlock(32, 64, stride=1, dropout_rate=dropout_rate)
        self.res_block2 = ResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate)
        self.res_block3 = ResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate)
        
        # 自适应平均池化，适应不同尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Kaiming初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 输出: 14x14x32
        
        # 残差块
        x = self.res_block1(x)  # 输出: 14x14x64
        x = self.res_block2(x)  # 输出: 7x7x128
        x = self.res_block3(x)  # 输出: 4x4x256
        
        # 全局平均池化
        x = self.adaptive_pool(x)  # 输出: 1x1x256
        x = x.view(x.size(0), -1)  # 展平
        
        # 全连接层
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ================ 训练和评估函数 ================

# 学习率调度器
def get_lr_scheduler(optimizer, total_epochs):
    """余弦退火学习率调度器"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6
    )

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch, model_name, lr_scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'{model_name} - Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{train_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # 更新学习率
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    train_accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    return avg_loss, train_accuracy

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, test_accuracy, all_predictions, all_targets

# 计算置信区间
def calculate_confidence_interval(accuracy, n_samples, confidence_level=0.95):
    """
    计算准确率的置信区间
    """
    # 使用Wilson score区间，适用于二项分布比例
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    p = accuracy / 100.0
    n = n_samples
    
    # Wilson score区间公式
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    lower_bound = (centre - half_width) * 100
    upper_bound = (centre + half_width) * 100
    
    return lower_bound, upper_bound

# 计算混淆矩阵
def calculate_confusion_matrix(predictions, targets, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, target in zip(predictions, targets):
        cm[target][pred] += 1
    return cm

# ================ 主训练循环 ================
def main():
    # 超参数设置
    batch_size = 128
    num_epochs = 15  # 增加epoch数以获得更好性能
    learning_rate = 0.001
    dropout_rate = 0.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("终极版本CNN模型训练")
    print("="*70)
    print(f"使用设备: {device}")
    print(f"Epoch数: {num_epochs}")
    print(f"Batch大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"Dropout率: {dropout_rate}")
    print("="*70)
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 初始化模型
    model = UltimateCNNResNet(dropout_rate=dropout_rate).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用AdamW
    lr_scheduler = get_lr_scheduler(optimizer, num_epochs)
    
    # 训练历史记录
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    learning_rates = []
    
    print("\n开始训练终极版本CNN模型...")
    print("="*70)
    
    best_accuracy = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train(
            model, device, train_loader, optimizer, criterion, epoch, 
            "终极CNN-ResNet", lr_scheduler
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 测试
        test_loss, test_acc, predictions, targets = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ 新的最佳准确率: {best_accuracy:.2f}%")
        
        print("-"*50)
    
    # 使用最佳模型进行最终评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("加载最佳模型进行最终评估...")
    
    # 最终测试
    test_loss, final_test_acc, predictions, targets = test(model, device, test_loader, criterion)
    
    # 计算置信区间
    test_size = len(test_loader.dataset)
    lower_bound, upper_bound = calculate_confidence_interval(final_test_acc, test_size)
    
    # 计算混淆矩阵
    confusion_matrix = calculate_confusion_matrix(predictions, targets)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 打印最终结果
    print("\n" + "="*70)
    print("终极版本CNN模型最终结果")
    print("="*70)
    print(f"最终测试准确率: {final_test_acc:.2f}%")
    print(f"95% 置信区间: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("="*70)
    
    # 打印模型架构详情
    print("\n模型架构详情:")
    print("-"*50)
    print("1. 三层CNN架构:")
    print("   - 初始卷积层: 32个3×3卷积核, BatchNorm, ReLU激活")
    print("   - 最大池化: 2×2窗口，步长2")
    print("\n2. 三个残差块（带ResNet shortcut）:")
    print("   - 残差块1: 32→64通道，保持尺寸")
    print("   - 残差块2: 64→128通道，步长2（下采样）")
    print("   - 残差块3: 128→256通道，步长2（下采样）")
    print("\n3. Dropout策略:")
    print("   - 空间Dropout2d: 在残差块中使用")
    print("   - 标准Dropout: 在全连接层前使用")
    print("\n4. 其他优化:")
    print("   - BatchNorm: 所有卷积层后使用")
    print("   - 自适应平均池化: 适应不同特征图尺寸")
    print("   - 梯度裁剪: 防止梯度爆炸")
    print("   - 余弦退火学习率调度")
    print("="*70)
    
    # 绘制训练曲线
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    ax1.plot(range(1, num_epochs + 1), train_losses, label='训练损失', marker='o', linewidth=2)
    ax1.plot(range(1, num_epochs + 1), test_losses, label='测试损失', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和测试损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率', marker='o', linewidth=2)
    ax2.plot(range(1, num_epochs + 1), test_accuracies, label='测试准确率', marker='s', linewidth=2)
    ax2.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7, label=f'最佳准确率: {best_accuracy:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练和测试准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax3.plot(range(1, num_epochs + 1), learning_rates, marker='o', linewidth=2, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('学习率')
    ax3.set_title('学习率变化（余弦退火）')
    ax3.grid(True, alpha=0.3)
    
    # 混淆矩阵热力图
    im = ax4.imshow(confusion_matrix, cmap='Blues')
    ax4.set_xlabel('预测标签')
    ax4.set_ylabel('真实标签')
    ax4.set_title('混淆矩阵热力图')
    
    # 添加数值到热力图
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax4.text(j, i, str(confusion_matrix[i, j]), 
                    ha='center', va='center', color='black' if confusion_matrix[i, j] < np.max(confusion_matrix)/2 else 'white')
    
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    plt.show()
    
    # 打印每类准确率
    print("\n每类准确率分析:")
    print("-"*50)
    for i in range(10):
        total_samples = np.sum(confusion_matrix[i, :])
        correct_samples = confusion_matrix[i, i]
        class_accuracy = 100.0 * correct_samples / total_samples if total_samples > 0 else 0
        print(f"数字 {i}: {class_accuracy:.2f}% ({correct_samples}/{total_samples})")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_accuracy,
        'total_params': total_params,
        'confusion_matrix': confusion_matrix,
        'final_test_acc': final_test_acc,
        'confidence_interval': (lower_bound, upper_bound)
    }, 'ultimate_cnn_resnet_mnist.pth')
    
    print(f"\n模型已保存为 'ultimate_cnn_resnet_mnist.pth'")
    
    return {
        'model': model,
        'final_accuracy': final_test_acc,
        'confidence_interval': (lower_bound, upper_bound),
        'best_accuracy': best_accuracy,
        'total_params': total_params,
        'confusion_matrix': confusion_matrix
    }

# 运行主函数
if __name__ == "__main__":
    results = main()