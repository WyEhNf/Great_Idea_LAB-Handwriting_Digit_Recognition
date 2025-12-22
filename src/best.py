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
from datetime import datetime

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
    # 数据预处理
    transform = transforms.Compose([
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
        transform=transform
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

# ================ 模型定义 ================

# ResNet Basic Block
class ResNetBasicBlock(nn.Module):
    """
    ResNet基础残差块
    包含两个3x3卷积层和shortcut连接
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out

# 三卷积层 + ResNet Shortcut 模型
class CNN3LayersResNet(nn.Module):
    """
    架构说明：三卷积层 + ResNet Shortcut
    结合了三层卷积的特征提取能力和ResNet的残差连接优势
    
    网络结构：
    1. 初始卷积层：32个3x3卷积核，ReLU，批归一化
    2. 最大池化：2x2，步长2
    
    3. ResNet块1：32->32通道，包含残差连接
    4. 最大池化：2x2，步长2
    
    5. ResNet块2：32->64通道，包含残差连接
    6. 最大池化：2x2，步长2
    
    7. ResNet块3：64->128通道，包含残差连接
    8. 全局平均池化
    
    9. 全连接层：128->10
    
    总参数：约200K，比传统三层CNN略多，但训练更稳定
    """
    def __init__(self):
        super(CNN3LayersResNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        # ResNet块1 (32->32)
        self.resblock1 = self._make_resblock(32, 32)
        
        # ResNet块2 (32->64)，需要下采样
        self.resblock2 = self._make_resblock(32, 64, stride=1)
        
        # ResNet块3 (64->128)
        self.resblock3 = self._make_resblock(64, 128, stride=1)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(128, 10)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_resblock(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        return ResNetBasicBlock(in_channels, out_channels, stride, downsample)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 14x14x32
        
        # ResNet块1
        x = self.resblock1(x)
        x = self.pool(x)  # 7x7x32
        
        # ResNet块2
        x = self.resblock2(x)
        x = self.pool(x)  # 3x3x64 (由于是7x7输入，池化后向下取整)
        
        # ResNet块3
        x = self.resblock3(x)
        
        # 全局平均池化
        x = self.global_pool(x)  # 1x1x128
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

# 传统的三卷积层CNN（用于对比）
class CNN3LayersStandard(nn.Module):
    """
    传统三卷积层CNN，用于与ResNet版本对比
    没有残差连接
    """
    def __init__(self):
        super(CNN3LayersStandard, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14x32
        x = self.pool(F.relu(self.conv2(x)))  # 7x7x64
        x = F.relu(self.conv3(x))  # 7x7x128
        x = self.global_pool(x)  # 1x1x128
        x = x.view(-1, 128)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ================ 训练和评估函数 ================

# 训练函数
def train_epoch(model, device, train_loader, optimizer, criterion, epoch, model_name):
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
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    train_accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    return avg_loss, train_accuracy

# 测试函数
def test_epoch(model, device, test_loader, criterion):
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

# 训练和比较两个模型
def train_and_compare_models():
    # 超参数设置
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"Epoch数: {num_epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 创建两个模型
    model_resnet = CNN3LayersResNet().to(device)
    model_standard = CNN3LayersStandard().to(device)
    
    # 计算参数量
    resnet_params = sum(p.numel() for p in model_resnet.parameters())
    standard_params = sum(p.numel() for p in model_standard.parameters())
    
    print("模型参数对比:")
    print(f"  ResNet版本: {resnet_params:,} 参数")
    print(f"  标准版本: {standard_params:,} 参数")
    print(f"  参数比: {resnet_params/standard_params:.2f}倍")
    print("="*80)
    
    # 定义损失函数和优化器（两个模型使用相同的超参数）
    criterion = nn.CrossEntropyLoss()
    
    optimizer_resnet = optim.Adam(model_resnet.parameters(), lr=learning_rate)
    optimizer_standard = optim.Adam(model_standard.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler_resnet = optim.lr_scheduler.StepLR(optimizer_resnet, step_size=3, gamma=0.5)
    scheduler_standard = optim.lr_scheduler.StepLR(optimizer_standard, step_size=3, gamma=0.5)
    
    # 存储训练历史
    history_resnet = {
        'epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': [],
        'confidence_intervals': []
    }
    
    history_standard = {
        'epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': [],
        'confidence_intervals': []
    }
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\n开始 Epoch {epoch}/{num_epochs}")
        print("-"*60)
        
        # ========== 训练ResNet模型 ==========
        print("训练 ResNet 模型...")
        train_loss_resnet, train_acc_resnet = train_epoch(
            model_resnet, device, train_loader, optimizer_resnet, 
            criterion, epoch, "ResNet-3Layers"
        )
        
        # ========== 训练标准模型 ==========
        print("训练 标准CNN 模型...")
        train_loss_standard, train_acc_standard = train_epoch(
            model_standard, device, train_loader, optimizer_standard,
            criterion, epoch, "Standard-3Layers"
        )
        
        # ========== 测试两个模型 ==========
        test_loss_resnet, test_acc_resnet, _, _ = test_epoch(
            model_resnet, device, test_loader, criterion
        )
        
        test_loss_standard, test_acc_standard, _, _ = test_epoch(
            model_standard, device, test_loader, criterion
        )
        
        # 更新学习率
        scheduler_resnet.step()
        scheduler_standard.step()
        
        # 计算置信区间
        test_size = len(test_loader.dataset)
        ci_resnet = calculate_confidence_interval(test_acc_resnet, test_size)
        ci_standard = calculate_confidence_interval(test_acc_standard, test_size)
        
        # 存储结果
        for history, train_loss, train_acc, test_loss, test_acc, ci in zip(
            [history_resnet, history_standard],
            [train_loss_resnet, train_loss_standard],
            [train_acc_resnet, train_acc_standard],
            [test_loss_resnet, test_loss_standard],
            [test_acc_resnet, test_acc_standard],
            [ci_resnet, ci_standard]
        ):
            history['epochs'].append(epoch)
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)
            history['test_losses'].append(test_loss)
            history['test_accuracies'].append(test_acc)
            history['confidence_intervals'].append(ci)
        
        # 打印当前epoch结果
        print(f"\nEpoch {epoch} 结果对比:")
        print(f"{'模型':<20} {'训练Acc':<10} {'测试Acc':<10} {'置信区间(95%)':<20}")
        print(f"{'-'*60}")
        print(f"{'ResNet-3Layers':<20} {train_acc_resnet:>9.2f}% {test_acc_resnet:>9.2f}% [{ci_resnet[0]:.2f}%, {ci_resnet[1]:.2f}%]")
        print(f"{'Standard-3Layers':<20} {train_acc_standard:>9.2f}% {test_acc_standard:>9.2f}% [{ci_standard[0]:.2f}%, {ci_standard[1]:.2f}%]")
        
        # 计算优势
        advantage = test_acc_resnet - test_acc_standard
        print(f"ResNet优势: {advantage:.2f}%")
    
    # 记录结束时间
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    # 最终评估
    print("\n" + "="*80)
    print("最终评估结果")
    print("="*80)
    
    # 最终测试
    final_test_loss_resnet, final_test_acc_resnet, _, _ = test_epoch(
        model_resnet, device, test_loader, criterion
    )
    
    final_test_loss_standard, final_test_acc_standard, _, _ = test_epoch(
        model_standard, device, test_loader, criterion
    )
    
    # 计算最终置信区间
    final_ci_resnet = calculate_confidence_interval(final_test_acc_resnet, test_size)
    final_ci_standard = calculate_confidence_interval(final_test_acc_standard, test_size)
    
    print(f"{'模型':<20} {'测试准确率':<15} {'置信区间(95%)':<20} {'训练时间':<10}")
    print(f"{'-'*80}")
    print(f"{'ResNet-3Layers':<20} {final_test_acc_resnet:>14.2f}% [{final_ci_resnet[0]:.2f}%, {final_ci_resnet[1]:.2f}%] {str(training_duration)[:10]:>10}")
    print(f"{'Standard-3Layers':<20} {final_test_acc_standard:>14.2f}% [{final_ci_standard[0]:.2f}%, {final_ci_standard[1]:.2f}%] {str(training_duration)[:10]:>10}")
    
    final_advantage = final_test_acc_resnet - final_test_acc_standard
    print(f"\n最终ResNet优势: {final_advantage:.2f}%")
    
    # 保存模型
    torch.save(model_resnet.state_dict(), 'cnn_3layers_resnet_final.pth')
    torch.save(model_standard.state_dict(), 'cnn_3layers_standard_final.pth')
    print("✓ 模型已保存")
    
    return model_resnet, model_standard, history_resnet, history_standard

# ================ 可视化函数 ================

def plot_comparison_results(history_resnet, history_standard):
    """
    绘制两个模型的对比结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = history_resnet['epochs']
    
    # 1. 测试准确率对比
    axes[0, 0].plot(epochs, history_resnet['test_accuracies'], 'b-', linewidth=2, marker='o', label='ResNet-3Layers')
    axes[0, 0].plot(epochs, history_standard['test_accuracies'], 'r-', linewidth=2, marker='s', label='Standard-3Layers')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('测试准确率 (%)')
    axes[0, 0].set_title('测试准确率对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 训练准确率对比
    axes[0, 1].plot(epochs, history_resnet['train_accuracies'], 'b-', linewidth=2, marker='o', label='ResNet-3Layers')
    axes[0, 1].plot(epochs, history_standard['train_accuracies'], 'r-', linewidth=2, marker='s', label='Standard-3Layers')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('训练准确率 (%)')
    axes[0, 1].set_title('训练准确率对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 训练损失对比
    axes[0, 2].plot(epochs, history_resnet['train_losses'], 'b-', linewidth=2, marker='o', label='ResNet-3Layers')
    axes[0, 2].plot(epochs, history_standard['train_losses'], 'r-', linewidth=2, marker='s', label='Standard-3Layers')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('训练损失')
    axes[0, 2].set_title('训练损失对比')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 准确率优势分析
    advantage = np.array(history_resnet['test_accuracies']) - np.array(history_standard['test_accuracies'])
    axes[1, 0].plot(epochs, advantage, 'g-', linewidth=2, marker='^')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(epochs, advantage, 0, where=advantage>=0, alpha=0.3, color='green', label='ResNet优势')
    axes[1, 0].fill_between(epochs, advantage, 0, where=advantage<0, alpha=0.3, color='red', label='标准CNN优势')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率差 (%)')
    axes[1, 0].set_title('ResNet vs 标准CNN准确率差异')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 置信区间宽度对比
    ci_width_resnet = [upper - lower for lower, upper in history_resnet['confidence_intervals']]
    ci_width_standard = [upper - lower for lower, upper in history_standard['confidence_intervals']]
    axes[1, 1].plot(epochs, ci_width_resnet, 'b-', linewidth=2, marker='o', label='ResNet-3Layers')
    axes[1, 1].plot(epochs, ci_width_standard, 'r-', linewidth=2, marker='s', label='Standard-3Layers')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('置信区间宽度 (%)')
    axes[1, 1].set_title('置信区间宽度对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 最终epoch的置信区间对比
    axes[1, 2].barh([0, 1], 
                    [history_resnet['test_accuracies'][-1], history_standard['test_accuracies'][-1]],
                    height=0.4, color=['blue', 'red'], alpha=0.7)
    
    # 添加误差条（置信区间）
    resnet_lower, resnet_upper = history_resnet['confidence_intervals'][-1]
    standard_lower, standard_upper = history_standard['confidence_intervals'][-1]
    
    axes[1, 2].errorbar([history_resnet['test_accuracies'][-1], history_standard['test_accuracies'][-1]], 
                       [0, 1], 
                       xerr=[[history_resnet['test_accuracies'][-1] - resnet_lower, 
                              history_standard['test_accuracies'][-1] - standard_lower],
                            [resnet_upper - history_resnet['test_accuracies'][-1], 
                             standard_upper - history_standard['test_accuracies'][-1]]],
                       fmt='none', ecolor='black', capsize=5)
    
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_yticklabels(['ResNet-3Layers', 'Standard-3Layers'])
    axes[1, 2].set_xlabel('测试准确率 (%)')
    axes[1, 2].set_title('最终准确率对比 (带置信区间)')
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('三卷积层CNN: ResNet vs 标准架构对比分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 创建总结表格
    print("\n" + "="*80)
    print("训练结果详细对比")
    print("="*80)
    print(f"{'Epoch':<6} {'ResNet Acc(%)':<15} {'Std Acc(%)':<15} {'优势(%)':<12} {'ResNet CI':<20} {'Std CI':<20}")
    print("-"*80)
    
    for i, epoch in enumerate(epochs):
        resnet_acc = history_resnet['test_accuracies'][i]
        std_acc = history_standard['test_accuracies'][i]
        advantage = resnet_acc - std_acc
        resnet_ci_lower, resnet_ci_upper = history_resnet['confidence_intervals'][i]
        std_ci_lower, std_ci_upper = history_standard['confidence_intervals'][i]
        
        print(f"{epoch:<6} {resnet_acc:<15.4f} {std_acc:<15.4f} {advantage:<12.4f} "
              f"[{resnet_ci_lower:.2f}%,{resnet_ci_upper:.2f}%] [{std_ci_lower:.2f}%,{std_ci_upper:.2f}%]")

# ================ ResNet优势分析 ================

def analyze_resnet_advantages(model_resnet, model_standard, device, test_loader):
    """
    分析ResNet模型相对于标准模型的优势
    """
    print("\n" + "="*80)
    print("ResNet优势详细分析")
    print("="*80)
    
    # 获取两个模型的预测
    model_resnet.eval()
    model_standard.eval()
    
    resnet_correct = 0
    standard_correct = 0
    both_correct = 0
    resnet_only_correct = 0
    standard_only_correct = 0
    both_wrong = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output_resnet = model_resnet(data)
            output_standard = model_standard(data)
            
            _, predicted_resnet = output_resnet.max(1)
            _, predicted_standard = output_standard.max(1)
            
            for i in range(len(target)):
                total += 1
                resnet_is_correct = predicted_resnet[i] == target[i]
                standard_is_correct = predicted_standard[i] == target[i]
                
                if resnet_is_correct:
                    resnet_correct += 1
                if standard_is_correct:
                    standard_correct += 1
                if resnet_is_correct and standard_is_correct:
                    both_correct += 1
                elif resnet_is_correct and not standard_is_correct:
                    resnet_only_correct += 1
                elif not resnet_is_correct and standard_is_correct:
                    standard_only_correct += 1
                else:
                    both_wrong += 1
    
    print(f"总测试样本数: {total}")
    print(f"ResNet正确数: {resnet_correct} ({100.*resnet_correct/total:.2f}%)")
    print(f"标准CNN正确数: {standard_correct} ({100.*standard_correct/total:.2f}%)")
    print(f"两者都正确: {both_correct} ({100.*both_correct/total:.2f}%)")
    print(f"只有ResNet正确: {resnet_only_correct} ({100.*resnet_only_correct/total:.2f}%)")
    print(f"只有标准CNN正确: {standard_only_correct} ({100.*standard_only_correct/total:.2f}%)")
    print(f"两者都错误: {both_wrong} ({100.*both_wrong/total:.2f}%)")
    
    # 计算ResNet的净优势
    net_advantage = resnet_only_correct - standard_only_correct
    print(f"ResNet净优势样本数: {net_advantage}")
    print(f"ResNet净优势比例: {100.*net_advantage/total:.2f}%")
    
    # 绘制Venn图样式的分析图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制圆形
    circle_resnet = plt.Circle((0.3, 0.5), 0.4, color='blue', alpha=0.3, label='ResNet正确')
    circle_standard = plt.Circle((0.7, 0.5), 0.4, color='red', alpha=0.3, label='标准CNN正确')
    
    ax.add_artist(circle_resnet)
    ax.add_artist(circle_standard)
    
    # 添加文本
    plt.text(0.3, 0.5, f'只有ResNet正确\n{resnet_only_correct}\n({100.*resnet_only_correct/total:.1f}%)', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(0.7, 0.5, f'只有标准CNN正确\n{standard_only_correct}\n({100.*standard_only_correct/total:.1f}%)', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(0.5, 0.5, f'两者都正确\n{both_correct}\n({100.*both_correct/total:.1f}%)', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(0.5, 0.1, f'两者都错误\n{both_wrong}\n({100.*both_wrong/total:.1f}%)', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title('ResNet vs 标准CNN正确预测分析', fontsize=14, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout()
    plt.show()

# ================ 主函数 ================

def main():
    """
    主函数：训练和比较ResNet与标准三卷积层CNN
    """
    print("开始训练和比较: 三卷积层 + ResNet Shortcut vs 标准三卷积层CNN")
    print("="*80)
    
    # 训练和比较两个模型
    model_resnet, model_standard, history_resnet, history_standard = train_and_compare_models()
    
    # 绘制对比结果
    plot_comparison_results(history_resnet, history_standard)
    
    # 详细分析ResNet优势
    _, test_loader = get_data_loaders(batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_resnet_advantages(model_resnet, model_standard, device, test_loader)
    
    # 最终总结
    print("\n" + "="*80)
    print("实验总结: 三卷积层 + ResNet Shortcut 的强大之处")
    print("="*80)
    print("1. 架构优势:")
    print("   - ResNet shortcut允许梯度直接流过网络，缓解梯度消失")
    print("   - 残差连接使得网络可以学习恒等映射，训练更稳定")
    print("   - 可以构建更深的网络而不会出现退化问题")
    print()
    print("2. 性能表现:")
    print(f"   - ResNet版本最终准确率: {history_resnet['test_accuracies'][-1]:.2f}%")
    print(f"   - 标准版本最终准确率: {history_standard['test_accuracies'][-1]:.2f}%")
    print(f"   - ResNet优势: {history_resnet['test_accuracies'][-1] - history_standard['test_accuracies'][-1]:.2f}%")
    print()
    print("3. 训练稳定性:")
    print("   - ResNet通常有更平滑的训练曲线")
    print("   - 更容易收敛到更好的局部最优解")
    print("   - 对超参数的选择相对更鲁棒")
    print()
    print("4. 实际应用建议:")
    print("   - 对于复杂任务，ResNet架构通常更优")
    print("   - 对于简单任务(如MNIST)，标准CNN可能足够")
    print("   - ResNet在深度增加时优势更明显")
    print("="*80)
    
    return model_resnet, model_standard, history_resnet, history_standard

# 运行主函数
if __name__ == "__main__":
    model_resnet, model_standard, history_resnet, history_standard = main()