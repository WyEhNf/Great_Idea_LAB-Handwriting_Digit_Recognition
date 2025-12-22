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

# 双卷积层（卷积核适中）模型
class CNN2Layers10Epochs(nn.Module):
    """
    架构说明（与之前双卷积层适中相同）：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 64个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度3136，输出维度128，ReLU激活
    - 全连接层2: 输入维度128，输出维度10
    
    特殊：训练10个Epoch，每个Epoch详细报告
    """
    def __init__(self):
        super(CNN2Layers10Epochs, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出: 14x14x64
        # 经过第二个池化: 7x7x64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
def test_epoch(model, device, test_loader, criterion, epoch=None):
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

# 详细报告函数
def generate_detailed_report(epoch, train_loss, train_acc, test_loss, test_acc, test_size):
    """
    为每个epoch生成详细报告，包括置信区间
    """
    # 计算置信区间
    ci_lower, ci_upper = calculate_confidence_interval(test_acc, test_size)
    
    report = f"""
{'='*80}
Epoch {epoch} 详细报告:
{'='*80}
训练结果:
  - 训练损失: {train_loss:.6f}
  - 训练准确率: {train_acc:.4f}%

测试结果:
  - 测试损失: {test_loss:.6f}
  - 测试准确率: {test_acc:.4f}%
  - 95% 置信区间: [{ci_lower:.4f}%, {ci_upper:.4f}%]
  - 区间宽度: {ci_upper - ci_lower:.4f}%

准确率提升分析:
  - 与初始准确率相比: {test_acc:.4f}%
  - 与前一epoch相比的变化: N/A
{'='*80}
"""
    return report

# 训练和评估函数（10个Epoch）
def train_10_epochs_model():
    # 超参数设置
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"模型: 双卷积层（卷积核适中）")
    print(f"Epoch数: {num_epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 初始化模型
    model = CNN2Layers10Epochs().to(device)
    model_name = "双卷积层（卷积核适中）- 10 Epochs"
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 存储训练历史
    history = {
        'epochs': [],
        'train_losses': [],
        'train_accuracies': [],
        'test_losses': [],
        'test_accuracies': [],
        'confidence_intervals': [],
        'learning_rates': []
    }
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 训练循环（10个Epoch）
    for epoch in range(1, num_epochs + 1):
        print(f"\n开始 Epoch {epoch}/{num_epochs}")
        print("-"*60)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # 训练
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch, model_name)
        
        # 测试
        test_loss, test_acc, predictions, targets = test_epoch(model, device, test_loader, criterion, epoch)
        
        # 计算置信区间
        test_size = len(test_loader.dataset)
        ci_lower, ci_upper = calculate_confidence_interval(test_acc, test_size)
        
        # 更新学习率
        scheduler.step()
        
        # 存储结果
        history['epochs'].append(epoch)
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accuracies'].append(test_acc)
        history['confidence_intervals'].append((ci_lower, ci_upper))
        
        # 生成详细报告
        report = generate_detailed_report(
            epoch, train_loss, train_acc, test_loss, test_acc, test_size
        )
        print(report)
        
        # 如果是第5个或第10个epoch，保存中间模型
        if epoch == 5 or epoch == 10:
            torch.save(model.state_dict(), f'cnn_2layer_epoch_{epoch}.pth')
            print(f"✓ 模型已保存为 'cnn_2layer_epoch_{epoch}.pth'")
    
    # 记录结束时间
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    # 最终评估
    print("\n" + "="*80)
    print("最终评估结果")
    print("="*80)
    
    final_test_loss, final_test_acc, final_predictions, final_targets = test_epoch(
        model, device, test_loader, criterion
    )
    
    # 计算最终置信区间
    final_ci_lower, final_ci_upper = calculate_confidence_interval(final_test_acc, test_size)
    
    print(f"最终测试准确率: {final_test_acc:.4f}%")
    print(f"最终95%置信区间: [{final_ci_lower:.4f}%, {final_ci_upper:.4f}%]")
    print(f"训练总时间: {training_duration}")
    print(f"平均每个epoch时间: {training_duration/num_epochs}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'cnn_2layer_final.pth')
    print("✓ 最终模型已保存为 'cnn_2layer_final.pth'")
    
    return model, history, final_test_acc, (final_ci_lower, final_ci_upper)

# ================ 可视化函数 ================

def plot_training_results(history):
    """
    绘制训练结果图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = history['epochs']
    
    # 1. 训练和测试损失
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', linewidth=2, marker='o', label='训练损失')
    axes[0, 0].plot(epochs, history['test_losses'], 'r-', linewidth=2, marker='s', label='测试损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和测试损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 训练和测试准确率
    axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', linewidth=2, marker='o', label='训练准确率')
    axes[0, 1].plot(epochs, history['test_accuracies'], 'r-', linewidth=2, marker='s', label='测试准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].set_title('训练和测试准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 置信区间可视化
    ci_lowers = [ci[0] for ci in history['confidence_intervals']]
    ci_uppers = [ci[1] for ci in history['confidence_intervals']]
    axes[0, 2].fill_between(epochs, ci_lowers, ci_uppers, alpha=0.3, color='gray', label='95% 置信区间')
    axes[0, 2].plot(epochs, history['test_accuracies'], 'r-', linewidth=2, marker='s', label='测试准确率')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('准确率 (%)')
    axes[0, 2].set_title('测试准确率及其置信区间')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 准确率提升分析
    accuracy_improvement = np.diff([0] + history['test_accuracies'])
    axes[1, 0].bar(epochs, accuracy_improvement, color='skyblue', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率提升 (%)')
    axes[1, 0].set_title('每个Epoch的准确率提升')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. 置信区间宽度变化
    ci_widths = [upper - lower for lower, upper in history['confidence_intervals']]
    axes[1, 1].plot(epochs, ci_widths, 'g-', linewidth=2, marker='^')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('置信区间宽度 (%)')
    axes[1, 1].set_title('置信区间宽度变化')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 学习率变化
    axes[1, 2].plot(epochs, history['learning_rates'], 'purple-', linewidth=2, marker='d')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('学习率')
    axes[1, 2].set_title('学习率调度')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('双卷积层（卷积核适中）- 10 Epochs训练结果分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 创建总结表格
    print("\n" + "="*80)
    print("训练结果总结表格")
    print("="*80)
    print(f"{'Epoch':<6} {'Train Acc(%)':<12} {'Test Acc(%)':<12} {'CI Lower(%)':<12} {'CI Upper(%)':<12} {'CI Width(%)':<12}")
    print("-"*80)
    
    for i, epoch in enumerate(epochs):
        train_acc = history['train_accuracies'][i]
        test_acc = history['test_accuracies'][i]
        ci_lower, ci_upper = history['confidence_intervals'][i]
        ci_width = ci_upper - ci_lower
        
        print(f"{epoch:<6} {train_acc:<12.4f} {test_acc:<12.4f} {ci_lower:<12.4f} {ci_upper:<12.4f} {ci_width:<12.4f}")

# ================ 错误分析函数 ================

def analyze_errors(model, device, test_loader):
    """
    分析模型在测试集上的错误
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            # 找出预测错误的样本
            incorrect_mask = predicted != target
            if incorrect_mask.any():
                incorrect_data = data[incorrect_mask]
                incorrect_targets = target[incorrect_mask]
                incorrect_predictions = predicted[incorrect_mask]
                
                for i in range(min(5, len(incorrect_data))):
                    errors.append({
                        'data': incorrect_data[i].cpu(),
                        'true': incorrect_targets[i].item(),
                        'predicted': incorrect_predictions[i].item()
                    })
    
    return errors

# ================ 主函数 ================

def main():
    """
    主函数：训练10个Epoch的双卷积层模型
    """
    print("开始训练双卷积层（卷积核适中）模型，共10个Epoch")
    print("="*80)
    
    # 训练模型
    model, history, final_acc, final_ci = train_10_epochs_model()
    
    # 绘制结果
    plot_training_results(history)
    
    # 错误分析
    print("\n" + "="*80)
    print("错误分析")
    print("="*80)
    
    _, test_loader = get_data_loaders(batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    errors = analyze_errors(model, device, test_loader)
    
    if errors:
        print(f"找到 {len(errors)} 个错误样本（显示前5个）:")
        for i, error in enumerate(errors[:5]):
            print(f"  样本 {i+1}: 真实标签={error['true']}, 预测标签={error['predicted']}")
    else:
        print("没有找到错误样本（模型完美！）")
    
    # 最终总结
    print("\n" + "="*80)
    print("模型训练完成总结")
    print("="*80)
    print(f"模型架构: 双卷积层（卷积核适中）")
    print(f"最终测试准确率: {final_acc:.4f}%")
    print(f"最终95%置信区间: [{final_ci[0]:.4f}%, {final_ci[1]:.4f}%]")
    print(f"模型已保存: 'cnn_2layer_final.pth'")
    print("="*80)
    
    return model, history

# 运行主函数
if __name__ == "__main__":
    model, history = main()