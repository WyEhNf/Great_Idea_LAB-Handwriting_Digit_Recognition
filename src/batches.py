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

# 1. 双卷积层（卷积核较少）
class CNN2LayersFew(nn.Module):
    """
    架构说明：
    - 卷积层1: 8个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 16个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度256，输出维度64，ReLU激活
    - 全连接层2: 输入维度64，输出维度10
    """
    def __init__(self):
        super(CNN2LayersFew, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 输出: 28x28x8
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 输出: 14x14x16
        # 经过第二个池化: 7x7x16 = 784
        self.fc1 = nn.Linear(7 * 7 * 16, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 16)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 双卷积层（卷积核适中）
class CNN2LayersMedium(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 64个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度3136，输出维度128，ReLU激活
    - 全连接层2: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN2LayersMedium, self).__init__()
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

# 3. 双卷积层（卷积核较多）
class CNN2LayersMany(nn.Module):
    """
    架构说明：
    - 卷积层1: 64个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 128个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度6272，输出维度256，ReLU激活
    - 全连接层2: 输入维度256，输出维度10
    """
    def __init__(self):
        super(CNN2LayersMany, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 输出: 28x28x64
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出: 14x14x128
        # 经过第二个池化: 7x7x128 = 6272
        self.fc1 = nn.Linear(7 * 7 * 128, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. 单卷积层
class CNN1Layer(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度6272，输出维度128，ReLU激活
    - 全连接层2: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN1Layer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x32 = 6272
        self.fc1 = nn.Linear(14 * 14 * 32, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 14 * 14 * 32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. 三卷积层
class CNN3Layers(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 64个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层3: 128个3x3卷积核，填充1，ReLU激活
    - 全局平均池化: 输出1x1x128
    - 全连接层: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN3Layers, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出: 14x14x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出: 7x7x128
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

# 6. 池化模式改为AveragePooling，双卷积层，卷积核适中
class CNN2LayersAvgPool(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 平均池化: 2x2窗口，步长2
    - 卷积层2: 64个3x3卷积核，填充1，ReLU激活
    - 平均池化: 2x2窗口，步长2
    - 全连接层1: 输入维度3136，输出维度128，ReLU激活
    - 全连接层2: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN2LayersAvgPool, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.AvgPool2d(2, 2)  # 输出: 14x14x32
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

# 7. 激活函数改为Sigmoid，双卷积层，卷积核适中
class CNN2LayersSigmoid(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，Sigmoid激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 64个3x3卷积核，填充1，Sigmoid激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度3136，输出维度128，Sigmoid激活
    - 全连接层2: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN2LayersSigmoid, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出: 14x14x64
        # 经过第二个池化: 7x7x64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.sigmoid(self.conv1(x)))
        x = self.pool(torch.sigmoid(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# 8. 双卷积层，第二层卷积核大小改为1x1
class CNN2LayersKernel1x1(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 64个1x1卷积核，无填充，ReLU激活（通道变换，无空间信息提取）
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度3136，输出维度128，ReLU激活
    - 全连接层2: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN2LayersKernel1x1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, padding=0)  # 输出: 14x14x64（1x1卷积只改变通道数，不改变空间尺寸）
        # 经过第二个池化: 7x7x64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14x32
        x = F.relu(self.conv2(x))  # 14x14x64（1x1卷积，无空间信息提取）
        x = self.pool(x)  # 7x7x64
        x = x.view(-1, 7 * 7 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 9. 双卷积层，第二层卷积核大小改为5x5
class CNN2LayersKernel5x5(nn.Module):
    """
    架构说明：
    - 卷积层1: 32个3x3卷积核，填充1，ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 卷积层2: 64个5x5卷积核，填充2（保持尺寸），ReLU激活
    - 最大池化: 2x2窗口，步长2
    - 全连接层1: 输入维度3136，输出维度128，ReLU激活
    - 全连接层2: 输入维度128，输出维度10
    """
    def __init__(self):
        super(CNN2LayersKernel5x5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出: 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # 输出: 14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 输出: 14x14x64（5x5卷积，更大感受野）
        # 经过第二个池化: 7x7x64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14x32
        x = self.pool(F.relu(self.conv2(x)))  # 7x7x64
        x = x.view(-1, 7 * 7 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ================ 训练和评估函数 ================

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch, model_name):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'{model_name} - Epoch {epoch}')
    for data, target in pbar:
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
        
        pbar.set_postfix({
            'Loss': f'{train_loss/total:.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
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

# 单个模型的训练和评估函数
def train_and_evaluate_model(model_class, model_name, device, train_loader, test_loader, num_epochs=5):
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    # 初始化模型
    model = model_class().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史记录
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch, model_name)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试
        test_loss, test_acc, predictions, targets = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
    
    # 最终测试
    test_loss, final_test_acc, predictions, targets = test(model, device, test_loader, criterion)
    
    # 计算置信区间
    test_size = len(test_loader.dataset)
    lower_bound, upper_bound = calculate_confidence_interval(final_test_acc, test_size)
    
    print(f"\n{model_name} 结果:")
    print(f"  最终测试准确率: {final_test_acc:.2f}%")
    print(f"  95% 置信区间: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': model_name,
        'final_accuracy': final_test_acc,
        'confidence_interval': (lower_bound, upper_bound),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'model': model
    }

# ================ 主函数 ================
def main():
    # 超参数设置
    batch_size = 128
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"Epoch数: {num_epochs}")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 定义所有要测试的模型
    models_to_test = [
        (CNN2LayersFew, "1. 双卷积层（卷积核较少）"),
        (CNN2LayersMedium, "2. 双卷积层（卷积核适中）"),
        (CNN2LayersMany, "3. 双卷积层（卷积核较多）"),
        (CNN1Layer, "4. 单卷积层"),
        (CNN3Layers, "5. 三卷积层"),
        (CNN2LayersAvgPool, "6. 双卷积层+平均池化"),
        (CNN2LayersSigmoid, "7. 双卷积层+Sigmoid激活"),
        (CNN2LayersKernel1x1, "8. 双卷积层+第二层1x1卷积核"),
        (CNN2LayersKernel5x5, "9. 双卷积层+第二层5x5卷积核")
    ]
    
    # 存储所有模型的结果
    results = []
    
    # 训练和评估每个模型
    for model_class, model_name in models_to_test:
        result = train_and_evaluate_model(model_class, model_name, device, train_loader, test_loader, num_epochs)
        results.append(result)
    
    # 打印所有模型的结果比较
    print("\n" + "="*80)
    print("模型性能比较")
    print("="*80)
    print(f"{'模型':<30} {'准确率(%)':<12} {'置信区间(95%)':<20} {'参数量':<15}")
    print("-"*80)
    
    for result in results:
        model_name = result['model_name']
        accuracy = result['final_accuracy']
        ci_lower, ci_upper = result['confidence_interval']
        params = result['total_params']
        
        print(f"{model_name:<30} {accuracy:>10.2f}%  [{ci_lower:.2f}%, {ci_upper:.2f}%]  {params:>10,}")
    
    # 绘制所有模型的准确率曲线
    plt.figure(figsize=(14, 8))
    
    for i, result in enumerate(results):
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, result['test_accuracies'], marker='o', linewidth=2, 
                 label=f"{result['model_name']} ({result['final_accuracy']:.2f}%)")
    
    plt.xlabel('Epoch')
    plt.ylabel('测试准确率 (%)')
    plt.title('不同CNN模型在MNIST上的性能比较')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 绘制参数量与准确率的关系
    plt.figure(figsize=(12, 6))
    
    for result in results:
        plt.scatter(result['total_params'], result['final_accuracy'], s=100)
        # 只显示简短的模型名称
        short_name = result['model_name'].split(' ')[1] if len(result['model_name'].split(' ')) > 1 else result['model_name']
        plt.annotate(short_name, 
                    (result['total_params'], result['final_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('参数量')
    plt.ylabel('最终测试准确率 (%)')
    plt.title('模型复杂度与性能关系')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 按不同卷积核大小分组比较
    kernel_comparison = [r for r in results if '卷积核' in r['model_name'] and '双卷积层' in r['model_name']]
    
    # 绘制卷积核大小比较
    plt.figure(figsize=(10, 6))
    kernel_sizes = []
    accuracies = []
    
    for result in kernel_comparison:
        if '较少' in result['model_name']:
            kernel_sizes.append('8/16个3x3')
        elif '适中' in result['model_name']:
            kernel_sizes.append('32/64个3x3')
        elif '较多' in result['model_name']:
            kernel_sizes.append('64/128个3x3')
        elif '1x1' in result['model_name']:
            kernel_sizes.append('32/64个1x1')
        elif '5x5' in result['model_name']:
            kernel_sizes.append('32/64个5x5')
        else:
            continue
        
        accuracies.append(result['final_accuracy'])
    
    plt.bar(kernel_sizes, accuracies)
    plt.xlabel('卷积核配置')
    plt.ylabel('测试准确率 (%)')
    plt.title('不同卷积核大小对性能的影响')
    plt.ylim(min(accuracies)-1, max(accuracies)+1)
    
    # 在柱状图上添加数值
    for i, (size, acc) in enumerate(zip(kernel_sizes, accuracies)):
        plt.text(i, acc + 0.1, f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # 找到最佳模型
    best_result = max(results, key=lambda x: x['final_accuracy'])
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_result['model_name']}")
    print(f"最佳准确率: {best_result['final_accuracy']:.2f}%")
    print(f"置信区间: [{best_result['confidence_interval'][0]:.2f}%, {best_result['confidence_interval'][1]:.2f}%]")
    print(f"参数量: {best_result['total_params']:,}")
    print(f"{'='*60}")
    
    return results

# 运行主函数
if __name__ == "__main__":
    results = main()