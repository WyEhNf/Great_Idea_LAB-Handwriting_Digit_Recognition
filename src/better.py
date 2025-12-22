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

# 定义带有ResNet shortcut的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# 定义多层CNN网络（受ResNet启发）
class ResNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCNN, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块层
        # 第一组残差块
        self.layer1 = self._make_layer(32, 32, 2)
        # 第二组残差块，下采样
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        # 第三组残差块
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(128, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
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
        
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = self.fc(x)
        
        return x

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

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
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

# 主训练循环
def main():
    # 超参数设置
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print("=" * 50)
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 初始化模型
    model = ResNetCNN(num_classes=10).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练历史记录
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    print("开始训练...")
    print("=" * 50)
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试
        test_loss, test_acc, predictions, targets = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch}:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
        print("=" * 50)
    
    # 最终测试
    print("最终评估...")
    print("=" * 50)
    
    test_loss, final_test_acc, predictions, targets = test(model, device, test_loader, criterion)
    
    # 计算置信区间
    test_size = len(test_loader.dataset)
    lower_bound, upper_bound = calculate_confidence_interval(final_test_acc, test_size)
    
    print(f"最终测试准确率: {final_test_acc:.2f}%")
    print(f"95% 置信区间: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
    
    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(range(1, num_epochs + 1), train_losses, label='训练损失', marker='o')
    ax1.plot(range(1, num_epochs + 1), test_losses, label='测试损失', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和测试损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率', marker='o')
    ax2.plot(range(1, num_epochs + 1), test_accuracies, label='测试准确率', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练和测试准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 输出模型结构信息
    print("\n模型结构:")
    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 保存模型
    torch.save(model.state_dict(), 'resnet_mnist_model.pth')
    print("模型已保存为 'resnet_mnist_model.pth'")
    
    return model, final_test_acc, (lower_bound, upper_bound)

# 运行主函数
if __name__ == "__main__":
    model, final_acc, confidence_interval = main()