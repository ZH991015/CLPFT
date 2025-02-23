import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F
# 加载数据
file_path = './论文数据集：中国台风记录(2)_processed.xlsx'  # 请确保路径正确
data = pd.read_excel(file_path)

# 选择特征和标签
features = ['最大风力', '最大风速', '受灾人口（万人）', '转移安置人口（万人）', '受灾面积（万公顷）', '死亡人口（人）']
label = '改进后的直接经济损失（亿元）'

# 检查数据类型并转换为数值类型
for feature in features:
    if data[feature].dtype == 'O':  # 检查是否为对象类型（通常是字符串）
        data[feature] = pd.to_numeric(data[feature].str.strip(), errors='coerce')
    else:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')


# 处理缺失值
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# 数据和标签
X = data[features].values
y = data[label].values
print(X.shape)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 创建数据加载器
test_domain_labels = torch.zeros(len(X_test_tensor), dtype=torch.long).to(device)  # 假设测试集为源域
test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_domain_labels)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# TADA模型定义
class TADA(nn.Module):
    def __init__(self, input_dim=len(features)):
        super(TADA, self).__init__()
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 时间自适应层
        self.temporal_adapter = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh()
        )

        # 域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        temporal_features = self.temporal_adapter(features)

        if self.training:
            reverse_features = ReverseLayerF.apply(temporal_features, alpha)
            domain_output = self.domain_classifier(reverse_features)
        else:
            domain_output = None

        prediction = self.predictor(temporal_features)
        return prediction, domain_output


# 反向梯度层
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# 创建源域和目标域数据
# 这里假设较早的数据为源域,较新的数据为目标域
split_idx = int(len(X_train) * 0.7)
source_idx = np.arange(split_idx)
target_idx = np.arange(split_idx, len(X_train))

# 准备源域和目标域数据
X_source = X_train_tensor[source_idx]
y_source = y_train_tensor[source_idx]
X_target = X_train_tensor[target_idx]
y_target = y_train_tensor[target_idx]

# 创建域标签
source_domain_labels = torch.zeros(len(X_source), dtype=torch.long).to(device)
target_domain_labels = torch.ones(len(X_target), dtype=torch.long).to(device)

# 创建数据加载器
source_dataset = TensorDataset(X_source, y_source, source_domain_labels)
target_dataset = TensorDataset(X_target, y_target, target_domain_labels)
source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

# 实例化模型
model = TADA().to(device)

# 损失函数
regression_criterion = nn.L1Loss()
domain_criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0006)


# 训练函数
def train_tada(num_epochs):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_regression_loss = 0
        total_domain_loss = 0

        # 计算alpha值
        p = epoch / num_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        # 源域训练
        for (source_data, source_labels, source_domain) in source_loader:
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            source_domain = source_domain.to(device)

            optimizer.zero_grad()

            # 前向传播
            predictions, domain_outputs = model(source_data, alpha)

            # 计算损失
            regression_loss = regression_criterion(predictions, source_labels)
            domain_loss = domain_criterion(domain_outputs, source_domain)

            loss = regression_loss + 0.1 * domain_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_regression_loss += regression_loss.item()
            total_domain_loss += domain_loss.item()

        # 目标域训练
        for (target_data, target_labels, target_domain) in target_loader:
            target_data = target_data.to(device)
            target_domain = target_domain.to(device)

            optimizer.zero_grad()

            # 前向传播
            _, domain_outputs = model(target_data, alpha)

            # 计算域损失
            domain_loss = domain_criterion(domain_outputs, target_domain)

            # 反向传播
            domain_loss.backward()
            optimizer.step()

            total_domain_loss += domain_loss.item()


        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target, domain in test_loader:
                data, target = data.to(device), target.to(device)
                predictions, _ = model(data)
                test_loss += regression_criterion(predictions, target).item()

        avg_train_loss = total_loss / len(source_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}')
        print(f'Regression Loss: {total_regression_loss / len(source_loader):.4f}')
        print(f'Domain Loss: {total_domain_loss / (len(source_loader) + len(target_loader)):.4f}')
        print('-------------------')

    return train_losses, test_losses


# 训练模型
num_epochs = 250
train_losses, test_losses = train_tada(num_epochs)

# 评估模型
model.eval()
with torch.no_grad():
    y_train_pred, _ = model(X_train_tensor)
    y_test_pred, _ = model(X_test_tensor)
    y_train_pred = y_train_pred.cpu().numpy()
    y_test_pred = y_test_pred.cpu().numpy()

# 计算评估指标
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f'Train MAE: {train_mae:.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test MSE: {test_mse:.4f}')
print(f'R2 Score: {r2:.4f}')
