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
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class TaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TaskEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.encoder(x)


class MetaNetPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MetaNetPlus, self).__init__()

        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 替换BatchNorm为LayerNorm
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)  # 替换BatchNorm为LayerNorm
        )

        # 任务编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 替换BatchNorm为LayerNorm
            nn.Dropout(0.2)
        )

        # 原型网络层
        self.prototype_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)  # 替换BatchNorm为LayerNorm
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # 元学习层
        self.meta_learner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 替换BatchNorm为LayerNorm
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 任务适应层
        self.task_adaptation = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

    def compute_prototype(self, support_set):
        features = self.encoder(support_set)
        task_encoding = self.task_encoder(support_set)
        combined = features + task_encoding
        prototype = self.prototype_layer(combined.mean(0, keepdim=True))
        return prototype, combined

    def forward(self, x, support_set=None):
        query_features = self.encoder(x)

        if support_set is not None:
            prototype, support_features = self.compute_prototype(support_set)

            adapted_features = query_features
            for layer in self.task_adaptation:
                adapted_features = layer(adapted_features) + adapted_features

            attention_input = torch.cat([adapted_features,
                                         prototype.expand(adapted_features.shape[0], -1)], dim=1)
            attention_weights = self.attention(attention_input)
            attended_features = attention_weights * adapted_features

            combined = torch.cat([attended_features,
                                  prototype.expand(attended_features.shape[0], -1)], dim=1)
        else:
            combined = torch.cat([query_features, query_features], dim=1)

        output = self.meta_learner(combined)
        return output

def meta_train_step(model, batch_data, support_data, criterion, meta_optimizer, device):
    # 内循环优化
    task_optimizer = optim.SGD(model.task_adaptation.parameters(), lr=0.006)

    support_x, support_y = support_data
    batch_x, batch_y = batch_data

    # 支持集上的快速适应
    model.train()
    with torch.enable_grad():
        support_pred = model(support_x, support_x)
        support_loss = criterion(support_pred, support_y)
        task_optimizer.zero_grad()
        support_loss.backward(retain_graph=True)
        task_optimizer.step()

    # 查询集上的元优化
    query_pred = model(batch_x, support_x)
    query_loss = criterion(query_pred, batch_y)
    meta_optimizer.zero_grad()
    query_loss.backward()
    meta_optimizer.step()

    return query_loss.item()


def train_metanet(model, train_loader, test_loader, num_epochs, device):
    criterion = nn.L1Loss()
    meta_optimizer = optim.Adam(model.parameters(), lr=0.0006)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=num_epochs)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 从batch中随机选择support set
            support_idx = torch.randperm(data.size(0))[:5]
            support_set = (data[support_idx], target[support_idx])
            query_set = (data, target)

            # 元训练步骤
            loss = meta_train_step(model, query_set, support_set, criterion,
                                   meta_optimizer, device)
            total_train_loss += loss

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 评估
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}')

    return train_losses, test_losses


# 实例化模型和训练
model = MetaNetPlus(len(features)).to(device)
train_losses, test_losses = train_metanet(model, train_loader, test_loader,
                                          num_epochs=100, device=device)

# 评估
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).cpu().numpy()
    y_test_pred = model(X_test_tensor).cpu().numpy()

# 计算评估指标
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, '
      f'Test MSE: {test_mse:.4f}, R2: {r2:.4f}')