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


# 构建神经网络模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(len(features), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x=F.dropout(x, training=self.training,p=0.2)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training, p=0.2)
        # x = self.fc4(x)
        return x


# 实例化模型并移至GPU
model = MLP().to(device)

# 损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)


# 训练循环
def train_model(num_epochs):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

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

        #scheduler.step()

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    return train_losses, test_losses


# 训练模型
num_epochs = 200
train_losses, test_losses = train_model(num_epochs)
model.eval()  # 设置为评估模式
with torch.no_grad():
    y_train_pred = model(X_train_tensor).cpu().numpy()
    y_test_pred = model(X_test_tensor).cpu().numpy()

# 计算评估指标
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, R2: {r2:.4f}')
