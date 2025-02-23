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
import matplotlib.pyplot as plt
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


class DN4PlusPlus(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DN4PlusPlus, self).__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # 局部描述符生成器
        self.local_descriptor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )

        # 注意力模块
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x, support_set=None):
        # 提取全局特征
        global_features = self.feature_extractor(x)

        # 生成局部描述符
        local_descs = self.local_descriptor(global_features)

        # 计算注意力权重
        attention_weights = self.attention(local_descs)

        # 加权局部描述符
        weighted_descs = local_descs * attention_weights

        if support_set is None:
            # 测试阶段直接预测
            return self.predictor(weighted_descs)

        # 处理支持集
        support_features = self.feature_extractor(support_set)
        support_descs = self.local_descriptor(support_features)
        support_attention = self.attention(support_descs)
        weighted_support = support_descs * support_attention

        # 计算与支持集的相似度
        batch_size = x.size(0)
        support_size = support_set.size(0)

        query_features = weighted_descs.unsqueeze(1).repeat(1, support_size, 1)
        support_features = weighted_support.unsqueeze(0).repeat(batch_size, 1, 1)

        # 计算余弦相似度
        similarity = F.cosine_similarity(query_features, support_features, dim=2)

        # 基于相似度的加权预测
        similarity_weights = F.softmax(similarity * 10, dim=1)  # 添加温度系数
        weighted_features = torch.sum(support_features * similarity_weights.unsqueeze(-1), dim=1)

        # 最终预测
        prediction = self.predictor(weighted_features)

        return prediction


def train_dn4plus(model, train_loader, test_loader, num_epochs, device, early_stopping_patience=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 随机选择支持集
            support_indices = torch.randperm(len(data))[:32]
            support_set = data[support_indices]

            optimizer.zero_grad()
            output = model(data, support_set)
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # 评估
        model.eval()
        test_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # 学习率调整
        scheduler.step(avg_test_loss)

        # 计算评估指标
        test_mae = mean_absolute_error(actuals, predictions)
        test_mse = mean_squared_error(actuals, predictions)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(actuals, predictions)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}')
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')
        print(f'Test R2: {test_r2:.4f}')
        print('-------------------')

        # 早停机制
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            model.load_state_dict(best_model_state)
            break

    return train_losses, test_losses


def plot_training_history(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print("\nModel Evaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return predictions, actuals


# 模型训练和评估的使用示例
def main():
    # 假设数据已经准备好，并创建了相应的DataLoader
    input_dim = X_train.shape[1]  # 特征维度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化模型
    model = DN4PlusPlus(input_dim).to(device)

    # 训练模型
    num_epochs = 200
    train_losses, test_losses = train_dn4plus(model, train_loader, test_loader, num_epochs, device)

    # 绘制训练历史
    plot_training_history(train_losses, test_losses)

    # 评估模型
    predictions, actuals = evaluate_model(model, test_loader, device)

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
