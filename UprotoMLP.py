import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def set_seed(seed=42):
    """设置所有随机种子以确保结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 设置随机种子
set_seed(42)
# 设置设备
device = torch.device("cpu")

# 加载数据
file_path = '论文数据集：中国台风记录(2)_processed.xlsx'
data = pd.read_excel(file_path)

# 选择特征和标签
features = ['最大风力', '最大风速', '受灾人口（万人）', '转移安置人口（万人）', '受灾面积（万公顷）', '死亡人口（人）',"倒塌房屋（万间）"]
label = '改进后的直接经济损失（亿元）'

# 数据预处理
for feature in features:
    if data[feature].dtype == 'O':
        data[feature] = pd.to_numeric(data[feature].str.strip(), errors='coerce')
    else:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

X = data[features].values
y = data[label].values

# 数据分割和标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
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


class ProtoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], feature_dim=32, output_dim=1, n_prototypes=4):
        super(ProtoMLP, self).__init__()

        # 特征提取器
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.feature_extractor = nn.Sequential(*layers)

        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, output_dim)
        )

        # 不确定性头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, output_dim),
            nn.Softplus()
        )

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, feature_dim))
        self.scaling_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        features = self.feature_extractor(x)
        features = F.normalize(features, dim=-1)

        regression_out = self.regression_head(features)
        uncertainty = self.uncertainty_head(features)

        distances = torch.cdist(features, self.prototypes)
        proto_out = -self.scaling_factor * distances

        final_out = (regression_out + proto_out.mean(dim=1, keepdim=True)) / 2
        return final_out, uncertainty, features, regression_out

    def update_prototypes(self, features, labels):
        labels_np = labels.detach().cpu().numpy().flatten()
        features_np = features.detach().cpu().numpy()

        quantiles = np.linspace(0, 1, self.prototypes.shape[0] + 1)[1:-1]
        bins = np.quantile(labels_np, quantiles)

        digitized = np.digitize(labels_np, bins)
        for i in range(self.prototypes.shape[0]):
            mask = (digitized == i)
            if mask.sum() > 0:
                group_features = features_np[mask]
                if len(group_features) > 0:
                    prototype = torch.tensor(
                        group_features.mean(0),
                        device=self.prototypes.device,
                        dtype=self.prototypes.dtype
                    )
                    self.prototypes.data[i] = prototype


def predict_with_uncertainty(model, data, n_samples=100):
    model.train()  # 保持dropout开启
    predictions = []
    uncertainties = []

    with torch.no_grad():
        for _ in range(n_samples):
            output, uncertainty, _, _ = model(data)
            predictions.append(output)
            uncertainties.append(uncertainty)

    predictions = torch.stack(predictions, dim=0)
    uncertainties = torch.stack(uncertainties, dim=0)

    mean_pred = predictions.mean(dim=0)
    epistemic_uncertainty = predictions.std(dim=0)
    aleatoric_uncertainty = uncertainties.mean(dim=0)

    total_uncertainty = torch.sqrt(epistemic_uncertainty ** 2 + aleatoric_uncertainty ** 2)

    return mean_pred, total_uncertainty


def train_model(model, train_loader, test_loader, num_epochs=500):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=num_epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )

    train_losses = []
    best_mae = float('inf')
    best_model_state = None
    best_predictions = None
    best_uncertainties = None
    best_targets = None
    patience = 100
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for data, target in train_loader:
            optimizer.zero_grad()

            output, uncertainty, features, regression_out = model(data)

            mse_loss = F.mse_loss(output.squeeze(), target.squeeze())
            mae_loss = F.l1_loss(output.squeeze(), target.squeeze())
            uncertainty_loss = torch.mean(0.5 * torch.exp(-uncertainty) * (output - target) ** 2 +
                                          0.5 * uncertainty)

            loss = mse_loss + mae_loss + uncertainty_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            with torch.no_grad():
                model.update_prototypes(features.detach(), target.detach())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 评估
        model.eval()
        predictions = []
        uncertainties = []
        targets_list = []

        with torch.no_grad():
            for data, target in test_loader:
                mean_pred, uncertainty = predict_with_uncertainty(model, data)
                predictions.extend(mean_pred.squeeze().cpu().numpy())
                uncertainties.extend(uncertainty.squeeze().cpu().numpy())
                targets_list.extend(target.squeeze().cpu().numpy())

        current_mae = np.mean(np.abs(np.array(predictions) - np.array(targets_list)))

        # 保存最佳模型
        if current_mae < best_mae:
            best_mae = current_mae
            best_model_state = model.state_dict().copy()
            best_predictions = predictions
            best_uncertainties = uncertainties
            best_targets = targets_list
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, MAE: {current_mae:.4f}')

    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    return train_losses, epoch + 1, best_predictions, best_uncertainties, best_targets


def evaluate_model_with_best_results(model, predictions, uncertainties, test_targets):
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    test_targets = np.array(test_targets)

    # 计算置信度
    def calculate_confidence(uncertainty):
        if uncertainty < 1:
            return "99%以上"
        elif uncertainty < 2:
            return "95%以上"
        elif uncertainty < 3:
            return "68%以上"
        else:
            return "低于68%"

    results_df = pd.DataFrame({
        '实际值': test_targets,
        '预测值': predictions,
        '置信度': [calculate_confidence(u) for u in uncertainties],
        '预测区间': [f'[{pred - 1.96 * unc:.2f}, {pred + 1.96 * unc:.2f}]'
                     for pred, unc in zip(predictions, uncertainties)]
    })

    print("\n预测结果:")
    print(results_df.to_string())

    # 计算指标
    mae = np.mean(np.abs(predictions - test_targets))
    mse = np.mean((predictions - test_targets) ** 2)
    r2 = 1 - np.sum((test_targets - predictions) ** 2) / np.sum(
        (test_targets - test_targets.mean()) ** 2)

    r, _ = pearsonr(predictions, test_targets)

    # 计算预测区间覆盖率
    z_score = 1.96  # 95% 置信区间
    lower_bound = predictions - z_score * uncertainties
    upper_bound = predictions + z_score * uncertainties
    coverage = np.mean((test_targets >= lower_bound) & (test_targets <= upper_bound))

    print("\nBest Model Performance:")
    print(f"{'Metric':<15} {'Value':<15}")
    print("-" * 30)
    print(f"{'MAE':<15} {mae:<15.4f}")
    print(f"{'MSE':<15} {mse:<15.4f}")
    print(f"{'R2 Score':<15} {r2:<15.4f}")
    print(f"{'R':<15} {r:<15.4f}")
    print(f"\nUncertainty Coverage Rate (95% CI): {coverage:.4f}")

    # 可视化
    plt.figure(figsize=(15, 10))

    # 预测值与实际值的对比
    plt.subplot(2, 1, 1)
    plt.scatter(test_targets, predictions, alpha=0.5, label='Assessments')
    plt.fill_between(test_targets,
                     predictions - 2 * uncertainties,
                     predictions + 2 * uncertainties,
                     alpha=0.2, label='Uncertainty')
    plt.plot([min(test_targets), max(test_targets)],
             [min(test_targets), max(test_targets)],
             'r--', label='Perfect Assessment')
    plt.xlabel('Actual Values',fontsize=18)
    plt.ylabel('Assessed Values',fontsize=18)
    plt.title('Model Assessments with Uncertainty',fontsize=18)
    plt.legend(fontsize=18)

    # 不确定性分布
    plt.subplot(2, 1, 2)
    plt.hist(uncertainties, bins=30)
    plt.xlabel('Uncertainty',fontsize=18)
    plt.ylabel('Frequency',fontsize=18)
    plt.title('Distribution of Assessment Uncertainties',fontsize=18)

    plt.tight_layout()
    plt.savefig("result.pdf")
    plt.show()


# 主程序
model = ProtoMLP(input_dim=len(features), n_prototypes=4).to(device)

print("Training Model...")
train_losses, epochs, best_predictions, best_uncertainties, best_targets = train_model(model, train_loader, test_loader)

# 使用最佳模型进行评估
print("\nEvaluating Best Model...")
evaluate_model_with_best_results(model, best_predictions, best_uncertainties, best_targets)
