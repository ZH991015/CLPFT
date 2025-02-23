import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

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

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建GBDT模型
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=10, random_state=42)

# 训练模型
model.fit(X_train_scaled, y_train)

# 预测
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 计算MAE
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse=mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
print(f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f},Test MSE: {test_mse:.4f}, R2: {r2:.4f}')

