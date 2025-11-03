# Heading 连续体机器人位姿预测系统：基于方向感知ResNet的完整解决方案

## Heading 项目概述

<p>本方案旨在构建一个基于ResNet的深度学习模型，用于预测连续体机器人在外力作用下的位姿变化。系统通过处理施加的横向外力、竖直方向的基准力和稳态力，以及NOKOV光学标记点数据，输出机器人的7维位姿（位置和四元数）</p>。


## Heading 输入数据规范与预处理

### Heading 输入数据组成

数据类型 维度 描述 示例值

施加外力向量 3维 横向外力的大小和方向 [Fx, Fy, Fz]

基准力向量 3维 竖直向下的基准力 [0, 0, -F_base]

稳态力向量 3维 竖直向下的稳态力 [0, 0, -F_steady]

### Heading 输出数据
nokov数据

### Heading 数据预处理代码

```python
import numpy as np
import torch

def prepare_input_features(F_applied, direction, F_base, F_steady, base_markers):
    """
    准备模型输入特征
    
    参数:
        F_applied: 施加的外力大小（标量，单位：N）
        direction: 外力方向向量 [dx, dy, dz]（需规范化）
        F_base: 基准力大小（标量，竖直向下）
        F_steady: 稳态力大小（标量，竖直向下） 
        base_markers: 基准标记点坐标列表（27维）
    
    返回:
        dict: 包含所有输入特征的字典
    """
    # 规范化外力方向向量
    dir_array = np.array(direction, dtype=np.float32)
    dir_normalized = dir_array / np.linalg.norm(dir_array)
    
    # 计算施加的外力向量
    applied_force_vector = F_applied * dir_normalized
    
    # 构建竖直向下的力向量（假设Z轴向下）
    base_force_vector = np.array([0.0, 0.0, -F_base], dtype=np.float32)
    steady_force_vector = np.array([0.0, 0.0, -F_steady], dtype=np.float32)
    
    # 确保标记点数据格式正确
    base_markers_array = np.array(base_markers, dtype=np.float32).flatten()
    if len(base_markers_array) != 27:
        raise ValueError(f"标记点数据应为27维，当前维度: {len(base_markers_array)}")
    
    features = {
        'applied_force_vector': applied_force_vector,
        'base_force_vector': base_force_vector,
        'steady_force_vector': steady_force_vector,
        'base_markers': base_markers_array
    }
    
    return features

def create_training_dataset(raw_experiment_data):
    """
    从原始实验数据创建训练数据集
    
    参数:
        raw_experiment_data: 原始实验数据列表
        
    返回:
        list: 标准化后的训练样本列表
    """
    training_samples = []
    
    for i, experiment in enumerate(raw_experiment_data):
        try:
            # 提取实验数据
            F_applied = experiment['applied_force_magnitude']
            direction = experiment['force_direction']
            F_base = experiment['base_force_magnitude'] 
            F_steady = experiment['steady_force_magnitude']
            base_markers = experiment['base_markers']
            target_pose = experiment['target_pose']  # 7维位姿
            
            # 准备输入特征
            features = prepare_input_features(F_applied, direction, F_base, F_steady, base_markers)
            
            # 转换为PyTorch Tensor
            sample = {
                'applied_force': torch.FloatTensor(features['applied_force_vector']),
                'base_force': torch.FloatTensor(features['base_force_vector']),
                'steady_force': torch.FloatTensor(features['steady_force_vector']),
                'markers': torch.FloatTensor(features['base_markers']),
                'target_pose': torch.FloatTensor(target_pose)
            }
            
            training_samples.append(sample)
            
        except Exception as e:
            print(f"处理实验数据 {i} 时出错: {e}")
            continue
    
    print(f"成功创建 {len(training_samples)} 个训练样本")
    return training_samples
```


2.3 数据标准化处理
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataNormalizer:
    """数据标准化处理器"""
    
    def __init__(self):
        self.force_scaler = StandardScaler()
        self.pose_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, training_samples):
        """基于训练数据拟合标准化参数"""
        force_data = []
        pose_data = []
        
        for sample in training_samples:
            # 收集力数据（合并所有力向量）
            force_vector = np.concatenate([
                sample['applied_force'].numpy(),
                sample['base_force'].numpy(), 
                sample['steady_force'].numpy()
            ])
            force_data.append(force_vector)
            
            # 收集位姿数据
            pose_data.append(sample['target_pose'].numpy())
        
        # 拟合标准化器
        self.force_scaler.fit(np.array(force_data))
        self.pose_scaler.fit(np.array(pose_data))
        self.is_fitted = True
        
        print("数据标准化器拟合完成")
    
    def transform_input(self, sample):
        """转换输入样本"""
        if not self.is_fitted:
            raise ValueError("标准化器尚未拟合，请先调用fit方法")
        
        # 标准化力数据
        force_vector = np.concatenate([
            sample['applied_force'].numpy(),
            sample['base_force'].numpy(),
            sample['steady_force'].numpy()
        ])
        force_normalized = self.force_scaler.transform([force_vector])[0]
        
        # 分割回各自的力向量
        applied_force_norm = torch.FloatTensor(force_normalized[:3])
        base_force_norm = torch.FloatTensor(force_normalized[3:6])
        steady_force_norm = torch.FloatTensor(force_normalized[6:9])
        
        # 标记点数据保持原样（已在同一尺度）
        markers_norm = sample['markers']
        
        return {
            'applied_force': applied_force_norm,
            'base_force': base_force_norm, 
            'steady_force': steady_force_norm,
            'markers': markers_norm,
            'target_pose': sample['target_pose']
        }
```

1. 网络架构设计

3.1 方向感知ResNet核心架构
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionAwareResNet(nn.Module):
    """
    方向感知ResNet网络
    专门处理不同方向的力输入和标记点数据
    """
    
    def __init__(self, input_dims=None, output_dim=7, hidden_dims=[128, 256, 512], dropout_rate=0.1):
        super().__init__()
        
        # 设置默认输入维度
        if input_dims is None:
            input_dims = {
                'applied_force': 3,    # 施加外力向量
                'base_force': 3,      # 基准力向量  
                'steady_force': 3,    # 稳态力向量
                'markers': 27         # 标记点数据
            }
        
        self.input_dims = input_dims
        self.dropout_rate = dropout_rate
        
        # 1. 多流编码器：分别处理不同输入
        self._build_stream_encoders()
        
        # 2. 特征融合层
        fusion_dim = 16 + 16 + 16 + 64  # 各编码器输出维度之和
        self.fusion_layer = nn.Linear(fusion_dim, hidden_dims[0])
        self.fusion_bn = nn.BatchNorm1d(hidden_dims[0])
        self.fusion_dropout = nn.Dropout(dropout_rate)
        
        # 3. ResNet主干网络
        self.res_blocks = self._build_res_blocks(hidden_dims)
        
        # 4. 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # 权重初始化
        self._initialize_weights()
    
    def _build_stream_encoders(self):
        """构建多流编码器"""
        # 施加外力编码器
        self.applied_force_encoder = nn.Sequential(
            nn.Linear(self.input_dims['applied_force'], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        # 基准力编码器
        self.base_force_encoder = nn.Sequential(
            nn.Linear(self.input_dims['base_force'], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), 
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        # 稳态力编码器
        self.steady_force_encoder = nn.Sequential(
            nn.Linear(self.input_dims['steady_force'], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        # 标记点编码器
        self.marker_encoder = nn.Sequential(
            nn.Linear(self.input_dims['markers'], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
    
    def _build_res_blocks(self, hidden_dims):
        """构建残差块序列"""
        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                ResidualBlock(
                    hidden_dims[i], 
                    hidden_dims[i+1],
                    dropout_rate=self.dropout_rate
                )
            )
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, applied_force, base_force, steady_force, markers):
        """
        前向传播
        
        参数:
            applied_force: 施加外力向量 [batch_size, 3]
            base_force: 基准力向量 [batch_size, 3]
            steady_force: 稳态力向量 [batch_size, 3] 
            markers: 标记点数据 [batch_size, 27]
            
        返回:
            预测位姿 [batch_size, 7]
        """
        batch_size = applied_force.size(0)
        
        # 1. 分别编码不同输入流
        applied_encoded = self.applied_force_encoder(applied_force)
        base_encoded = self.base_force_encoder(base_force)
        steady_encoded = self.steady_force_encoder(steady_force)
        markers_encoded = self.marker_encoder(markers)
        
        # 2. 特征融合
        combined = torch.cat([applied_encoded, base_encoded, steady_encoded, markers_encoded], dim=1)
        fused = self.fusion_layer(combined)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused)
        fused = self.fusion_dropout(fused)
        
        # 3. 通过ResNet块
        features = self.res_blocks(fused)
        
        # 4. 输出预测
        return self.output_layer(features)

class ResidualBlock(nn.Module):
    """残差块实现"""
    
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        
        # 主路径
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 主路径
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += identity
        return F.relu(out)


3.2 网络配置参数说明

def get_model_config(model_type='standard'):
    """获取不同规模的模型配置"""
    
    configs = {
        'small': {
            'hidden_dims': [64, 128, 256],
            'dropout_rate': 0.1
        },
        'standard': {
            'hidden_dims': [128, 256, 512], 
            'dropout_rate': 0.1
        },
        'large': {
            'hidden_dims': [256, 512, 1024],
            'dropout_rate': 0.2
        }
    }
    
    return configs.get(model_type, configs['standard'])

# 模型初始化示例
config = get_model_config('standard')
model = DirectionAwareResNet(
    hidden_dims=config['hidden_dims'],
    dropout_rate=config['dropout_rate']
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
```

4. 训练流程与优化策略

4.1 数据加载器实现
```python
from torch.utils.data import Dataset, DataLoader
import random

class ForcePoseDataset(Dataset):
    """力-位姿数据集"""
    
    def __init__(self, samples, normalizer=None):
        self.samples = samples
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.normalizer:
            sample = self.normalizer.transform_input(sample)
        
        return sample
    
    def split_train_val(self, val_ratio=0.2):
        """分割训练集和验证集"""
        total_size = len(self.samples)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        
        return ForcePoseDataset(train_samples, self.normalizer), \
               ForcePoseDataset(val_samples, self.normalizer)

def create_data_loaders(dataset, batch_size=32, num_workers=4):
    """创建数据加载器"""
    
    # 自定义collate函数处理字典数据
    def collate_fn(batch):
        applied_forces = torch.stack([item['applied_force'] for item in batch])
        base_forces = torch.stack([item['base_force'] for item in batch])
        steady_forces = torch.stack([item['steady_force'] for item in batch])
        markers = torch.stack([item['markers'] for item in batch])
        target_poses = torch.stack([item['target_pose'] for item in batch])
        
        return {
            'applied_force': applied_forces,
            'base_force': base_forces,
            'steady_force': steady_forces,
            'markers': markers,
            'target_pose': target_poses
        }
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
```

4.2 训练循环实现
```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, optimizer, criterion, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据转移到设备
            applied_force = batch['applied_force'].to(self.device)
            base_force = batch['base_force'].to(self.device)
            steady_force = batch['steady_force'].to(self.device)
            markers = batch['markers'].to(self.device)
            target_pose = batch['target_pose'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(applied_force, base_force, steady_force, markers)
            loss = criterion(outputs, target_pose)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / len(train_loader)
        return epoch_loss
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                applied_force = batch['applied_force'].to(self.device)
                base_force = batch['base_force'].to(self.device)
                steady_force = batch['steady_force'].to(self.device)
                markers = batch['markers'].to(self.device)
                target_pose = batch['target_pose'].to(self.device)
                
                outputs = self.model(applied_force, base_force, steady_force, markers)
                loss = criterion(outputs, target_pose)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=100, 
              learning_rate=1e-3, weight_decay=1e-4):
        """完整训练流程"""
        
        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.MSELoss()
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        print("开始训练...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_loss = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            
            # 验证阶段
            val_loss = self.validate(val_loader, criterion)
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"保存最佳模型，验证损失: {val_loss:.6f}")
            
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                  f'Time: {epoch_time:.2f}s')
        
        total_time = time.time() - start_time
        print(f"训练完成，总时间: {total_time:.2f}s")
        
        return history
```

4.3 损失函数定制
```python
class PoseLoss(nn.Module):
    """位姿专用损失函数"""
    
    def __init__(self, position_weight=2.0, orientation_weight=1.0):
        super().__init__()
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # 分离位置和姿态
        pred_pos, pred_orient = pred[:, :3], pred[:, 3:]
        target_pos, target_orient = target[:, :3], target[:, 3:]
        
        # 位置损失（更重要的部分）
        pos_loss = self.mse_loss(pred_pos, target_pos) * self.position_weight
        
        # 姿态损失（四元数需要特殊处理）
        orient_loss = self.quaternion_loss(pred_orient, target_orient) * self.orientation_weight
        
        return pos_loss + orient_loss
    
    def quaternion_loss(self, q1, q2):
        """四元数差异损失"""
        # 确保四元数规范化
        q1_norm = F.normalize(q1, p=2, dim=1)
        q2_norm = F.normalize(q2, p=2, dim=1)
        
        # 计算角度差
        dot_product = torch.sum(q1_norm * q2_norm, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_diff = 2 * torch.acos(torch.abs(dot_product))
        
        return torch.mean(angle_diff)

```
5. 模型评估与可视化

5.1 评估指标计算

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(self, data_loader):
        """全面评估模型性能"""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                applied_force = batch['applied_force'].to(self.device)
                base_force = batch['base_force'].to(self.device)
                steady_force = batch['steady_force'].to(self.device)
                markers = batch['markers'].to(self.device)
                targets = batch['target_pose'].to(self.device)
                
                predictions = self.model(applied_force, base_force, steady_force, markers)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self._calculate_metrics(predictions, targets)
        return metrics, predictions, targets
    
    def _calculate_metrics(self, predictions, targets):
        """计算各种评估指标"""
        metrics = {}
        
        # 位置误差（欧氏距离）
        pred_pos = predictions[:, :3]
        target_pos = targets[:, :3]
        position_errors = np.linalg.norm(pred_pos - target_pos, axis=1)
        metrics['position_rmse'] = np.sqrt(np.mean(position_errors2))
        metrics['position_mae'] = np.mean(position_errors)
        
        # 姿态误差（角度差）
        pred_orient = predictions[:, 3:]
        target_orient = targets[:, 3:]
        orientation_errors = self._calculate_orientation_error(pred_orient, target_orient)

```