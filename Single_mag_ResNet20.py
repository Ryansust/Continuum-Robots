import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
# 用于训练模型

# Define ResUnit
class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResUnit, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

# Define ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.resunit1 = ResUnit(in_channels, out_channels, stride)
        self.resunit2 = ResUnit(out_channels, out_channels)
        self.resunit3 = ResUnit(out_channels, out_channels)

    def forward(self, x):
        out = self.resunit1(x)
        out = self.resunit2(out)
        out = self.resunit3(out)
        return out

# Define the complete ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=2)  # Initial Conv layer
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.resblock1 = ResBlock(16, 16)
        self.resblock2 = ResBlock(16, 32, stride=2)
        self.resblock3 = ResBlock(32, 64, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.fc_position = nn.Linear(64, 3)
        self.fc_orientation = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        ###########################
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        ###########################
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        position = self.fc_position(x)
        orientation = torch.tanh(self.fc_orientation(x))

        prediction = torch.cat((position, orientation), dim=1)

        #x = self.fc(x)
        return prediction

def get_labels(file_name):
    data = pd.read_csv(file_name)

    pos_x = data.iloc[:, 1]
    pos_x = pos_x.values
    pos_x = pos_x.reshape(1, -1)

    pos_y = data.iloc[:, 2]
    pos_y = pos_y.values
    pos_y = pos_y.reshape(1, -1)

    pos_z = data.iloc[:, 3]
    pos_z = pos_z.values
    pos_z = pos_z.reshape(1, -1)

    ori_x = data.iloc[:, 4]
    ori_x = ori_x.values
    ori_x = ori_x.reshape(1, -1)

    ori_y = data.iloc[:, 5]
    ori_y = ori_y.values
    ori_y = ori_y.reshape(1, -1)

    ori_z = data.iloc[:, 6]
    ori_z = ori_z.values
    ori_z = ori_z.reshape(1, -1)

    pos_x = pos_x.astype(np.float32)
    pos_y = pos_y.astype(np.float32)
    pos_z = pos_z.astype(np.float32)
    ori_x = ori_x.astype(np.float32)
    ori_y = ori_y.astype(np.float32)
    ori_z = ori_z.astype(np.float32)

    mag_label = np.vstack([pos_x, pos_y, pos_z, ori_x, ori_y, ori_z])
    mag_label = np.transpose(mag_label)
    return mag_label

def get_data(file_name):
    data = pd.read_csv(file_name)
    Tx = data[
        ['Tx_13', 'Tx_14', 'Tx_15', 'Tx_16', 'Tx_9', 'Tx_10', 'Tx_11', 'Tx_12', 'Tx_5', 'Tx_6', 'Tx_7', 'Tx_8', 'Tx_1',
         'Tx_2', 'Tx_3', 'Tx_4']].values
    Tx = Tx.reshape(-1, 4, 4)

    Ty = data[
        ['Ty_13', 'Ty_14', 'Ty_15', 'Ty_16', 'Ty_9', 'Ty_10', 'Ty_11', 'Ty_12', 'Ty_5', 'Ty_6', 'Ty_7', 'Ty_8', 'Ty_1',
         'Ty_2', 'Ty_3', 'Ty_4']].values
    Ty = Ty.reshape(-1, 4, 4)

    Tz = data[
        ['Tz_13', 'Tz_14', 'Tz_15', 'Tz_16', 'Tz_9', 'Tz_10', 'Tz_11', 'Tz_12', 'Tz_5', 'Tz_6', 'Tz_7', 'Tz_8', 'Tz_1',
         'Tz_2', 'Tz_3', 'Tz_4']].values
    Tz = Tz.reshape(-1, 4, 4)

    Tx = Tx.astype(np.float32)
    Ty = Ty.astype(np.float32)
    Tz = Tz.astype(np.float32)

    mag_data = np.stack([Tx, Ty, Tz], axis=1)
    return mag_data

mag_data = get_data("Magtracking_guopengfei-main/Single_mag_database.csv")
mag_label = get_labels("Magtracking_guopengfei-main/Single_mag_database.csv")

# mag_data = get_data("normalized_processed_data_for_train_202401.csv")
# mag_label = get_labels("normalized_processed_data_for_train_202401.csv")


indexes = np.arange(0, mag_label.shape[0])
np.random.shuffle(indexes)
total_length = len(indexes)
index_train_len = int(total_length * 0.7)
index_val_len = int(total_length * 0.1)
index_test_len = total_length - index_train_len - index_val_len

# 切分成三个数组
index_train = indexes[:index_train_len]
index_val = indexes[index_train_len: index_train_len + index_val_len]
index_test = indexes[index_train_len + index_val_len:]


train_data = []
train_labels = []
val_data = []
val_labels = []
test_data = []
test_labels = []

for i in index_train:
    train_data.append(mag_data[i])
    train_labels.append(mag_label[i])
train_data = np.stack(train_data, axis = 0)
# noise_factor = 0.003  # 注意是标准差
# # 生成与 test_data 相同形状的高斯噪声
# gaussian_noise = np.random.normal(loc=0, scale=noise_factor, size=train_data.shape)
# # print(gaussian_noise)
# # 将高斯噪声加到原始数据上
# train_data = train_data + gaussian_noise
train_data = train_data.astype(np.float32)
train_labels = np.stack(train_labels, axis = 0)
train_data = torch.from_numpy(train_data)
train_labels = torch.from_numpy(train_labels)

for i in index_val:
    val_data.append(mag_data[i])
    val_labels.append(mag_label[i])
val_data = np.stack(val_data, axis=0)
# noise_factor = 0.003  # 注意是标准差
# # 生成与 test_data 相同形状的高斯噪声
# gaussian_noise = np.random.normal(loc=0, scale=noise_factor, size=val_data.shape)
# # print(gaussian_noise)
# # 将高斯噪声加到原始数据上
# val_data = val_data + gaussian_noise
val_data = val_data.astype(np.float32)
val_labels = np.stack(val_labels, axis=0)
val_data = torch.from_numpy(val_data)
val_labels = torch.from_numpy(val_labels)

for i in index_test:
    test_data.append(mag_data[i])
    test_labels.append(mag_label[i])
test_data = np.stack(test_data, axis=0)
# 调整噪声的强度,给测试加入人工噪声，检测模型鲁棒性
# noise_factor = 0.003  # 注意是标准差
# # 生成与 test_data 相同形状的高斯噪声
# gaussian_noise = np.random.normal(loc=0, scale=noise_factor, size=test_data.shape)
# # print(gaussian_noise)
# # 将高斯噪声加到原始数据上
# test_data = test_data + gaussian_noise
test_data = test_data.astype(np.float32)
test_labels = np.stack(test_labels, axis=0)
test_data = torch.from_numpy(test_data)
test_labels = torch.from_numpy(test_labels)

#print(test_data.shape)
#print(test_labels.shape)

# # Generate random data for training, validation, and testing
# # 生成随机数据时，将目标的最后一个维度展平
# train_data = torch.randn(700, 3, 4, 4)
# train_labels = torch.randn(700, 3, 1).view(-1, 3) # 之后尺寸变为(700,3)，好像直接randn(700,3)就行
#
# val_data = torch.randn(100, 3, 4, 4)
# val_labels = torch.randn(100, 3, 1).view(-1, 3)
#
# test_data = torch.randn(200, 3, 4, 4)
# test_labels = torch.randn(200, 3, 1).view(-1, 3)


# Create DataLoader
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# 准确度计算函数
def calculate_accuracy(outputs, labels, threshold_position=0.004, threshold_orientation=5):  #要改
    #correct_predictions = ((torch.abs(outputs - labels) <= threshold).all(dim=1)).sum().item()
    total_samples = labels.size(0)
    position_pred = outputs[:, :3]
    orientation_pred = outputs[:, 3:]
    position_labels = labels[:, :3]
    orientation_labels = labels[:, 3:]
    #print('position_pred', position_pred)
    #print('labels_load', position_labels)
    #print()

    # Accuracy calculation for position (Euclidean distance)
    # position_diff = torch.norm(position_pred - position_labels, dim=1)
    # correct_position_predictions = (position_diff <= threshold_position).sum().item()

    correct_position_predictions = ((torch.abs(position_pred - position_labels) <= threshold_position).all(dim=1)).sum().item()
    accuracy_position = correct_position_predictions / total_samples

    # Accuracy calculation for orientation (angular difference)
    orientation_diff = torch.acos(torch.clamp(torch.sum(orientation_pred * orientation_labels, dim=1), -1.0, 1.0))
    orientation_diff = orientation_diff * (180.0 / math.pi)  # Convert radians to degrees
    correct_orientation_predictions = (orientation_diff <= threshold_orientation).sum().item()
    accuracy_orientation = correct_orientation_predictions / total_samples

    return accuracy_position, accuracy_orientation




# Initialize the model, loss function, and optimizer
model = ResNet()
#print('mag_ResNet20 is', model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Training loop
#num_epochs = 150

# Train
acc_list_val_pos = []
acc_list_val_ori = []
epoch = 0
#for epoch in range(num_epochs):
while True:
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs)
        #print(outputs.shape) # torch.Size([16, 6])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += criterion(outputs, labels).item()

    train_loss /= len(train_loader)
    print("train loss:", train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_accuracy_pos = 0.0
        val_accuracy_ori = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            #print("output", outputs)
            #print("label", labels)
            val_loss += criterion(outputs, labels).item()
            val_accuracy_pos += calculate_accuracy(outputs, labels)[0]
            val_accuracy_ori += calculate_accuracy(outputs, labels)[1]

    #print(inputs)
    #print(outputs)
    #print(labels)

    #print("len(val_loader)", len(val_loader))
    val_loss /= len(val_loader)
    #print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}')
    epoch+=1
    print(f'Epoch {epoch}, Loss: {val_loss}')

    val_accuracy_pos /= len(val_loader)
    val_accuracy_ori /= len(val_loader)
    print(f'Val Position Accuracy: {val_accuracy_pos * 100:.2f}%')
    print(f'Val Orientation Accuracy: {val_accuracy_ori * 100:.2f}%')
    print()

    acc_list_val_pos.append(val_accuracy_pos)
    acc_list_val_ori.append(val_accuracy_ori)

    if (val_accuracy_pos >= 0.95) and (val_accuracy_ori >= 0.97):
        torch.save(model, 'MagResNet20_20240313.pth')
        break


plt.plot(acc_list_val_pos)
plt.plot(acc_list_val_ori)
plt.xlabel('Epoch')
plt.ylabel('Accuracy On val_dataset')
    #plt.pause(0.1)  # 留出一点时间给图形窗口进行更新

# 关闭交互式绘图模式
#plt.ioff()
plt.show()







# Testing
# model = torch.load('MagResNet20_7.pth')
model.eval()
test_loss = 0.0
test_accuracy_pos = 0.0
test_accuracy_ori = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        test_accuracy_pos += calculate_accuracy(outputs, labels)[0]
        test_accuracy_ori += calculate_accuracy(outputs, labels)[1]

test_loss /= len(test_loader)
print()
print(f'Test Loss: {test_loss}')

test_accuracy_pos /= len(test_loader)
test_accuracy_ori /= len(test_loader)
print(f'Test Position Accuracy: {test_accuracy_pos * 100:.2f}%')
print(f'Test Orientation Accuracy: {test_accuracy_ori * 100:.2f}%')

