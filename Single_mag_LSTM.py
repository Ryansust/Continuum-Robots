import re
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

# 超参数
input_size = 16
#input_size = 36
hidden_size = 16
num_layers = 8
output_size = 6
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=2)  # Initial Conv layer 这里出来是6*6=36
        # self.bn1 = nn.BatchNorm2d(16)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_position = nn.Linear(hidden_size, output_size - 3)  # Output for the main part
        self.fc_orientation = nn.Linear(hidden_size, 3)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)
        #
        # ###########################
        # x = x.view(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        ###########################
        # 思考一下加一个求平均值
        # lstm_out = self.avgpool(lstm_out)
        # lstm_out = lstm_out.view(lstm_out.size(0), -1)


        position = self.fc_position(lstm_out)
        orientation = torch.tanh(self.fc_orientation(lstm_out))

        prediction = torch.cat((position, orientation), dim=1)

        return prediction



# train_data = torch.randn(700, 3, 4, 4)
# train_labels = torch.randn(700, 6)
#
# val_data = torch.randn(100, 3, 4, 4)
# val_labels = torch.randn(100, 6)
#
# test_data = torch.randn(200, 3, 4, 4)
# test_labels = torch.randn(200, 6)


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

mag_data = get_data("normalized_processed_data_for_train_202401.csv")
mag_label = get_labels("normalized_processed_data_for_train_202401.csv")


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
train_data = train_data.astype(np.float32)
train_labels = np.stack(train_labels, axis = 0)
train_data = torch.from_numpy(train_data)
train_labels = torch.from_numpy(train_labels)

for i in index_val:
    val_data.append(mag_data[i])
    val_labels.append(mag_label[i])
val_data = np.stack(val_data, axis=0)
val_data = val_data.astype(np.float32)
val_labels = np.stack(val_labels, axis=0)
val_data = torch.from_numpy(val_data)
val_labels = torch.from_numpy(val_labels)

for i in index_test:
    test_data.append(mag_data[i])
    test_labels.append(mag_label[i])
test_data = np.stack(test_data, axis=0)
test_data = test_data.astype(np.float32)
test_labels = np.stack(test_labels, axis=0)
test_data = torch.from_numpy(test_data)
test_labels = torch.from_numpy(test_labels)


# 将数据从四维张量整形成三维张量
train_data = train_data.view(train_data.size(0), 3, -1)  # (700, 3, 16)
val_data = val_data.view(val_data.size(0), 3, -1)
test_data = test_data.view(test_data.size(0), 3, -1)

#print(train_data.shape) #torch.Size([1587, 3, 4, 4])

# 转换为 DataLoader
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def calculate_accuracy(outputs, labels, threshold_position=0.004, threshold_orientation=5):  #要改
    #correct_predictions = ((torch.abs(outputs - labels) <= threshold).all(dim=1)).sum().item()
    total_samples = labels.size(0)
    position_pred = outputs[:, :3]
    orientation_pred = outputs[:, 3:]
    position_labels = labels[:, :3]
    orientation_labels = labels[:, 3:]
    #print('outputs', outputs)
    #print('labels', labels)
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


# 创建模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
acc_list_val_pos = []
acc_list_val_ori = []
epoch = 0
#for epoch in range(num_epochs):
while True:
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在验证集上计算损失
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_accuracy_pos = 0.0
        val_accuracy_ori = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_accuracy_pos += calculate_accuracy(outputs, labels)[0]
            val_accuracy_ori += calculate_accuracy(outputs, labels)[1]
    #print(outputs)
    val_loss /= len(val_loader)
    epoch += 1
    print(f'Epoch {epoch}, Loss: {val_loss}')
    val_accuracy_pos /= len(val_loader)
    val_accuracy_ori /= len(val_loader)
    print(f'Val Position Accuracy: {val_accuracy_pos * 100:.2f}%')
    print(f'Val Orientation Accuracy: {val_accuracy_ori * 100:.2f}%')
    print()

    acc_list_val_pos.append(val_accuracy_pos)
    acc_list_val_ori.append(val_accuracy_ori)

    if (val_accuracy_pos >= 0.95) and (val_accuracy_ori >= 0.97):
        torch.save(model, 'Single_mag_LSTM_1.pth')
        break

plt.plot(acc_list_val_pos)
plt.plot(acc_list_val_ori)
plt.xlabel('Epoch')
plt.ylabel('Accuracy On val_dataset')
    #plt.pause(0.1)  # 留出一点时间给图形窗口进行更新

# 关闭交互式绘图模式
#plt.ioff()
plt.show()

# 测试模型
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

# 模型一直不收敛