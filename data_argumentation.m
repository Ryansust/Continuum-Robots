clc; clear; close all;

%% 1. 加载原始数据
disp('正在加载原始数据...');
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% --- 提取输入 (6个肌腱力) ---
% 假设在 5-10 列
raw_forces = table2array(dataTable(3:end, 5:10)); 

% --- 提取输出 (所有Marker点坐标) ---
% 假设在 29 列，格式为字符串
position_text_array = dataTable{3:end, 29};
num_samples = size(raw_forces, 1);
num_markers = 9; % 假设有9个Marker

% 解析Marker数据
raw_positions = zeros(num_samples, num_markers * 3); % [N x 27]

for i = 1:num_samples
    markers_3x9 = get_RealOffset_1S3CT(position_text_array{i});
    raw_positions(i, :) = reshape(markers_3x9, 1, []);
end

fprintf('原始样本数: %d\n', num_samples);

%% 2. 旋转增强 (Rotational Augmentation)
disp('正在执行旋转增强 (120度和240度)...');

% === 准备工作 ===
% 定义旋转角度
theta1 = 120;
theta2 = 240;

% 定义旋转矩阵 (绕Z轴)
Rz_120 = [cosd(theta1), -sind(theta1), 0;
          sind(theta1),  cosd(theta1), 0;
          0,             0,            1];

Rz_240 = [cosd(theta2), -sind(theta2), 0;
          sind(theta2),  cosd(theta2), 0;
          0,             0,            1];

% 初始化增强数据容器
aug_forces_120 = zeros(size(raw_forces));
aug_positions_120 = zeros(size(raw_positions));

aug_forces_240 = zeros(size(raw_forces));
aug_positions_240 = zeros(size(raw_positions));

% === 第一组：旋转 120 度 ===
% 1. 力轮换 (Permutation): 
aug_forces_120 = raw_forces(:,[5,6,1,2,3,4]);

% 2. 坐标旋转 (Rotation)
for i = 1:num_samples
    current_pos = reshape(raw_positions(i, :), 3, num_markers);
    rotated_pos = Rz_120 * current_pos;
    aug_positions_120(i, :) = reshape(rotated_pos, 1, []);
end

% === 第二组：旋转 240 度 (相当于 -120 度) ===
% 1. 力轮换 (Permutation):
aug_forces_240 = raw_forces(:,[3,4,5,6,1,2]);
% 2. 坐标旋转 (Rotation)
for i = 1:num_samples
    current_pos = reshape(raw_positions(i, :), 3, num_markers);
    rotated_pos = Rz_240 * current_pos;
    aug_positions_240(i, :) = reshape(rotated_pos, 1, []);
end

%% 3. 合并所有数据
disp('正在合并数据...');

% 最终训练集 = 原始 + 120度版 + 240度版
force_after_final_inputs = [raw_forces; aug_forces_120; aug_forces_240];
position_final_targets = [raw_positions; aug_positions_120; aug_positions_240];

fprintf('增强后样本数: %d (扩大了 3 倍)\n', size(force_after_final_inputs, 1));

%% 4. 验证与可视化 (抽样画在一张图)
% 随机抽取一个样本进行验证
idx = randi(num_samples); % 随机选一个索引
% idx = 10; % 或者指定一个索引

% 提取三种形态的坐标
xyz_raw = reshape(raw_positions(idx,:), 3, []);
xyz_120 = reshape(aug_positions_120(idx,:), 3, []);
xyz_240 = reshape(aug_positions_240(idx,:), 3, []);
raw_forces(idx,:)
aug_forces_120(idx,:)
aug_forces_240(idx,:)
% 绘图
figure('Color', 'white', 'Name', 'Rotation Augmentation Verify');
hold on; grid on; axis equal;

% 画原始形态 (蓝色)
plot3(xyz_raw(1,:), xyz_raw(2,:), xyz_raw(3,:), 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'Original (0°)');
% 画120度形态 (红色)
plot3(xyz_120(1,:), xyz_120(2,:), xyz_120(3,:), 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'Rotated (120°)');
% 画240度形态 (绿色)
plot3(xyz_240(1,:), xyz_240(2,:), xyz_240(3,:), 'g-o', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'DisplayName', 'Rotated (240°)');

% 添加坐标系参考
quiver3(0,0,0, 0.05,0,0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off'); % X轴
quiver3(0,0,0, 0,0.05,0, 'k', 'LineWidth', 1, 'HandleVisibility', 'off'); % Y轴
quiver3(0,0,0, 0,0,0.05, 'k', 'LineWidth', 1, 'HandleVisibility', 'off'); % Z轴

title(['Sample ID: ', num2str(idx), ' - Symmetry Verification']);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
legend('Location', 'best');
view(45, 30); % 调整视角以便观察
hold off;

%% 5. 保存
save('augmented_dataset_3x.mat', 'force_after_final_inputs', 'position_final_targets');
disp('数据已保存为 augmented_dataset_3x.mat');