clc; clear; close all;
rng('default');

%% === Net B 数据准备 ===
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% 1. 准备输入 (X)
F_after = table2array(dataTable(3:end, 23:28))'; 
F_before = table2array(dataTable(3:end, 11:16))'; 
F_diff = F_after - F_before;

inputs_B = [F_after; F_diff; F_before]; % [18 x N]

% 2. 准备输出目标 (Y)
N = size(inputs_B, 2);
targets_B_force = zeros(3, N); 
targets_B_height = zeros(1, N); 

raw_mag = abs(table2array(dataTable(3:end, 2))); 
raw_dir = table2array(dataTable(3:end, 3)); 
raw_hgt = table2array(dataTable(3:end, 4)); 

for i = 1:N
    mag = raw_mag(i);
    d_code = raw_dir(i);
    h_code = raw_hgt(i);
    
    u_vec = [0;0;0];
    switch d_code
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
        % 你的 switch case 似乎不全，记得补全其他方向，或者检查是否有遗漏
        otherwise, u_vec = [0;0;0]; 
    end
    
    targets_B_force(:, i) = mag * u_vec;
    targets_B_height(:, i) = h_code; % 【注意】这里直接用原始高度 1-9，不要除以9
end

targets_B = [targets_B_force; targets_B_height];

%% === 3. 数据集划分与归一化 (核心修正部分) ===

% 先划分原始数据
cv = cvpartition(N, 'HoldOut', 0.2);
train_in_raw = inputs_B(:, cv.training);
train_tg_raw = targets_B(:, cv.training);
test_in_raw  = inputs_B(:, cv.test);
test_tg_raw  = targets_B(:, cv.test);

% 对训练集进行归一化，并获取参数 ps
[train_in_n, ps_in] = mapminmax(train_in_raw);
[train_tg_n, ps_out] = mapminmax(train_tg_raw);

% 【关键技巧】Loss 加权
% 高度信息在第4行。由于高度只有1-9，归一化后如果不加权，网络可能觉得它不如力向量重要。
% 我们手动把训练集中高度这一行的数值放大，强迫网络重视它。
weight_factor = 1.0; % 权重系数，可以尝试 2.0 - 5.0
train_tg_n(4, :) = train_tg_n(4, :) * weight_factor;

%% === 4. 网络训练 ===

% 网络架构：因为输入特征多且有非线性，稍微加宽一点
hiddenLayerSize = [50, 30]; 
net_B = feedforwardnet(hiddenLayerSize);

% 训练算法：小样本 (993组) 必须用 trainlm
net_B.trainFcn = 'trainlm';       

% 参数设置
net_B.trainParam.epochs = 1000;
net_B.trainParam.goal = 1e-6;
net_B.trainParam.max_fail = 20; % 早停
net_B.trainParam.showWindow = false; 
net_B.trainParam.showCommandLine = true;

% 训练
fprintf('开始训练 Net B...\n');
[net_B, tr] = train(net_B, train_in_n, train_tg_n);

%% === 5. 预测与反归一化 (Evaluation Phase) ===

% 1. 对测试集输入进行归一化 (应用训练集的参数)
test_in_n = mapminmax('apply', test_in_raw, ps_in);

% 2. 网络预测
pred_n = net_B(test_in_n);

% 3. 【关键】反向去除权重
pred_n(4, :) = pred_n(4, :) / weight_factor;

% 4. 反归一化 (还原到真实物理量纲)
pred_real = mapminmax('reverse', pred_n, ps_out);

% 5. 拆分结果
F_vec_real = test_tg_raw(1:3, :);
F_vec_pred = pred_real(1:3, :);

H_real = test_tg_raw(4, :); % 真实高度 (1-9)
H_pred = pred_real(4, :);   % 预测高度 (连续值)
valid_directions = [
    -1, 0, 0; 
    -sind(45), cosd(45), 0; 
    0, 1, 0
]'; % 转置为 [3 x 3]，每列是一个标准向量

% 2. 对每一个预测结果进行“吸附”
[~, N_test] = size(F_vec_pred);
F_vec_corrected = zeros(3, N_test);

for i = 1:N_test
    % 获取当前预测的向量
    v_curr = F_vec_pred(:, i);
    mag_curr = norm(v_curr); % 暂存预测的大小
    
    % 如果预测模长太小，说明可能是噪声，不修正或设为0
    if mag_curr < 0.005
        F_vec_corrected(:, i) = [0,0,0];
        continue;
    end
    
    % 计算当前向量与三个标准向量的夹角余弦 (Cosine Similarity)
    % Dot Product
    scores = valid_directions' * (v_curr / mag_curr);
    
    % 找到匹配度最高（分数最大）的那个方向的索引
    [~, best_idx] = max(scores);
    % 【核心修正】保持预测的大小(Magnitude)，但强制修正方向(Direction)
    F_vec_corrected(:, i) = mag_curr * valid_directions(:, best_idx);
end

% === 用修正后的向量替换原始预测，参与后续误差计算 ===
fprintf('已应用方向物理修正 (Snap-to-Grid)...\n');
F_vec_pred = F_vec_corrected; 

%% === 6. 误差分析 (Error Analysis) ===

% --- 指标 A: 力的大小误差 ---
mag_real = sqrt(sum(F_vec_real.^2, 1));
mag_pred = sqrt(sum(F_vec_pred.^2, 1));
err_mag = abs(mag_real - mag_pred);
fprintf('\n--- 评估结果 ---\n');
fprintf('外力大小平均误差 (MAE): %.3f N\n', mean(err_mag));

% --- 指标 B: 力的方向误差 ---
dot_prod = sum(F_vec_real .* F_vec_pred, 1);
norms = mag_real .* mag_pred;
% 防止除零
norms(norms < 1e-6) = 1e-6; 
cos_theta = dot_prod ./ norms;
% 防止数值误差导致 acos 越界
cos_theta(cos_theta > 1) = 1; 
cos_theta(cos_theta < -1) = -1;
angle_err_deg = acosd(cos_theta);
fprintf('外力方向平均误差: %.2f 度\n', mean(angle_err_deg));

% --- 指标 C: 高度准确率 ---
% 四舍五入取整
H_pred_round = round(H_pred);
% 限制范围
H_pred_round(H_pred_round < 1) = 1;
H_pred_round(H_pred_round > 9) = 9;

acc = sum(H_pred_round == H_real) / length(H_real);
fprintf('接触点高度预测准确率: %.2f%% (%d/%d)\n', ...
    acc * 100, sum(H_pred_round == H_real), length(H_real));

%% === 7. 图例可视化 ===

% --- 图1: 外力矢量的 3D 对比 ---
figure('Name', 'Force Vector', 'Color', 'w');
% 随机选 30 个画图，太密了看不清
num_plot = min(30, length(H_real));
idx_sample = randperm(length(H_real), num_plot);

hold on; grid on; axis equal;
quiver3(zeros(1,num_plot), zeros(1,num_plot), zeros(1,num_plot), ...
    F_vec_real(1,idx_sample), F_vec_real(2,idx_sample), F_vec_real(3,idx_sample), ...
    'b', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth', 'MaxHeadSize', 0.5);
quiver3(zeros(1,num_plot), zeros(1,num_plot), zeros(1,num_plot), ...
    F_vec_pred(1,idx_sample), F_vec_pred(2,idx_sample), F_vec_pred(3,idx_sample), ...
    'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction', 'MaxHeadSize', 0.5);
xlabel('Fx'); ylabel('Fy'); zlabel('Fz');
legend; view(30, 30);
title('外力向量对比 (随机采样)');

% --- 图2: 高度混淆矩阵 ---
figure('Name', 'Height Confusion', 'Color', 'w');
confusionchart(H_real, H_pred_round);
title(sprintf('高度混淆矩阵 (Acc: %.1f%%)', acc*100));