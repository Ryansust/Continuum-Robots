%% ========================================================================
%  Dual-Net Architecture for External Interaction Sensing
%  Net_B_Force (Regression) + Net_B_Height (Classification)
% =========================================================================
clc; clear; close all;
rng('default');

%% === 1. 全局数据读取与预处理 ===
disp('正在读取数据...');
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% 提取原始数据
F_after = table2array(dataTable(3:end, 23:28))';  % [6 x N]
F_before = table2array(dataTable(3:end, 11:16))'; % [6 x N]
F_diff = F_after - F_before;

raw_mag = abs(table2array(dataTable(3:end, 2)))'; % 外力大小
raw_dir = table2array(dataTable(3:end, 3))';      % 方向编码
raw_hgt = table2array(dataTable(3:end, 4))';      % 高度编码 (1-9)

N = length(raw_mag);

% --- 计算真实力向量 (Ground Truth Force Vector) ---
gt_F_vec = zeros(3, N);
valid_directions = [-1, 0, 0; -sind(45), cosd(45), 0; 0, 1, 0]'; % 标准方向库

for i = 1:N
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];         % -X
        case 3, u_vec = [-sind(45); cosd(45); 0]; % 斜向
        case 4, u_vec = [0; 1; 0];          % +Y
        % 确保这里包含了你所有可能的方向 Case
        otherwise, u_vec = [0;0;0];
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

fprintf('数据加载完成，共 %d 组样本。\n', N);


%% ========================================================================
%  PART A: Net_B_Force (外力矢量回归)
%  目标：精准预测力的大小和方向
% =========================================================================
disp('--------------------------------------------------');
disp('正在训练 Net_B_Force (回归模型)...');

% 1. 输入构建 (18维)
inputs_force = [F_after; F_diff; F_before]; 
targets_force = gt_F_vec; % 输出是 3维 向量

% 2. 划分与归一化
cv = cvpartition(N, 'HoldOut', 0.2);
train_in_f = inputs_force(:, cv.training);
train_tg_f = targets_force(:, cv.training);
test_in_f  = inputs_force(:, cv.test);
test_tg_f  = targets_force(:, cv.test);

[train_in_f_n, ps_in_f] = mapminmax(train_in_f);
[train_tg_f_n, ps_out_f] = mapminmax(train_tg_f);

% 3. 网络训练 (使用 trainlm 回归最强算法)
net_force = feedforwardnet([30, 15]); 
net_force.trainFcn = 'trainlm';
net_force.trainParam.epochs = 800;
net_force.trainParam.showWindow = false; % 静默训练
net_force.divideParam.trainRatio = 0.8;
net_force.divideParam.valRatio = 0.2;
net_force.divideParam.testRatio = 0.0; % 我们手动留了测试集

[net_force, ~] = train(net_force, train_in_f_n, train_tg_f_n);

% 4. 预测与评估
test_in_f_n = mapminmax('apply', test_in_f, ps_in_f);
pred_f_n = net_force(test_in_f_n);
pred_f_real = mapminmax('reverse', pred_f_n, ps_out_f);

% --- 核心技巧：Snap-to-Grid (方向吸附) ---
[~, N_test] = size(pred_f_real);
pred_f_corrected = zeros(3, N_test);

for i = 1:N_test
    v = pred_f_real(:, i);
    m = norm(v);
    if m < 0.005 % 滤除微小噪声
        pred_f_corrected(:, i) = [0;0;0];
        continue;
    end
    % 找最接近的方向
    scores = valid_directions' * (v / m);
    [~, best_idx] = max(scores);
    % 修正方向，保留模长
    pred_f_corrected(:, i) = m * valid_directions(:, best_idx);
end

% 5. 误差计算
err_mag = abs(sqrt(sum(pred_f_corrected.^2)) - sqrt(sum(test_tg_f.^2)));
fprintf('>> [Net_B_Force] 外力大小 MAE: %.4f N\n', mean(err_mag));%% === 专项评估：力的方向误差分析 ===
% 假设变量名：
% F_real = test_tg_f;      % 真实力向量 [3 x N]
% F_pred = pred_f_corrected; % 预测力向量 (建议使用吸附后的结果) [3 x N]

% 为了通用性，先重命名一下（你可以直接用上面的变量名）
F_real_eval = test_tg_f; 
F_pred_eval = pred_f_corrected; 

% 1. 计算模长
mag_real = sqrt(sum(F_real_eval.^2, 1));
mag_pred = sqrt(sum(F_pred_eval.^2, 1));

% 2. 计算点积 (Dot Product)
dot_prod = sum(F_real_eval .* F_pred_eval, 1);

% 3. 计算夹角余弦 (Cosine Similarity)
% 公式: cos(theta) = (A . B) / (|A| * |B|)
denominator = mag_real .* mag_pred;

% 【重要】防止除以零：如果真实力或预测力几乎为0，方向是无意义的
valid_idx = (mag_real > 0.05) & (mag_pred > 0.05); % 只分析力大于 0.05N 的样本
if sum(valid_idx) == 0
    warning('没有有效的外力样本用于计算方向误差！');
else
    % 仅提取有效数据
    dot_prod = dot_prod(valid_idx);
    denominator = denominator(valid_idx);
    
    cos_theta = dot_prod ./ denominator;
    
    % 4. 数值截断 (防止浮点误差导致 1.0000001 这种不可算 acos 的数)
    cos_theta(cos_theta > 1) = 1;
    cos_theta(cos_theta < -1) = -1;
    
    % 5. 计算角度 (度)
    angle_errors = acosd(cos_theta); % [1 x N_valid]
    
    %% === 结果打印 ===
    fprintf('\n========== 方向误差专项报告 ==========\n');
    fprintf('有效样本数: %d (剔除了微小力)\n', sum(valid_idx));
    fprintf('平均方向误差 (Mean): %.4f 度\n', mean(angle_errors));
    fprintf('最大方向误差 (Max) : %.4f 度\n', max(angle_errors));
    fprintf('方向完全正确 (误差<1度) 的比例: %.2f%%\n', sum(angle_errors < 1) / length(angle_errors) * 100);
    
    %% === 可视化分析 ===
    
    % 图 1: 误差直方图
    figure('Name', 'Angle Error Distribution', 'Color', 'w');
    histogram(angle_errors, 30); % 分30个区间
    xlabel('角度误差 (度)');
    ylabel('样本数量');
    title('力的方向误差分布直方图');
    grid on;
    % 解释：如果不使用吸附，这里应该是一个靠近0的高斯分布。
    % 如果使用了吸附，这里应该是大部分在0，少部分在90度(指错方向)或45度。
    
    % 图 2: 误差 vs 力的大小
    % 看看是不是只有小力的时候方向才不准？
    figure('Name', 'Error vs Magnitude', 'Color', 'w');
    scatter(mag_real(valid_idx), angle_errors, 20, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('真实外力大小 (N)');
    ylabel('角度误差 (度)');
    title('方向误差与力大小的关系');
    grid on;
    % 解释：通常力越大，信噪比越高，方向应该越准。
end

%% ========================================================================
%  PART B: Net_B_Height (高度分类器)
%  目标：高准确率识别接触位置 (Class 1-9)
% =========================================================================
disp('--------------------------------------------------');
disp('正在训练 Net_B_Height (分类模型)...');

% 1. 数据清洗与特征增强
% 【关键策略】只保留外力足够大的数据进行训练
% 力太小的时候，高度特征会被噪声淹没，强行训练会干扰网络
min_force_threshold = 0.08; 
valid_mask = raw_mag > min_force_threshold;

fprintf('   > 数据清洗: 剔除微小力样本 %d 个，保留有效样本 %d 个。\n', ...
    sum(~valid_mask), sum(valid_mask));

% 提取有效数据
valid_F_after = F_after(:, valid_mask);
valid_F_diff  = F_diff(:, valid_mask);
valid_targets = raw_hgt(valid_mask);

% 【关键特征】归一化力差值 (提取力的"形状"而非"大小")
diff_norms = sqrt(sum(valid_F_diff.^2, 1));
valid_F_pattern = valid_F_diff ./ (diff_norms + 1e-8);

% 组合输入 (12维: 当前力 + 力模式) 
% 注：去掉了F_before，因为F_pattern已经包含了变化信息，精简输入有助于分类
inputs_height = [valid_F_after; valid_F_pattern]; 

% 目标转 One-Hot 编码
targets_height = full(ind2vec(valid_targets));

% 2. 划分数据集 (只在有效数据中划分)
cv_h = cvpartition(sum(valid_mask), 'HoldOut', 0.2);
train_in_h = inputs_height(:, cv_h.training);
train_tg_h = targets_height(:, cv_h.training);
test_in_h  = inputs_height(:, cv_h.test);
test_tg_h  = targets_height(:, cv_h.test);

% 3. 网络训练 (PatternNet + SCG算法)
% 输出层必须是 9 (对应9个类别)
net_height = patternnet([20, 10]); 
net_height.trainFcn = 'trainscg'; % 分类首选算法
net_height.performFcn = 'crossentropy'; % 交叉熵损失
net_height.trainParam.epochs = 1000;
net_height.trainParam.showWindow = false;

[net_height, tr_h] = train(net_height, train_in_h, train_tg_h);

% 4. 预测与评估
pred_scores = net_height(test_in_h);
[~, pred_classes] = max(pred_scores); % 找概率最大的类别
[~, true_classes] = max(test_tg_h);

% 5. 准确率计算
acc = sum(pred_classes == true_classes) / length(true_classes);
acc_tol = sum(abs(pred_classes - true_classes) <= 1) / length(true_classes);

fprintf('>> [Net_B_Height] 严格准确率 (Accuracy): %.2f%%\n', acc * 100);
%fprintf('>> [Net_B_Height] 容忍±1误差准确率: %.2f%%\n', acc_tol * 100);


%% ========================================================================
%  PART C: 结果可视化
% =========================================================================

% 1. 外力向量对比图
figure('Name', 'Force Regression', 'Color', 'w', 'Position', [100, 100, 500, 400]);
idx_sample = randperm(size(test_tg_f, 2), min(30, size(test_tg_f, 2)));
hold on; grid on; axis equal;
quiver3(zeros(1,length(idx_sample)), zeros(1,length(idx_sample)), zeros(1,length(idx_sample)), ...
    test_tg_f(1,idx_sample), test_tg_f(2,idx_sample), test_tg_f(3,idx_sample), ...
    'b', 'LineWidth', 2, 'DisplayName', 'Ground Truth', 'MaxHeadSize', 0.5);
quiver3(zeros(1,length(idx_sample)), zeros(1,length(idx_sample)), zeros(1,length(idx_sample)), ...
    pred_f_corrected(1,idx_sample), pred_f_corrected(2,idx_sample), pred_f_corrected(3,idx_sample), ...
    'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction (Corrected)', 'MaxHeadSize', 0.5);
xlabel('Fx'); ylabel('Fy'); zlabel('Fz');
legend; view(30, 30); title('Net B Force: 外力矢量预测');

% 2. 高度混淆矩阵
figure('Name', 'Height Classification', 'Color', 'w', 'Position', [650, 100, 500, 400]);
confusionchart(true_classes, pred_classes);
title(sprintf('Net B Height: 接触位置混淆矩阵 (Acc: %.1f%%)', acc*100));

% 保存模型
%save('dual_net_B_models.mat', 'net_force', 'ps_in_f', 'ps_out_f', 'net_height', 'min_force_threshold');
%disp('模型已保存至 dual_net_B_models.mat');