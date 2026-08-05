%% ========================================================================
%  Project: 基于深度残差学习的连续体机器人本体感知系统
%  Module:  Net_B (Final Brutal Safety Fix)
% =========================================================================
clc; clear; close all;
rng('default');

% %% === 1. 数据读取与位姿解析 ===
% disp('--------------------------------------------------');
% disp('1. 正在读取原始数据...');
% FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
% dataTable = readtable(FileName);
% 
% % 提取原始数据 (全部转double)
% F_after  = double(table2array(dataTable(3:end, 23:28)))';  
% F_before = double(table2array(dataTable(3:end, 11:16)))';  
% F_diff   = F_after - F_before;                     
% 
% raw_mag  = double(abs(table2array(dataTable(3:end, 2))))'; 
% raw_dir  = double(table2array(dataTable(3:end, 3)))';      
% raw_hgt  = double(table2array(dataTable(3:end, 4)))';      
% pos_text = dataTable{3:end, 38}; % 位姿字符串
%% === 1. 数据读取与 ROI 筛选 ===
disp('--------------------------------------------------');
disp('1. 正在读取原始数据...');
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% 提取原始数据
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))';  
F_before_raw = double(table2array(dataTable(3:end, 11:16)))';  
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; % 接触位置 (1-9)
pos_text_raw = dataTable{3:end, 38}; 

%% === 1.0 [核心战术] 聚焦中段：只保留 3, 4, 5 节 ===
disp('   > 正在执行 ROI 筛选 (保留 Node 3, 4, 5)...');

% 定义我们关心的节点
target_nodes = [3, 4, 5];

% 生成保留掩码 (只保留高度为 3, 4, 5 的样本)
roi_mask = ismember(raw_hgt_raw, target_nodes);

% 应用筛选
F_after_raw  = F_after_raw(:, roi_mask);
F_before_raw = F_before_raw(:, roi_mask);
raw_mag_raw  = raw_mag_raw(roi_mask);
raw_dir_raw  = raw_dir_raw(roi_mask);
raw_hgt_raw  = raw_hgt_raw(roi_mask);
pos_text_raw = pos_text_raw(roi_mask);

fprintf('   > ROI 筛选完成。保留样本数: %d (丢弃了非核心区数据)\n', length(raw_mag_raw));

if length(raw_mag_raw) < 50
    error('筛选后样本太少！请确认 Excel 中第 4 列确实包含 3,4,5 这些数值。');
end


%% === 1.1 数据清洗 (剔除 NaN/Inf) ===
disp('   > 正在检查并剔除无效样本...');

% 找出任何包含 NaN 的列索引
bad_idx = any(isnan(F_after_raw), 1) | any(isnan(F_before_raw), 1) | ...
          isnan(raw_mag_raw) | isnan(raw_dir_raw) | isnan(raw_hgt_raw);

% 剔除坏数据
F_after  = F_after_raw(:, ~bad_idx);
F_before = F_before_raw(:, ~bad_idx);
raw_mag  = raw_mag_raw(~bad_idx);
raw_dir  = raw_dir_raw(~bad_idx);
raw_hgt  = raw_hgt_raw(~bad_idx);
pos_text = pos_text_raw(~bad_idx); 

F_diff = F_after - F_before;
N = length(raw_mag);

fprintf('   > 最终有效样本: %d\n', N);
N = length(raw_mag);

% --- 位姿特征提取 ---
disp('   > 正在解析位姿数据 (P_before)...');
P_before = zeros(21, N); 
for i = 1:N
    % 这里可能会产生 NaN，之前漏查了这里
    real_offset = get_RealOffset_1S3CT(pos_text{i});
    body_markers = real_offset(:, 3:end); 
    P_before(:, i) = reshape(body_markers, [], 1); 
end

% --- 计算外力矢量真值 ---
gt_F_vec = zeros(3, N);
for i = 1:N
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
        otherwise, u_vec = [0;0;0]; % 防止未知方向导致NaN
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

%% === 2. 数据增强 ===
disp('--------------------------------------------------');
disp('2. 正在执行旋转增强...');
[aug_F_diff, aug_F_after, aug_F_before, aug_P_before, aug_gt_F, aug_hgt] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F_vec, raw_hgt);

%% === 3. [绝对核心修复] 暴力安检门 ===
% 无论之前的步骤发生了什么，这里进行最终清洗。
% 只要数据要进网络，必须先过这一关。
disp('--------------------------------------------------');
disp('3. 执行训练前最终暴力安检 (Final Safety Check)...');

% 3.1 准备 Force 网络数据
inputs_f_final  = [aug_F_after; aug_F_diff; aug_F_before];
targets_f_final = aug_gt_F;

% 3.2 准备 Location 网络数据
inputs_loc_final = [aug_F_diff; aug_F_after; aug_P_before]; 
targets_loc_final = double(aug_hgt) / 9.0; % 归一化

% 3.3 检查 NaN / Inf
bad_idx_f = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(targets_f_final), 1) | any(isinf(targets_f_final), 1);
        
bad_idx_l = any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1) | ...
            any(isnan(targets_loc_final), 1) | any(isinf(targets_loc_final), 1);

bad_idx_total = bad_idx_f | bad_idx_l;

if sum(bad_idx_total) > 0
    fprintf('   【警报】发现 %d 组坏数据！正在强制剔除...\n', sum(bad_idx_total));
    inputs_f_final(:, bad_idx_total) = [];
    targets_f_final(:, bad_idx_total) = [];
    inputs_loc_final(:, bad_idx_total) = [];
    targets_loc_final(:, bad_idx_total) = [];
    % 同时也剔除 aug_gt_F 用于后续筛选
    aug_gt_F(:, bad_idx_total) = []; 
else
    disp('   > 数据完整性检查通过 (无NaN/Inf)。');
end

% 3.4 注入微量噪声 (防止全零行导致 mapminmax 崩溃)
% 之前的噪声加早了，这里加最保险
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));

inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));
% target_loc 不加噪声，保持纯净

% 3.5 再次检查数据量
if isempty(inputs_f_final)
    error('错误：所有数据均被剔除，请检查原始数据源！');
end
fprintf('   > 最终入网样本数: %d\n', size(inputs_f_final, 2));


%% === 4. 训练 Net_B_Force ===
disp('--------------------------------------------------');
disp('4. 正在训练 Net_B_Force...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;

% 这一步如果报错，说明数据里还有鬼，或者 MATLAB 环境问题
[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% 验证是否 NaN
pred_f_test = net_force(inputs_f_final(:, tr_f.testInd));
targ_f_test = targets_f_final(:, tr_f.testInd);

if any(isnan(pred_f_test(:)))
    warning('Net_B_Force 输出了 NaN。尝试降低学习率或更换算法。');
    mae_f = NaN;
else
    mae_f = mean(abs(sqrt(sum(pred_f_test.^2)) - sqrt(sum(targ_f_test.^2))));
    fprintf('   > Force MAE: %.4f N\n', mae_f);
end


%% === 4. [物理加权版] 训练 Net_B_Location (Weighted Loss) ===
disp('--------------------------------------------------');
disp('4. 正在训练 Net_B_Location (使用加权Loss函数优化)...');

% 1. 准备原始数据 (不含任何噪声)
min_force_threshold = 0.08; 
v_mask = sqrt(sum(aug_gt_F.^2)) > min_force_threshold;

raw_in = inputs_loc_final(:, v_mask);
raw_tg = targets_loc_final(:, v_mask); % 归一化后的目标

% 2. 计算每个样本属于哪一节 (用于计算权重)
% 还原出物理节数 (3, 4, 5)
node_labels = round(raw_tg * 9.0); 

% 3. [核心] 设计加权 Loss 函数 (Inverse Class Frequency)
% 统计每类样本数
nodes_of_interest = [3, 4, 5];
num_samples = length(node_labels);
weights_vector = ones(1, num_samples); % 初始化权重向量

fprintf('   > 样本分布与权重计算:\n');
for k = nodes_of_interest
    % 找到属于第 k 节的所有样本
    idx_k = (node_labels == k);
    count_k = sum(idx_k);
    
    if count_k > 0
        % 权重公式：总样本数 / (类别数 * 该类样本数)
        % 这样所有类别的总权重贡献是相等的
        w_k = num_samples / (length(nodes_of_interest) * count_k);
        weights_vector(idx_k) = w_k;
        
        fprintf('     - 第 %d 节: %d 个样本 -> 权重设为 %.2f\n', k, count_k, w_k);
    end
end

% 4. Z-Score 归一化 (输入输出都做，加速收敛)
[in_norm, ps_in] = mapstd(raw_in); 
[tg_norm, ps_out] = mapstd(raw_tg);

% 5. 网络配置
net_loc = fitnet([60, 40, 20]); % 经典的塔式结构

net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = true; 
net_loc.trainParam.epochs = 1500;
net_loc.trainParam.max_fail = 20; 
net_loc.trainParam.goal = 1e-6; % 追求更高精度

% 划分
net_loc.divideParam.trainRatio = 0.80;
net_loc.divideParam.valRatio   = 0.20;
net_loc.divideParam.testRatio  = 0.0; % 手动测试

% 6. [关键] 带权重的训练
% 最后一个参数 weights_vector 就是告诉 MATLAB 修改 Loss 函数
% 让它对第 3、5 节的误差极其敏感
disp('   > 开始训练 (Loss函数已修正)...');
[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [], [], weights_vector);


%% === 5. 真实评估与可视化 ===
disp('--------------------------------------------------');
disp('5. 评估模型性能...');

% 使用全部数据进行回测 (或手动划分的测试集)
test_in = raw_in;
test_tg = raw_tg;

% 1. 预测
test_in_norm = mapstd('apply', test_in, ps_in);
pred_norm = net_loc(test_in_norm);
pred_val = mapstd('reverse', pred_norm, ps_out);

% 2. 还原物理量
pred_node = pred_val * 9.0;
real_node = test_tg * 9.0;

% 3. 严格边界约束 (只允许 3, 4, 5)
% 既然我们只关注 ROI，就把预测值强行拉回这个区间，过滤掉离谱的偏差
pred_node(pred_node < 3) = 3;
pred_node(pred_node > 5) = 5;

% 4. 计算指标
rmse_node = sqrt(mean((pred_node - real_node).^2));
acc_strict = sum(round(pred_node) == round(real_node)) / length(real_node);

fprintf('   > [最终结果] RMSE: %.2f 节\n', rmse_node);
fprintf('   > [最终结果] 严格准确率: %.2f%%\n', acc_strict * 100);

% 5. 绘图
figure('Name', 'Weighted Loss Result', 'Color', 'w', 'Position', [100, 100, 1200, 500]);

% 散点图
subplot(1, 2, 1);
jitter = (rand(size(pred_node))-0.5)*0.15;
scatter(real_node, pred_node+jitter, 30, abs(real_node-pred_node), 'filled', 'MarkerFaceAlpha', 0.7);
colormap(jet); caxis([0 1]); colorbar;
hold on; plot([2, 6], [2, 6], 'k--', 'LineWidth', 1.5);
title(['Regression (RMSE: ', num2str(rmse_node, '%.2f'), ')']);
xlabel('Truth (Node)'); ylabel('Prediction (Node)');
grid on; axis([2.5 5.5 2.5 5.5]); xticks([3 4 5]); yticks([3 4 5]);

% 混淆矩阵
subplot(1, 2, 2);
test_pred_class = round(pred_node);
test_real_class = round(real_node);
cm = confusionchart(test_real_class, test_pred_class);
cm.Title = 'Confusion Matrix (Weighted)';
cm.RowSummary = 'row-normalized'; % 关注这里的召回率
sortClasses(cm, 'ascending');

disp('运行结束。请观察第3节和第5节的识别率是否大幅提升。');

%% netc
%% ========================================================================
%  Project: 终极形态 —— Net_C (全形态三维重构)
%  Goal:    融合 Net_B 的感知结果，完美预测机器人受力后的 3D 骨架
% =========================================================================

disp('==================================================');
disp('   🚀 进入最终阶段：Net C 形态重构网络的设计与训练');
disp('==================================================');

%% === 1. 准备 Net_C 的输入数据 (独立提取) ===
disp('1. 构建级联特征 (Feature Fusion)...');

% 为了防止变量名报错，我们直接从【第3步：暴力安检】的全局变量中提取
% 必须保证 inputs_loc_final, targets_f_final, targets_loc_final 存在
if ~exist('inputs_loc_final', 'var')
    error('缺少全局变量 inputs_loc_final，请先运行第 1-3 步！');
end

% 1. 重新应用第 4 步的筛选掩码 (只取有力交互的数据)
min_force_threshold = 0.08; 
v_mask = sqrt(sum(targets_f_final.^2)) > min_force_threshold;

% 2. 从全局池中提取数据
% inputs_loc_final 结构: [F_diff(1-6); F_after(7-12); P_before(13-33)]
data_pool = inputs_loc_final(:, v_mask);

% --- 提取 Net C 所需的组件 ---
% A. 内部肌腱力 (F_after) -> 取第 7-12 行
feat_internal = data_pool(7:12, :);

% B. 外部交互力 (F_ext) -> 取 targets_f_final
feat_external = targets_f_final(:, v_mask);

% C. 接触位置 (Location) -> 取 targets_loc_final
feat_location = targets_loc_final(:, v_mask);

% --- 核心：构建 Net C 输入向量 (10维) ---
% Input = [肌腱力(6) + 外力矢量(3) + 接触位置(1)]
inputs_net_c = [feat_internal; feat_external; feat_location];

% --- 核心：构建 Net C 目标向量 (21维) ---
% Target = [真实形态坐标 (7个点 x 3)] -> 取 data_pool 的第 13-33 行
targets_net_c = data_pool(13:33, :);

fprintf('   > [数据构建成功]\n');
fprintf('   > Net_C 输入维度: %d (力+感知信息)\n', size(inputs_net_c, 1));
fprintf('   > Net_C 输出维度: %d (3D形态坐标)\n', size(targets_net_c, 1));
fprintf('   > 样本数量: %d\n', size(inputs_net_c, 2));


%% === 2. 训练 Net C (Shape Reconstruction Network) ===
disp('--------------------------------------------------');
disp('2. 正在训练 Net_C ...');

% 2.1 Z-Score 归一化 (关键步骤)
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

% 2.2 网络设计 (深层感知网络)
% 输入10维 -> 输出21维，是一个复杂的非线性回归
net_shape = fitnet([80, 60, 40]); 

net_shape.trainFcn = 'trainscg';
net_shape.trainParam.showWindow = true;
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-7; 
net_shape.trainParam.max_fail = 50;

% 划分数据集
net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio   = 0.1;
net_shape.divideParam.testRatio  = 0.1;

% 2.3 训练
[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);


%% === 3. 全系统联合测试 (Visualization) ===
disp('--------------------------------------------------');
disp('3. 执行全系统闭环测试与可视化...');

% 选取测试集索引
test_idx = tr_c.testInd;
if isempty(test_idx)
    % 如果自动划分没分到，手动随机取50个
    test_idx = randperm(size(inputs_net_c,2), 50); 
end

% 1. 预测
in_test = inputs_net_c(:, test_idx);
target_test = targets_net_c(:, test_idx);

in_test_norm = mapstd('apply', in_test, ps_in_c);
pred_test_norm = net_shape(in_test_norm);
pred_test = mapstd('reverse', pred_test_norm, ps_out_c);

% 2. 计算平均误差 (Mean Reconstruction Error)
% 计算所有点、所有样本的平均欧式距离
err_abs = abs(pred_test - target_test);
mre_total = mean(err_abs(:)); % 简单平均 (m)

% 更精确的距离误差 (Distance Error)
dist_errs = [];
for i = 1:length(test_idx)
    p_pred = reshape(pred_test(:, i), 3, []);
    p_real = reshape(target_test(:, i), 3, []);
    dist = sqrt(sum((p_pred - p_real).^2, 1)); % 每个点的距离
    dist_errs = [dist_errs, mean(dist)];
end
mean_dist_err = mean(dist_errs);

fprintf('   > [Net C 最终精度] 平均形态误差: %.4f m (%.2f mm)\n', mean_dist_err, mean_dist_err*1000);


%% === 4. 终极可视化：3D 骨架对比 ===
disp('--------------------------------------------------');
disp('4. 生成 3D 机器人形态对比图...');

figure('Name', 'Robot 3D Shape Reconstruction', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

% 随机抽取 4 个样本
num_plot = 3;
plot_indices = test_idx(randperm(length(test_idx), num_plot));

for k = 1:num_plot
    idx = plot_indices(k);
    
    % 提取坐标
    P_pred = reshape(pred_test(:, find(test_idx==idx)), 3, []);
    P_real = reshape(target_test(:, find(test_idx==idx)), 3, []);
    
    % 加基座原点 (0,0,0) 用于画图
    P_pred = [[0;0;0], P_pred];
    P_real = [[0;0;0], P_real];
    
    % 绘图
    subplot(1, num_plot, k);
    
    % 真实骨架 (黑实线)
    plot3(P_real(1,:), P_real(2,:), P_real(3,:), 'k-o', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'k');
    hold on;
    % 预测骨架 (红虚线)
    plot3(P_pred(1,:), P_pred(2,:), P_pred(3,:), 'r--.', 'LineWidth', 1.5, 'MarkerSize', 10);
    
    grid on; axis equal;
    xlabel('X (m)'); zlabel('Z (m)'); % 俯视图或侧视图可能更清晰
    title(['Sample ', num2str(k)]);
    
    if k==1, legend('Truth', 'Reconstruction'); end
    view(30, 20); % 调整视角
end

disp('🎉 Net C 训练完成！请查看 3D 对比图。');
%% === 5. [修正版] 末端(Tip) 专项误差分析 ===
disp('--------------------------------------------------');
disp('5. 正在计算末端(Tip)位姿的独立误差...');

% 1. 提取数据
% 我们的输出是 21 维 (7个点 * 3坐标)
% Tip 点是第 7 个点，对应索引 19, 20, 21
tip_indices = [19, 20, 21];

tip_pred = pred_test(tip_indices, :);   % [3 x N_test]
tip_real = target_test(tip_indices, :); % [3 x N_test]

% 2. 计算误差 (欧氏距离)
tip_err_vec = tip_pred - tip_real;
% 【变量名修正】这里定义的是 tip_err_dist
tip_err_dist = sqrt(sum(tip_err_vec.^2, 1)); 

% 3. 统计指标
% 【变量名修正】下面全部统一使用 tip_err_dist，不会再报错了
tip_mae = mean(tip_err_dist);
tip_rmse = sqrt(mean(tip_err_dist.^2));
tip_max = max(tip_err_dist);

fprintf('   > [Tip 专项] 平均误差 (MAE):  %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);
fprintf('   > [Tip 专项] 均方根误差 (RMSE): %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip 专项] 最大误差 (Max):    %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

% 4. 可视化：末端追踪效果图
figure('Name', 'Tip Tracking Performance', 'Color', 'w', 'Position', [100, 200, 1000, 400]);

% 子图1：空间追踪对比 (随机抽样50个点，画连线)
subplot(1, 2, 1);
num_show = min(50, length(tip_err_dist));
idx_show = randperm(length(tip_err_dist), num_show);

hold on; grid on; axis equal;
h1 = plot3(NaN,NaN,NaN, 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b'); % 占位句柄
h2 = plot3(NaN,NaN,NaN, 'r.', 'MarkerSize', 10); % 占位句柄

for k = idx_show
    % 画一条灰线连接真值和预测值，直观展示偏差
    p_r = tip_real(:, k);
    p_p = tip_pred(:, k);
    plot3([p_r(1), p_p(1)], [p_r(2), p_p(2)], [p_r(3), p_p(3)], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
    
    % 画点
    plot3(p_r(1), p_r(2), p_r(3), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b'); % 真值
    plot3(p_p(1), p_p(2), p_p(3), 'r.', 'MarkerSize', 10); % 预测
end
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Tip Position Tracking (Blue=True, Red=Pred)');
legend([h1, h2], {'Ground Truth', 'Prediction'}, 'Location', 'best');
view(45, 30);

% 子图2：误差分布直方图
subplot(1, 2, 2);
histogram(tip_err_dist * 1000, 30, 'FaceColor', [0.2 0.6 0.3]);
xline(tip_mae * 1000, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f mm', tip_mae*1000));
xlabel('Tip Position Error (mm)');
ylabel('Sample Count');
title('Tip Error Distribution');
grid on;


%% === 旋转函数 ===
function [aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_gF, aug_h] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F, hgt)
    
    N = size(F_diff, 2);
    R120 = [cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 = [cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    idx120 = [5, 6, 1, 2, 3, 4];
    idx240 = [3, 4, 5, 6, 1, 2];
    
    Fd_120 = F_diff(idx120, :); Fa_120 = F_after(idx120, :); Fb_120 = F_before(idx120, :);
    gF_120 = R120 * gt_F;
    P_tmp = reshape(P_before, 3, []);
    P_120 = reshape(R120 * P_tmp, 21, N);
    
    Fd_240 = F_diff(idx240, :); Fa_240 = F_after(idx240, :); Fb_240 = F_before(idx240, :);
    gF_240 = R240 * gt_F;
    P_240 = reshape(R240 * P_tmp, 21, N);
    
    aug_Fd = [F_diff, Fd_120, Fd_240];
    aug_Fa = [F_after, Fa_120, Fa_240];
    aug_Fb = [F_before, Fb_120, Fb_240];
    aug_Pb = [P_before, P_120, P_240];
    aug_gF = [gt_F, gF_120, gF_240];
    aug_h  = [hgt, hgt, hgt];
end