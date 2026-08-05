%% ========================================================================
%  Project: 基于物理特征增强的连续体机器人本体感知系统 (Net_B + Net_C)
%  Status:  逻辑严密修正版 (Input=Physics-Ideal, Target=Sensor-Actual After)
%  Author:  Lin Yongxi (Refined with Physics-Guided Logic)
% =========================================================================
clc; clear; close all;
rng('default'); 

%% ========================================================================
%  Step 1: 数据读取、ROI 筛选与清洗
% =========================================================================
disp('--------------------------------------------------');
disp('1. 正在读取原始数据并进行物理预处理...');

% 1.1 读取 Excel
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(FileName), error('未找到 Excel 文件，请检查路径！'); end
dataTable = readtable(FileName);

% 1.2 提取原始信号
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))'; % 碰撞后拉力
F_before_raw = double(table2array(dataTable(3:end, 11:16)))'; % 碰撞前拉力
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; % 外力大小真值
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';     % 外力方向代码
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))';     % 接触位置 (1-9)
pos_text_raw = dataTable{3:end, 29}; % 🌟 碰撞发生后的传感器位姿真值 (Target)

% 1.3 ROI 筛选 (保留核心受力区 Node 3, 4, 5)
disp('   > 正在执行 ROI 筛选 (保留 Node 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw, [3, 4, 5]);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_sub = pos_text_raw(roi_mask);

% 1.4 数据清洗 (剔除 NaN/Inf)
disp('   > 正在剔除无效样本...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | ...
          isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);
known_outliers = [686]; 
bad_idx(known_outliers) = true; 

F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
pos_text = pos_text_sub(~bad_idx); 

N = length(raw_mag);
F_diff = F_after - F_before;

% 1.5 物理增强：生成理想起始位姿 (Feature) 与 提取碰撞后真值 (Target)
disp('   > 正在生成物理理想位姿 (Input Feature) 并解析碰撞后真值 (Target)...');

% 物理参数设置
tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; 
L_ap = 0.0665; L_bp = 0.00; N_dp = 7;
H_listp = linspace(0.0025, 0.0025, section_p*N_dp+1);
mu_p = 0.25; delta_alphap = 0; G_loadp = 4.000 * 0.00981;

P_before_ideal = zeros(21, N); % 网络输入特征：物理模型生成的理想位姿
P_after_actual = zeros(21, N); % 网络训练目标：传感器实测的碰撞后位姿
gt_F_vec = zeros(3, N);

for i = 1:N
    % --- A. 物理模型生成理想起始形态 (P_before_ideal) ---
    Fb_raw = F_before(:, i) * 0.00981; 
    Fb_sim = [Fb_raw(5); Fb_raw(6); Fb_raw(1); Fb_raw(2); Fb_raw(3); Fb_raw(4)];
    [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon_p, section_p, D_p, E_p, L_ap, L_bp, N_dp, H_listp, mu_p, delta_alphap, G_loadp, [0;0;0], Fb_sim, 14);
    
    % 偏置 4mm 并提取 7 个躯干点
    V_local = [0; -0.004; 0]; 
    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2), P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; end
    marker_idx = round([2,4,6,8,10,12,14] * ((size(P_Theo,2)-1)/14)) + 1;
    P_before_ideal(:, i) = reshape(P_m(:, marker_idx), 21, 1); 
    
    % --- B. 解析传感器捕捉到的真实碰撞形态 (P_after_actual) ---
    real_offset_after = get_RealOffset_1S3CT(pos_text{i});
    P_after_actual(:, i) = reshape(real_offset_after(:, 3:end), 21, 1); 
    
    % --- C. 计算外力矢量真值 ---
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

fprintf('   > 数据预处理完成。有效样本: %d\n', N);

%% ========================================================================
%  Step 2: 数据增强 (旋转同步)
% =========================================================================
disp('--------------------------------------------------');
disp('2. 正在执行旋转增强 (同步处理 Feature 与 Target)...');
[aug_F_diff, aug_F_after, aug_F_before, aug_Pb_ideal, aug_Pa_actual, aug_gt_F, aug_hgt] = ...
    augment_data_consistent(F_diff, F_after, F_before, P_before_ideal, P_after_actual, gt_F_vec, raw_hgt);

%% ========================================================================
%  Step 3: 最终训练集构建与安全检查
% =========================================================================
disp('--------------------------------------------------');
disp('3. 正在构建训练数据集...');

% 3.1 Net B 数据集
inputs_f_final   = [aug_F_after; aug_F_diff; aug_F_before]; 
targets_f_final  = aug_gt_F;

% 3.2 Net C 数据集 (输入含 Pb_ideal, 目标为 Pa_actual)
inputs_loc_final = [aug_F_diff; aug_F_after; aug_Pb_ideal]; 
targets_loc_final = double(aug_hgt) / 9.0; 
targets_shape_final = aug_Pa_actual; % 🌟 核心修正

% 3.3 暴力剔除 NaN/Inf
bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1) | ...
            any(isnan(targets_shape_final), 1);
inputs_f_final(:, bad_total) = []; targets_f_final(:, bad_total) = [];
inputs_loc_final(:, bad_total) = []; targets_loc_final(:, bad_total) = [];
aug_gt_F(:, bad_total) = []; targets_shape_final(:, bad_total) = [];

% 3.4 注入微量噪声 (防止 mapstd 崩溃)
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > 入网样本总数: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: 训练 Net B Force (外力大小方向)
% =========================================================================
disp('--------------------------------------------------');
disp('4. 正在训练 Net B Force...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;

[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% 验证指标
pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force Magnitude MAE: %.4f N\n', mae_f);

%% ========================================================================
%  Step 5: 训练 Net B Location (受力位置 - 加权 Loss)
% =========================================================================
disp('--------------------------------------------------');
disp('5. 正在训练 Net B Location (Weighted Loss)...');

% 5.1 筛选高受力样本进行训练
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in_l = inputs_loc_final(:, v_mask);
raw_tg_l = targets_loc_final(:, v_mask);
raw_tg_s = targets_shape_final(:, v_mask); % 对应的形态真值
node_labels = round(raw_tg_l * 9.0);

% 5.2 计算类别权重
nodes_interest = [3, 4, 5];
weights_vec = ones(1, length(node_labels));
for k = nodes_interest
    idx_k = (node_labels == k); count_k = sum(idx_k);
    if count_k > 0
        weights_vec(idx_k) = length(node_labels) / (length(nodes_interest) * count_k);
    end
end

% 5.3 训练
[in_l_norm, ps_in] = mapstd(raw_in_l); 
[tg_l_norm, ps_out] = mapstd(raw_tg_l);

net_loc = fitnet([60, 40, 20]);
net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = true; 
net_loc.divideParam.trainRatio = 0.8;
net_loc.divideParam.valRatio   = 0.2;
net_loc.divideParam.testRatio  = 0.0;

[net_loc, tr_l] = train(net_loc, in_l_norm, tg_l_norm, [], [], weights_vec);

%% ========================================================================
%  Step 6: Net B 性能评估可视化
% =========================================================================
disp('--------------------------------------------------');
disp('6. 正在生成 Net B 评估图表...');

pred_val_l = mapstd('reverse', net_loc(mapstd('apply', raw_in_l, ps_in)), ps_out);
pred_node = pred_val_l * 9.0;
real_node = raw_tg_l * 9.0;
pred_node(pred_node < 3) = 3; pred_node(pred_node > 5) = 5;

rmse_node = sqrt(mean((pred_node - real_node).^2));
acc_strict = sum(round(pred_node) == round(real_node)) / length(real_node);
fprintf('   > Location RMSE: %.2f Segment | Strict Acc: %.2f%%\n', rmse_node, acc_strict*100);

figure('Name', 'Net B Evaluation', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
subplot(1, 2, 1);
jitter = (rand(size(pred_node))-0.5)*0.2;
scatter(real_node, pred_node+jitter, 30, abs(real_node-pred_node), 'filled', 'MarkerFaceAlpha', 0.6);
colormap(jet); colorbar; hold on; plot([2.5, 5.5], [2.5, 5.5], 'k--');
title('Location Regression'); xlabel('Truth'); ylabel('Prediction'); grid on;

subplot(1, 2, 2);
cm = confusionchart(round(real_node), round(pred_node));
cm.Title = 'Location Confusion Matrix';

%% ========================================================================
%  Step 7: 训练 Net C Shape Reconstruction (受力形变重构)
% =========================================================================
disp('--------------------------------------------------');
disp('7. 正在训练 Net C (形态重构 - 目标为实际碰撞位姿)...');

% 7.1 构建 Net C 级联输入
feat_internal = raw_in_l(7:12, :); % F_after
feat_external = targets_f_final(:, v_mask); % F_ext (GT)
feat_location = raw_tg_l; % Location (GT)

inputs_net_c = [feat_internal; feat_external; feat_location];
targets_net_c = raw_tg_s; % 🌟 碰撞后真值！

% 7.2 训练
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

net_shape = fitnet([80, 60, 40]); 
net_shape.trainFcn = 'trainscg';
net_shape.trainParam.showWindow = true;
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-7;

[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8: Net C 全形态评估可视化
% =========================================================================
disp('--------------------------------------------------');
disp('8. 正在评估 Net C 碰撞重构性能...');

test_idx = tr_c.testInd;
in_test = inputs_net_c(:, test_idx);
target_test = targets_net_c(:, test_idx);

pred_test = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);

% 3D 骨架对比
figure('Name', '3D Shape Reconstruction (Post-Collision)', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
plot_ids = test_idx(randperm(length(test_idx), 4));

for k = 1:4
    idx = plot_ids(k);
    P_p = [[0;0;0], reshape(pred_test(:, find(test_idx==idx, 1)), 3, [])];
    P_r = [[0;0;0], reshape(target_test(:, find(test_idx==idx, 1)), 3, [])];
    
    subplot(1, 4, k);
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-o', 'LineWidth', 2); hold on;
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--.', 'LineWidth', 1.5);
    grid on; axis equal; title(['Sample ', num2str(k)]);
    if k==1, legend('Actual After', 'Predicted After'); end
    view(30, 20);
end

%% ========================================================================
%  Step 9: 尖端 (Tip) 专项误差分析
% =========================================================================
disp('--------------------------------------------------');
disp('9. 正在进行末端精度专项分析...');

tip_pred = pred_test(19:21, :);
tip_real = target_test(19:21, :);
tip_dist = sqrt(sum((tip_pred - tip_real).^2, 1)); 

fprintf('   > [Final Tip Accuracy] MAE: %.2f mm | RMSE: %.2f mm | Max: %.2f mm\n', ...
    mean(tip_dist)*1000, sqrt(mean(tip_dist.^2))*1000, max(tip_dist)*1000);

% Tip 追踪可视化
figure('Name', 'Tip Positioning Performance', 'Color', 'w', 'Position', [100, 200, 1000, 400]);
subplot(1, 2, 1);
hold on; grid on; axis equal;
num_show = min(50, length(tip_dist));
idx_show = randperm(length(tip_dist), num_show);
for k = idx_show
    p_r = tip_real(:, k); p_p = tip_pred(:, k);
    plot3([p_r(1), p_p(1)], [p_r(2), p_p(2)], [p_r(3), p_p(3)], 'Color', [0.7 0.7 0.7]);
    plot3(p_r(1), p_r(2), p_r(3), 'bo', 'MarkerFaceColor', 'b');
    plot3(p_p(1), p_p(2), p_p(3), 'r.', 'MarkerSize', 12);
end
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Tip Tracking (Gray line = Error)');

subplot(1, 2, 2);
histogram(tip_dist * 1000, 25, 'FaceColor', [0.2 0.6 0.3]);
xline(mean(tip_dist)*1000, 'r--', 'LineWidth', 2, 'Label', 'MAE');
xlabel('Tip Error (mm)'); ylabel('Count'); title('Error Distribution');

save('Final_Proprioception_Model.mat', 'net_force', 'net_loc', 'net_shape', 'ps_in_c', 'ps_out_c');
disp('>>> 系统全部流程运行完毕，模型已保存。');

%% ========================================================================
%  Helper Functions
% =========================================================================
function [aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_Pa, aug_gF, aug_h] = ...
    augment_data_consistent(F_diff, F_after, F_before, P_b, P_a, gt_F, hgt)
    
    N = size(F_diff, 2);
    R120 = [cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 = [cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    idx120 = [5, 6, 1, 2, 3, 4]; idx240 = [3, 4, 5, 6, 1, 2];
    
    rotP = @(P, R) reshape(R * reshape(P, 3, []), 21, N);
    
    aug_Fd = [F_diff, F_diff(idx120,:), F_diff(idx240,:)];
    aug_Fa = [F_after, F_after(idx120,:), F_after(idx240,:)];
    aug_Fb = [F_before, F_before(idx120,:), F_before(idx240,:)];
    aug_Pb = [P_b, rotP(P_b, R120), rotP(P_b, R240)];
    aug_Pa = [P_a, rotP(P_a, R120), rotP(P_a, R240)];
    aug_gF = [gt_F, R120*gt_F, R240*gt_F];
    aug_h  = [hgt, hgt, hgt];
end