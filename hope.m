%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Status:  FINAL CORRECTED VERSION (Physics-Guided Residual Learning)
%  Author:  Lin Yongxi
% =========================================================================
clc; clear; close all;
rng('default'); % Ensure reproducibility 

%% ========================================================================
%  Step 1: Data Loading, ROI Filtering & Cleaning
% =========================================================================
disp('--------------------------------------------------');
disp('1. Loading and preprocessing data...');

% 1.1 Load Data
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(FileName), error('File not found!'); end
dataTable = readtable(FileName);

% 1.2 Extract Signals
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))';  
F_before_raw = double(table2array(dataTable(3:end, 11:16)))';  
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 
pos_text_raw = dataTable{3:end, 29}; % 🌟 碰撞发生后的传感器位姿真值 (Target)

% 1.3 ROI Filtering
disp('   > Executing ROI filtering (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw, [3, 4, 5]);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_sub = pos_text_raw(roi_mask);

if length(raw_mag_sub) < 50, error('Insufficient data after ROI filtering.'); end

% 1.4 Data Cleaning
disp('   > Removing invalid samples...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | ...
          isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);
known_outliers = [686]; 
if ~isempty(known_outliers), bad_idx(known_outliers) = true; end
% 假设你通过前面的 Step 1.6 找到了 Excel 中的行号
excel_rows_to_kill = [206,207,224,608,609,610,611,612,613,614,615,616,617,618,619,620,621]; 

% 在当前 ROI 筛选后的序列中，找到对应这些 Excel 行号的样本并标记为坏点
bad_idx(ismember(track_rows, excel_rows_to_kill)) = true;

F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
pos_text = pos_text_sub(~bad_idx); 

F_diff = F_after - F_before;
N = length(raw_mag);
fprintf('   > Final effective samples: %d\n', N);

% 1.5 使用物理模型生成理论 P_before 作为【输入特征】
disp('   > 正在生成物理理想 $P_{before}$ (Feature) 与解析碰撞后真值 (Target)...');

tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; 
L_ap = 0.0665; L_bp = 0.00; N_dp = 7;
H_listp = linspace(0.0025, 0.0025, section_p*N_dp+1);
mu_p = 0.25; delta_alphap = 0; G_loadp = 4.000 * 0.00981;

P_before_ideal = zeros(21, N); 
P_after_sensor = zeros(21, N); 
gt_F_vec = zeros(3, N);

for i = 1:N
    % A. 生成物理理想 P_before (用于 Net C 输入特征)
    Fb_raw = F_before(:, i) * 0.00981; 
    Fb_sim = [Fb_raw(5); Fb_raw(6); Fb_raw(1); Fb_raw(2); Fb_raw(3); Fb_raw(4)];
    [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon_p, section_p, D_p, E_p, L_ap, L_bp, N_dp, H_listp, mu_p, delta_alphap, G_loadp, [0;0;0], Fb_sim, 14);
    
    V_local = [0; -0.004; 0]; 
    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2), P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; end
    marker_idx = round([2,4,6,8,10,12,14] * ((size(P_Theo,2)-1)/14)) + 1;
    P_before_ideal(:, i) = reshape(P_m(:, marker_idx), 21, 1); 
    
    % B. 解析碰撞后传感器真值 (用于 Net C 训练目标)
    real_offset_after = get_RealOffset_1S3CT(pos_text{i});
    P_after_sensor(:, i) = reshape(real_offset_after(:, 3:end), 21, 1); 
    
    % C. 外力矢量
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

%% ========================================================================
%  Step 2: Data Augmentation (同步旋转特征与目标)
% =========================================================================
disp('--------------------------------------------------');
disp('2. Executing synchronized rotational augmentation...');
[aug_F_diff, aug_F_after, aug_F_before, aug_Pb_ideal, aug_Pa_sensor, aug_gt_F, aug_hgt] = ...
    augment_data_consistent(F_diff, F_after, F_before, P_before_ideal, P_after_sensor, gt_F_vec, raw_hgt);

%% ========================================================================
%  Step 3: Dataset Construction & Safety Check
% =========================================================================
disp('--------------------------------------------------');
disp('3. Constructing final training set...');

inputs_f_final   = [aug_F_after; aug_F_diff; aug_F_before]; 
targets_f_final  = aug_gt_F;

% Net C 的特征池：包含理想起始位姿 Pb_ideal
inputs_loc_final = [aug_F_diff; aug_F_after; aug_Pb_ideal]; 
targets_loc_final = double(aug_hgt) / 9.0; 
targets_shape_final = aug_Pa_sensor; % 碰撞后实测真值

% Safety Check
bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1) | ...
            any(isnan(targets_shape_final), 1);
inputs_f_final(:, bad_total) = []; targets_f_final(:, bad_total) = [];
inputs_loc_final(:, bad_total) = []; targets_loc_final(:, bad_total) = [];
aug_gt_F(:, bad_total) = []; targets_shape_final(:, bad_total) = [];

epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > Final input samples: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation (保持不变)
% =========================================================================
disp('--------------------------------------------------');
disp('4. Training Net B Force...');
net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm'; net_force.trainParam.showWindow = false;
[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);
pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force MAE: %.4f N\n', mae_f);

%% ========================================================================
%  Step 5: Net B - Location Sensing (保持不变)
% =========================================================================
disp('--------------------------------------------------');
disp('5. Training Net B Location (Weighted Loss)...');
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in_l = inputs_loc_final(:, v_mask); raw_tg_l = targets_loc_final(:, v_mask);
node_labels = round(raw_tg_l * 9.0);
nodes_interest = [3, 4, 5]; weights_vec = ones(1, length(node_labels));
for k = nodes_interest
    idx_k = (node_labels == k); count_k = sum(idx_k);
    if count_k > 0, weights_vec(idx_k) = length(node_labels) / (length(nodes_interest) * count_k); end
end
[in_norm, ps_in] = mapstd(raw_in_l); [tg_norm, ps_out] = mapstd(raw_tg_l);
net_loc = fitnet([60, 40, 20]); net_loc.trainFcn = 'trainlm'; net_loc.trainParam.showWindow = true; 
[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [], [], weights_vec);

%% ========================================================================
%  Step 6: Net B Evaluation (保持不变)
% =========================================================================
pred_val = mapstd('reverse', net_loc(mapstd('apply', raw_in_l, ps_in)), ps_out);
pred_node = pred_val * 9.0; real_node = raw_tg_l * 9.0;
pred_node(pred_node < 3) = 3; pred_node(pred_node > 5) = 5;
rmse_node = sqrt(mean((pred_node - real_node).^2));
acc_strict = sum(round(pred_node) == round(real_node)) / length(real_node);
fprintf('   > [Location] RMSE: %.2f | Acc: %.2f%%\n', rmse_node, acc_strict*100);

%% ========================================================================
%  Step 7: Net C - Shape Reconstruction (🌟残差学习优化版)
% =========================================================================
disp('--------------------------------------------------');
disp('7. Training Net C (Predicting ACTUAL Collision Shape via Residuals)...');

% 7.1 构建残差训练集
feat_internal = raw_in_l(7:12, :);      % F_after
feat_external = targets_f_final(:, v_mask); % F_ext (GT)
feat_location = raw_tg_l;               % Loc (GT)
feat_Pb_ideal = raw_in_l(13:33, :);     % 🌟 引入理想起始位姿作为特征

% 输入级联 (31 维)
inputs_net_c = [feat_internal; feat_external; feat_location; feat_Pb_ideal];

% 🌟 训练目标：位移残差 Delta_P = 碰撞后实测 - 碰撞前理论
targets_net_c_actual = targets_shape_final(:, v_mask);
targets_net_c_delta  = targets_net_c_actual - feat_Pb_ideal;

% 7.2 训练
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c_delta);

net_shape = fitnet([100, 80, 40]); % 略微增加神经元处理 31 维输入
net_shape.trainFcn = 'trainscg'; 
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-8;

[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8: Net C Evaluation (还原绝对坐标)
% =========================================================================
disp('--------------------------------------------------');
disp('8. Evaluating Net C Performance (vs ACTUAL SENSOR AFTER)...');

test_idx = tr_c.testInd;
in_test = inputs_net_c(:, test_idx);
target_test_actual = targets_net_c_actual(:, test_idx); % 终极真值考卷
Pb_test = feat_Pb_ideal(:, test_idx);

% 预测残差并还原绝对坐标
pred_delta = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);
pred_test_actual = Pb_test + pred_delta;

% MAE 计算
dist_errs = zeros(1, length(test_idx));
for i = 1:length(test_idx)
    p_p = reshape(pred_test_actual(:, i), 3, 7);
    p_r = reshape(target_test_actual(:, i), 3, 7);
    dist_errs(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
end
mean_dist = mean(dist_errs);
fprintf('   > [Net C] Mean Shape Error: %.4f m (%.2f mm)\n', mean_dist, mean_dist*1000);

% 3D Visualization
figure('Name', '3D Shape Reconstruction', 'Color', 'w', 'Position', [100, 100, 1200, 500]);
plot_ids = test_idx(randperm(length(test_idx), 4));
for k = 1:4
    subplot(1, 4, k); hold on; grid on; axis equal;
    idx_sub = find(test_idx == plot_ids(k));
    P_p = [[0;0;0], reshape(pred_test_actual(:, idx_sub), 3, [])];
    P_r = [[0;0;0], reshape(target_test_actual(:, idx_sub), 3, [])];
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-o', 'LineWidth', 2);
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--.', 'LineWidth', 1.5);
    title(['Sample ', num2str(k)]); view(30, 20);
end

%% ========================================================================
%  Step 9: Tip Error Analysis
% =========================================================================
disp('--------------------------------------------------');
disp('9. Analyzing Tip-Specific Error...');
tip_pred = pred_test_actual(19:21, :); tip_real = target_test_actual(19:21, :);
tip_dist = sqrt(sum((tip_pred - tip_real).^2, 1)); 

fprintf('   > [Tip Accuracy] MAE: %.2f mm | RMSE: %.2f mm\n', mean(tip_dist)*1000, sqrt(mean(tip_dist.^2))*1000);

figure('Name', 'Tip Error Analysis', 'Color', 'w', 'Position', [100, 200, 1000, 400]);
subplot(1, 2, 1); hold on; grid on; axis equal;
for k = 1:min(50, length(tip_dist))
    plot3([tip_real(1,k), tip_pred(1,k)], [tip_real(2,k), tip_pred(2,k)], [tip_real(3,k), tip_pred(3,k)], 'Color', [0.7 0.7 0.7]);
    plot3(tip_real(1,k), tip_real(2,k), tip_real(3,k), 'bo', 'MarkerFaceColor','b');
    plot3(tip_pred(1,k), tip_pred(2,k), tip_pred(3,k), 'r.');
end
title('Tip Tracking'); view(45, 30);
subplot(1, 2, 2);
histogram(tip_dist * 1000, 25, 'FaceColor', [0.2 0.6 0.3]);
xline(mean(tip_dist)*1000, 'r--', 'LineWidth', 2, 'Label', 'MAE');
title('Error Dist.'); grid on;

save('Final_Proprioception_Model.mat', 'net_force', 'net_loc', 'net_shape', 'ps_in_c', 'ps_out_c');
disp('>>> All done. Logic Corrected.');
%% ========================================================================
%  Step 10: 最差案例深度诊断 (Top 5 Worst Tip Error Cases)
%  目的：单独画出误差最大的 5 组，观察网络在什么情况下会失效
% =========================================================================
disp('--------------------------------------------------');
disp('10. 正在提取并绘制 Top 5 Worst Cases...');

% 1. 降序排列误差，获取前 5 名的索引
[sorted_tip_err, s_idx] = sort(tip_dist, 'descend');
num_worst = min(10, length(tip_dist));

% 🌟 物理原点
base_origin = [0; 0; 0];

for k = 1:num_worst
    % 获取该样本在测试集中的局部索引
    local_idx = s_idx(k);
    % 获取对应的坐标 (3x7)
    P_p = reshape(pred_test_actual(:, local_idx), 3, 7);
    P_r = reshape(target_test_actual(:, local_idx), 3, 7);
    P_b = reshape(Pb_test(:, local_idx), 3, 7); % 初始理论位姿
    
    % 计算该点的误差值 (mm)
    err_val = sorted_tip_err(k) * 1000;
    
    % --- 创建独立大图 ---
    fig_name = sprintf('Worst Case #%d (Tip Error: %.2f mm)', k, err_val);
    figure('Name', fig_name, 'Color', 'w', 'Position', [200 + k*30, 200, 800, 700]);
    hold on; grid on; axis equal;
    
    % 🌟 对齐坐标轴视角 (与物理实验一致)
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    
    % 1. 绘制基座指示 (黄星)
    plot3(0, 0, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin');
    
    % 2. 绘制初始参考形态 (绿色点线 - Pb_ideal)
    Pb_plot = [base_origin, P_b];
    plot3(Pb_plot(1,:), Pb_plot(2,:), Pb_plot(3,:), 'g:', 'LineWidth', 1.2, 'DisplayName', 'Initial State (Theory)');
    
    % 3. 绘制传感器实测形态 (黑色实线 - Pa_actual)
    Pr_plot = [base_origin, P_r];
    h_real = plot3(Pr_plot(1,:), Pr_plot(2,:), Pr_plot(3,:), 'k-s', 'LineWidth', 2.5, ...
                   'MarkerSize', 7, 'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth (Sensor After)');
               
    % 4. 绘制网络预测形态 (红色虚线 - Pa_predicted)
    Pp_plot = [base_origin, P_p];
    h_pred = plot3(Pp_plot(1,:), Pp_plot(2,:), Pp_plot(3,:), 'r--o', 'LineWidth', 2, ...
                   'MarkerSize', 8, 'MarkerFaceColor', 'w', 'DisplayName', 'System Prediction (Net C)');
               
    % 5. 🌟 重点标注：末端误差向量 (紫红色粗线)
    tip_real = P_r(:, end);
    tip_pred = P_p(:, end);
    plot3([tip_real(1), tip_pred(1)], [tip_real(2), tip_pred(2)], [tip_real(3), tip_pred(3)], ...
          'm-', 'LineWidth', 4, 'DisplayName', 'Tip Error Vector');
      
    % 6. 装饰与标注
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title_str = sprintf('Worst Case Analysis #%d\nTip Deviation: %.2f mm', k, err_val);
    title(title_str, 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 10);
    
    % 绘制基准轴指示
    quiver3(0,0,0, 0.04,0,0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5); 
    quiver3(0,0,0, 0,0.04,0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5); 
    quiver3(0,0,0, 0,0,0.04, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    
    view(35, 25);
    
    % 控制台输出详细诊断
    fprintf('   > Worst #%d: Tip Error = %.2f mm\n', k, err_val);
end

disp('✅ 所有最差案例图已生成。请观察绿色参考线与黑/红线的偏差。');
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

function [aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_gF, aug_h] = augment_data_by_rotation(varargin)
    % 此函数已被 augment_data_consistent 取代，为防止报错保留空壳
end