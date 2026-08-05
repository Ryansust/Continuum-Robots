%% ========================================================================
%  Continuous Robot Validation - SENSOR GROUND TRUTH VERSION
%  Goal: Compare Net C Output (Post-Collision) with ACTUAL Sensor Data
% =========================================================================
clc; clear; close all;
fprintf('正在启动系统验证：对比 Net C 输出与碰撞后传感器真值...\n');

% 1. 加载模型
if ~isfile('Final_Proprioception_Model.mat'), error('未找到模型文件！'); end
load('Final_Proprioception_Model.mat'); 

% 2. 读取 Excel 原始数据 (作为真值来源)
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% 3. 严格对齐预处理逻辑
raw_hgt_all = double(table2array(dataTable(3:end, 4)))';
roi_mask = ismember(raw_hgt_all, [3, 4, 5]);
track_rows = (3 : (length(raw_hgt_all) + 2));
track_rows = track_rows(roi_mask);

% 提取拉力信号
F_before_raw = double(table2array(dataTable(3:end, 11:16)))'; 
F_after_raw  = double(table2array(dataTable(3:end, 23:28)))';
F_before_sub = F_before_raw(:, roi_mask);
F_after_sub  = F_after_raw(:, roi_mask);

% 🌟 提取位姿字符串 (此处必须确保列号对应碰撞后的 Nokov 数据)
% 假设列 38 是受力后的 Pose Text
pos_text_after_sub = dataTable{3:end, 38}; 
pos_text_after_sub = pos_text_after_sub(roi_mask);

% 坏样本剔除
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1);
bad_idx(686) = true; 

F_before_eval = F_before_sub(:, ~bad_idx);
F_after_eval  = F_after_sub(:, ~bad_idx);
pos_text_after_eval = pos_text_after_sub(~bad_idx);
excel_rows     = track_rows(~bad_idx);

% 过滤受力过小的点 (对应训练时的 v_mask)
% 注意：此处直接提取原始数据对应 v_mask 的前 N 部分
N_clean = length(excel_rows);
v_mask_orig = v_mask(1:N_clean); 
valid_indices = find(v_mask_orig);

F_b_final = F_before_eval(:, valid_indices);
F_a_final = F_after_eval(:, valid_indices);
pos_t_final = pos_text_after_eval(valid_indices);
rows_final = excel_rows(valid_indices);
N_eval = length(valid_indices);

%% ========================================================================
%  MODULE 1: 物理模型 P_before 精度 (地基验证)
% =========================================================================
fprintf('>>> 模块 1: 验证物理模型生成的理想 P_before...\n');
P_phys_before = zeros(21, N_eval);
err_pbefore = zeros(1, N_eval);

% 物理参数
tendon_p=3; section_p=2; D_p=0.0006; E_p=0.516e+12; L_ap=0.0665; N_dp=7;
H_listp = linspace(0.0025, 0.0025, 15); mu_p=0.25; G_loadp=4.0*0.00981;

for i = 1:N_eval
    Fs = F_b_final(:, i) * 0.00981;
    Fs = Fs([5,6,1,2,3,4]);
    [Pt, ~, Rm, ~, ~, ~] = solve_continuum_shape_nofig(tendon_p, section_p, D_p, E_p, L_ap, 0, N_dp, H_listp, mu_p, 0, G_loadp, [0;0;0], Fs, 14);
    % 偏置
    Pm = zeros(3, size(Pt,2));
    for pt=1:size(Pt,2), Pm(:,pt) = Pt(:,pt) + Rm(:,:,pt)*[0;-0.004;0]; end
    k_r = (size(Pt,2)-1)/14;
    P_ideal = Pm(:, round([2,4,6,8,10,12,14]*k_r)+1);
    P_phys_before(:, i) = reshape(P_ideal, 21, 1);
end

%% ========================================================================
%  MODULE 2: Net B (受力大小与位置) 验证
% =========================================================================
fprintf('>>> 模块 2: 验证 Net B 预测性能...\n');
gt_mag_all = abs(double(table2array(dataTable(3:end, 2))));
gt_hgt_all = double(table2array(dataTable(3:end, 4)));
gt_mag = zeros(1, N_eval); gt_hgt = zeros(1, N_eval);
for i = 1:N_eval
    ridx = rows_final(i) - 2;
    gt_mag(i) = gt_mag_all(ridx); gt_hgt(i) = gt_hgt_all(ridx);
end

pred_f_vec = net_force([F_a_final; (F_a_final - F_b_final); F_b_final]);
pred_l_norm = net_loc(mapstd('apply', [(F_a_final - F_b_final); F_a_final; P_phys_before], ps_in));
pred_l = mapstd('reverse', pred_l_norm, ps_out) * 9.0;

err_mag = abs(sqrt(sum(pred_f_vec.^2, 1)) - gt_mag);
err_loc = abs(pred_l - gt_hgt);

%% ========================================================================
%  MODULE 3: Net C (碰撞后形态重构) 终极验证
% =========================================================================
fprintf('>>> 模块 3: 验证 Net C 预测形态 vs 传感器碰撞后真值...\n');
in_c = [F_a_final; pred_f_vec; pred_l/9.0];
pred_P_after = mapstd('reverse', net_shape(mapstd('apply', in_c, ps_in_c)), ps_out_c);

% 🌟 提取传感器在碰撞瞬间的真实形态 (Nokov $P_{after}$)
P_sensor_after = zeros(21, N_eval);
err_tip = zeros(1, N_eval);
err_shape_avg = zeros(1, N_eval);

for i = 1:N_eval
    P_s_raw = get_RealOffset_1S3CT(pos_t_final{i});
    P_s_after = reshape(P_s_raw(:, 3:end), 21, 1);
    P_sensor_after(:, i) = P_s_after;
    
    % 计算重构误差
    p_p = reshape(pred_P_after(:, i), 3, 7);
    p_r = reshape(P_s_after, 3, 7);
    
    err_shape_avg(i) = mean(sqrt(sum((p_p - p_r).^2, 1))) * 1000; % mm
    err_tip(i) = norm(p_p(:, end) - p_r(:, end)) * 1000; % mm
end

%% ========================================================================
%  📊 命令行综合报表
% =========================================================================
MetricNames = {'NetB_Force_MAE'; 'NetB_Location_RMSE'; 'Final_System_MeanShape_MAE'; 'Final_System_Tip_MAE'};
Values = [mean(err_mag); sqrt(mean(err_loc.^2)); mean(err_shape_avg); mean(err_tip)];
Units = {'N'; 'segment'; 'mm'; 'mm'};
Accuracy_Report = table(MetricNames, Values, Units);
disp(' '); disp('================ FINAL ACCURACY REPORT (COLLISION VERIFIED) ================');
disp(Accuracy_Report);
disp('============================================================================');

%% ========================================================================
%  🚨 WORST CASE ANALYSIS: 独立大图展示 (Net C 碰撞后重构)
% =========================================================================
[sorted_err, sort_idx] = sort(err_tip, 'descend');
worst_indices = sort_idx(1:5);

for k = 1:5
    idx = worst_indices(k);
    fig = figure('Color','w','Name',sprintf('Worst Reconstruction Case #%d', k), 'Position', [150 150 850 650]);
    hold on; grid on; axis equal;
    
    p_pred = reshape(pred_P_after(:, idx), 3, 7);
    p_real = reshape(P_sensor_after(:, idx), 3, 7);
    
    % 画图 (黑色代表传感器真实碰撞后，红色代表系统预测结果)
    h1 = plot3(p_real(1,:), p_real(2,:), p_real(3,:), 'k-o', 'LineWidth', 3, 'MarkerFaceColor', 'k');
    h2 = plot3(p_pred(1,:), p_pred(2,:), p_pred(3,:), 'r--s', 'LineWidth', 2, 'MarkerSize', 8);
    
    % 在末端标注误差向量
    plot3(p_real(1,end), p_real(2,end), p_real(3,end), 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'k');
    plot3(p_pred(1,end), p_pred(2,end), p_pred(3,end), 'rp', 'MarkerSize', 16, 'MarkerFaceColor', 'r');
    line([p_real(1,end), p_pred(1,end)], [p_real(2,end), p_pred(2,end)], [p_real(3,end), p_pred(3,end)], ...
        'Color', 'm', 'LineWidth', 3, 'LineStyle', '-');
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title_str = sprintf('Worst Case #%d\nExcel Row: %d | Tip Error: %.2f mm', k, rows_final(idx), err_tip(idx));
    title(title_str, 'FontSize', 14);
    legend([h1, h2], {'Actual Sensor (Post-Collision)', 'System Prediction (Net C)'}, 'Location', 'best');
    view(30, 20);
end

% 误差直方图
figure('Color','w','Name','Post-Collision Tip Error Distribution');
histogram(err_tip, 25, 'FaceColor', [0.8 0.3 0.3]); grid on;
xline(mean(err_tip), 'k--', 'LineWidth', 2.5, 'Label', sprintf('MAE: %.2fmm', mean(err_tip)));
title('System Prediction Accuracy (vs Post-Collision GT)', 'FontSize', 14);
xlabel('Euclidean Distance Error at Tip (mm)'); ylabel('Frequency');

fprintf('>>> 验证完成。结果已准确反映系统在碰撞后的真实感知能力。\n');