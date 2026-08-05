%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author:  Lin Yongxi (Corrected Hardcore Physics Version)
%  功能：物理模型验证 - 严格补偿 4mm 偏置与基座对齐
% =========================================================================
clc; clear; close all;

%% 1. 读取数据
disp('1. 正在读取数据...');
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

% 这里的列索引请务必根据你的 Excel 真实情况微调：
% 假设 11:16 是碰撞前的腱绳力 F_before
force_array = double(table2array(dataTable(3:end, 11:16))); 
direction_array = double(table2array(dataTable(3:end, 3)));
height_array = double(table2array(dataTable(3:end, 4)));
position_text_array = dataTable{3:end, 38}; % 碰撞前形态文本

%% 2. 随机抽样
num_fast_test = 100; 
total_rows = size(force_array, 1);
selected_indices = randperm(total_rows, num_fast_test);

%% 3. 物理参数设置
tendon = 3; section = 2; D = 0.0006; E = 0.516e+12; 
L_a = 0.0665; L_b = 0.00; N_d = 7;
H_list = linspace(0.0025, 0.0025, section*N_d+1);
mu = 0.25; delta_alpha = 0; G_load = 4.000 * 0.00981;
V_offset = [0; -0.004; 0]; % 核心：传感器相对于主梁的 4mm 偏置向量

%% 4. 极速计算 (包含基座对齐与 4mm 补偿)
fprintf('2. 开始计算 (包含基座归零与 4mm 偏置补偿)...\n');
tip_errors = zeros(1, num_fast_test);
all_P_theo = cell(1, num_fast_test);
all_P_real = cell(1, num_fast_test);

for k = 1:num_fast_test
    exp_id = selected_indices(k);
    
    % --- A. 腱绳力预处理 ---
    F_raw = force_array(exp_id, :) * 0.00981; 
    F_sim = F_raw([5, 6, 1, 2, 3, 4]); % 索引重排
    
    % --- B. 解析真值并执行【基座归一化】 ---
    P_Real_Raw = get_RealOffset_1S3CT(position_text_array{exp_id});
    % 核心：以基座两个 Marker 的中点为原点 (0,0,0)
    base_center = (P_Real_Raw(:, 1) + P_Real_Raw(:, 2)) / 2;
    P_Real_Centered = P_Real_Raw(:, 3:end) - base_center; 
    
    % --- C. 求解物理模型并执行【4mm 偏置补偿】 ---
    [P_model, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon, section, D, E, L_a, L_b, N_d, H_list, mu, delta_alpha, G_load, [0;0;0], F_sim, 14);
    
    % 只取与传感器对应的 7 个主干节点坐标
    marker_idx = round([2,4,6,8,10,12,14] * ((size(P_model,2)-1)/14)) + 1;
    P_Theo_Centerline = P_model(:, marker_idx);
    R_mat_selected = R_mat(:, :, marker_idx);
    
    % 对每个点应用 R_mat * V_offset 补偿
    P_Theo_Surface = zeros(3, 7);
    for j = 1:7
        P_Theo_Surface(:, j) = P_Theo_Centerline(:, j) + R_mat_selected(:, :, j) * V_offset;
    end
    
    % --- D. 计算误差 ---
    tip_errors(k) = norm(P_Theo_Surface(:, end) - P_Real_Centered(:, end)) * 1000;
    all_P_theo{k} = P_Theo_Surface;
    all_P_real{k} = P_Real_Centered;
end

avg_err = mean(tip_errors);
fprintf('\n======================================================\n');
fprintf('🎯 修正后平均尖端误差: %.2f mm\n', avg_err);
fprintf('======================================================\n');

%% 5. 绘制 Worst Cases
[sorted_errs, sort_idx] = sort(tip_errors, 'descend');
num_plots = 5;

for i = 1:num_plots
    k = sort_idx(i);
    exp_id = selected_indices(k);
    err_val = sorted_errs(i);
    
    P_R = [[0;0;0], all_P_real{k}];
    P_T = [[0;0;0], all_P_theo{k}];
    
    figure('Name', sprintf('Worst Case #%d (Row: %d)', i, exp_id+2), 'Color', 'w');
    hold on; grid on; axis equal;
    
    % 绘制基座参考
    plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
    quiver3(0,0,0, 0.03,0,0, 'r', 'LineWidth', 2); quiver3(0,0,0, 0,0.03,0, 'g', 'LineWidth', 2); quiver3(0,0,0, 0,0,0.03, 'b', 'LineWidth', 2);
    
    % 绘制形态
    h1 = plot3(P_R(1,:), P_R(2,:), P_R(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'k');
    h2 = plot3(P_T(1,:), P_T(2,:), P_T(3,:), 'r--o', 'LineWidth', 2);
    
    % 绘制误差向量 (紫线)
    plot3([P_R(1,end), P_T(1,end)], [P_R(2,end), P_T(2,end)], [P_R(3,end), P_T(3,end)], 'm-', 'LineWidth', 3);
    
    % 视角与坐标轴翻转 (严格执行实验视角)
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    view(30, 20);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Worst Case %d | Tip Error: %.2f mm\nExcel Row: %d', i, err_val, exp_id+2));
    legend([h1, h2], {'Mocap Truth (Recentered)', 'Phys Model (4mm Compensated)'}, 'Location', 'southoutside');
end