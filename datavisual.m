%% ========================================================================
%  Script: Dataset Visualization (Hanging Style & Uniform Color)
%  Goal:   Visualize robot shapes hanging downwards in a single color
% =========================================================================
clc; clear; close all;

%% 1. 数据读取与预处理
disp('1. Loading Data...');
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';

if ~isfile(FileName)
    error('错误：找不到文件，请检查路径！');
end

dataTable = readtable(FileName);
num_total_exp = height(dataTable) - 2; 

% 提取关键列
direction_array = table2array(dataTable(3:end, 3)); 
position_text_array = dataTable{3:end, 29}; 

%% 2. 筛选与配置
% ====================================================
target_dir_code = 4;  % 2=NegX, 3=45deg, 4=PosY
num_samples_to_plot = 100; % 画多一点，单色图不怕密集
% ====================================================

candidate_indices = find(direction_array == target_dir_code);
if isempty(candidate_indices), error('无数据'); end

if length(candidate_indices) > num_samples_to_plot
    perm = randperm(length(candidate_indices), num_samples_to_plot);
    selected_indices = candidate_indices(perm);
else
    selected_indices = candidate_indices;
end

%% 3. 绘图 (绝对忠实于原始数据)
disp('2. Rendering Truth Plot...');

figure('Name', 'Hanging Visualization (Raw Data)', 'Color', 'w');
hold on; grid on; axis equal;

% 统一颜色
uni_color = [0.2, 0.4, 0.6]; 
line_alpha = 0.3; 

for k = 1:length(selected_indices)
    idx = selected_indices(k);
    try
        P = get_RealOffset_1S3CT(position_text_array{idx}); 
        
        px = P(1, :);
        py = P(2, :);
        pz = P(3, :); 
        
        plot3(px, py, pz, '-', 'Color', [uni_color, line_alpha], 'LineWidth', 1.2);
        
        % 标记末端
        plot3(px(end), py(end), pz(end), '.', 'Color', [0.1 0.1 0.1], 'MarkerSize', 6);
    catch
        continue;
    end
end

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Raw Data Visualization (Dir: %d)', target_dir_code));

% --- 关键：通过调整坐标轴方向来实现"倒挂"视觉，而不是改数据 ---
% 检查数据的 Z 值分布
mean_z = mean(parsed_shapes(3,:,:), 'all');

if mean_z > 0
    % 如果数据全是正的（说明数据定义是向下的延伸量），但Matlab默认Z向上
    % 我们反转 Z 轴显示方向
    set(gca, 'ZDir', 'normal'); 
    disp('检测到 Z 值为正，已反转 Z 轴显示方向以模拟倒挂。');
else
    % 如果数据本来就是负的（说明数据定义就是真实物理坐标），正常显示
    set(gca, 'ZDir', 'reverse');
end

