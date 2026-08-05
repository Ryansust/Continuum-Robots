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
%  Step 1.6: [数据病理检查] 自动发现 Ground Truth 中的“突起”异常点
%  检测原理：计算每个 Marker 偏离其前后邻居中点的距离（中值偏差法）
% =========================================================================
disp('--------------------------------------------------');
disp('1.6 正在检查 Ground Truth 物理一致性（寻找畸形点）...');

% 设定阈值 (单位: 米)
% 连续体机器人相邻 Marker 距离通常很固定，如果某个点偏离“平滑线”超过 8mm-10mm，基本就是错点
threshold_protrusion = 0.008; 

outlier_list = []; % 记录发现的坏账行号
max_deviations = zeros(1, N); % 记录每个样本的最大畸形程度

for i = 1:N
    % 提取当前样本的 7 个点 (3x7)
    P = reshape(P_after_sensor(:, i), 3, 7);
    
    % 计算内部点 (Marker 2 到 6) 的畸形度
    % 畸形度 = 当前点 距离其 前后两点连线中点 的欧式距离
    sample_deviations = [];
    for m = 2:6
        mid_point = (P(:, m-1) + P(:, m+1)) / 2;
        dev = norm(P(:, m) - mid_point);
        sample_deviations = [sample_deviations, dev];
    end
    
    % 记录该样本中最离谱的一个点的偏差
    max_deviations(i) = max(sample_deviations);
    
    % 如果偏差超过阈值，判定为“畸形数据”
    if max_deviations(i) > threshold_protrusion
        excel_row = i + 2; % 映射回 Excel 行号 (假设没过虑前的索引)
        % 如果你用了 ROI 筛选和 Cleaning，请使用我们之前的 track_rows 变量：
        % excel_row = excel_rows(i); 
        
        outlier_list = [outlier_list; excel_row, max_deviations(i)*1000];
    end
end

% -------------------------------------------------------------------------
% 📊 结果汇报与“通缉令”
% -------------------------------------------------------------------------
if isempty(outlier_list)
    disp('✅ 恭喜！所有 Ground Truth 均符合平滑曲线逻辑。');
else
    fprintf('⚠️ 警报！发现 %d 组 Ground Truth 存在点位突起异常：\n', size(outlier_list, 1));
    fprintf('   Excel行号    突起程度(mm)\n');
    disp(outlier_list);
    
    % 自动画出最离谱的前 3 个异常点，让你确认
    [~, sort_idx] = sort(max_deviations, 'descend');
    num_verify = min(3, length(outlier_list));
    
    for v = 1:num_verify
        idx_bad = sort_idx(v);
        P_bad = reshape(P_after_sensor(:, idx_bad), 3, 7);
        
        %figure('Name', sprintf('异常点确认 - Excel Row: %d', excel_row(idx_bad)), 'Color', 'w');
        plot3(P_bad(1,:), P_bad(2,:), P_bad(3,:), 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
        grid on; axis equal; hold on;
        %title(['畸形数据特写 (Row: ', num2str(excel_rows(idx_bad)), ')']);
        xlabel('X'); ylabel('Y'); zlabel('Z');
        
        % 圈出那个突出的点
        for m = 2:6
            mid = (P_bad(:, m-1) + P_bad(:, m+1)) / 2;
            if norm(P_bad(:, m) - mid) > threshold_protrusion
                plot3(P_bad(1,m), P_bad(2,m), P_bad(3,m), 'ko', 'MarkerSize', 15, 'LineWidth', 2);
                text(P_bad(1,m), P_bad(2,m), P_bad(3,m), '  此处畸形!', 'FontSize', 12, 'Color', 'k');
            end
        end
        view(30, 20);
    end
end