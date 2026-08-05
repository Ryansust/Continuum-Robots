%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author: Lin Yongxi (Bulletproof Data Parsing Version)
% =========================================================================
clc; clear; close all;
rng('default');

%% ========================================================================
%  Step 1: Robust Data Loading (Column-by-Column Parsing)
% =========================================================================
disp('1. Loading and parsing data (Strict individual column parsing)...');
FileName = '/Users/ryan/Desktop/continuum robot/dynamic_reconstruction_data.xlsx'; 
if ~isfile(FileName), error('Excel file not found!'); end

% 读取 Table，不进行任何自动转换
dataTable = readtable(FileName, 'VariableNamingRule', 'preserve');

% --- 定义健壮的单列提取函数 ---
% 解决 cell 和 double 无法串联的问题
extract_col = @(tab, colIdx) reshape(double(string(tab{3:end, colIdx})), [], 1);

% --- 定义批量列提取函数 ---
extract_range = @(tab, colRange) cell2mat(arrayfun(@(c) extract_col(tab, c), colRange, 'UniformOutput', false));

% A. 时间戳 (A列 - 1)
time_raw = extract_col(dataTable, 1);

% B. 实时六轴拉力 (B-G列 - 2-7)
F_all_raw = extract_range(dataTable, 2:7);

% C. 外力重组 (J列 - 10)
% 需求：提取 Z 分量，映射到机器人坐标系的 -X 方向
Fz_val = extract_col(dataTable, 10);
F_ext_mapped = [-abs(Fz_val), zeros(size(Fz_val)), zeros(size(Fz_val))];

% D. Marker 坐标提取 (Q-AQ列 - 17-43)
% 我们先完整提取 17 到 43 列，存储为矩阵，然后再进行顺序重排
raw_markers_mat = extract_range(dataTable, 17:43);

% 物理转换因子
conversion_factor = 0.00981;
F_all_raw = F_all_raw * conversion_factor;

%% ========================================================================
%  Step 2: Sequence Filtering (Time-Step Delta)
% =========================================================================
disp('2. Processing time-step differences (t vs t+1)...');
F_diff_threshold = 1e-4; 

F_before_list = []; F_after_list = [];
time_list = []; F_ext_gt_list = []; P_after_gt_list = [];

% 这里的索引是基于 raw_markers_mat 的列索引 (1-27)
% raw_markers_mat 的 1-3列对应 Q-S (Base)
% Q-S(1-3):Base, T-V(4-6):Mid, W-Y(7-9):Tip, Z-AB(10-12):M1, AC-AE(13-15):M2, 
% AF-AH(16-18):M3, AI-AK(19-21):M4, AL-AN(22-24):M5, AO-AQ(25-27):M6
for i = 1 : size(F_all_raw, 1) - 1
    Fb = F_all_raw(i, :)';
    Fa = F_all_raw(i+1, :)';
    dF = Fa - Fb;
    
    if norm(dF) > F_diff_threshold
        F_before_list = [F_before_list, Fb];
        F_after_list  = [F_after_list, Fa];
        time_list     = [time_list, time_raw(i+1)];
        F_ext_gt_list = [F_ext_gt_list, F_ext_mapped(i+1, :)'];
        
        % 提取当前行的 Marker 坐标
        m = raw_markers_mat(i+1, :);
        base_pos = m(1:3); % Base (Q-S)
        
        % 严格按照从下往上数(19->1)的物理顺序重排 Marker:
        % Base(19):1-3, M1(17):10-12, M2(15):13-15, M3(12):16-18, Mid(10):4-6, 
        % M4(8):19-21, M5(6):22-24, M6(3):25-27, Tip(1):7-9
        p_seq = [m(1:3); m(10:12); m(13:15); m(16:18); m(4:6); m(19:21); m(22:24); m(25:27); m(7:9)]';
        
        % 减去基座坐标实现局部坐标系对齐
        p_aligned = p_seq - base_pos'; 
        P_after_gt_list = [P_after_gt_list, reshape(p_aligned, [], 1)];
    end
end

N_samples = size(F_before_list, 2);
if N_samples == 0, error('No valid data after filtering! Check F_diff_threshold.'); end
fprintf('   > Samples after cleaning: %d\n', N_samples);

%% ========================================================================
%  Step 3: Physics-Guided Prior (CSBCM)
% =========================================================================
disp('3. Generating Physics Priors...');
% 物理参数 (针对新机器人请确认)
tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; 
L_ap = 0.0665; L_bp = 0.00; N_dp = 18; % 19个盘
H_listp = linspace(0.0025, 0.0025, section_p*N_dp+1);
mu_p = 0.25; G_loadp = 4.000 * 0.00981;

P_before_ideal = zeros(27, N_samples);

for i = 1:N_samples
    Fb_raw = F_before_list(:, i);
    Fb_sim = [Fb_raw(5); Fb_raw(6); Fb_raw(1); Fb_raw(2); Fb_raw(3); Fb_raw(4)];
    
[P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon_p, section_p, D_p, E_p, L_ap, L_bp, N_dp, H_listp, mu_p, 0, G_loadp, [0;0;0], Fb_sim, 14);
    
    V_local = [0; -0.004; 0]; 
    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2)
        P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; 
    end
    
    % 提取 9个 Marker 对应的物理节点 (1-19)
    node_idx = [1, 3, 5, 8, 10, 12, 14, 17, 19]; 
    P_before_ideal(:, i) = reshape(P_m(:, node_idx), 27, 1);
end

%% ========================================================================
%  Step 4: Training & Dynamic Inference
% =========================================================================
disp('4. Training Networks...');

% Net B: Regression of External Force X-component
inputs_B = [F_after_list; (F_after_list - F_before_list); F_before_list];
targets_B_force = F_ext_gt_list(1, :); % -X 方向力

net_force = feedforwardnet([40, 20]);
net_force.trainParam.showWindow = true;
net_force.trainFcn = 'trainlm';
net_force = train(net_force, inputs_B, targets_B_force);
pred_F_x = net_force(inputs_B);

% Net C: Residual Shape Prediction
loc_label = repmat(4/19, 1, N_samples);
inputs_C = [F_after_list; pred_F_x; loc_label; P_before_ideal];
targets_C_delta = P_after_gt_list - P_before_ideal;

net_shape = fitnet([80, 60, 40]);
net_shape.trainParam.showWindow = true;
net_shape.trainFcn = 'trainscg';
net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio = 0.2;
net_shape = train(net_shape, inputs_C, targets_C_delta);

%% ========================================================================
%  Step 5: Visualizing Results
% =========================================================================
disp('5. Visualizing Time-Series Tracking...');

% 全序列推理
pred_delta = net_shape(inputs_C);
P_recon = P_before_ideal + pred_delta;

% 提取 Tip (最后 3 维)
tip_gt = P_after_gt_list(25:27, :);
tip_pred = P_recon(25:27, :);

figure('Name', 'Dynamic Reconstruction Tracking', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
dim_labels = {'X (m)', 'Y (m)', 'Z (m)'};
for i = 1:3
    subplot(3, 1, i);
    plot(time_list, tip_gt(i, :), 'k-', 'LineWidth', 1.5); hold on;
    plot(time_list, tip_pred(i, :), 'r--', 'LineWidth', 1.5);
    grid on; ylabel(dim_labels{i});
    if i==3, xlabel('Time (s)'); end
    legend('Truth', 'Prediction');
end

final_mae = mean(sqrt(sum((tip_gt - tip_pred).^2, 1))) * 1000;
fprintf('--------------------------------------------------\n');
fprintf('Analysis Complete. Dynamic Tip MAE: %.2f mm\n', final_mae);