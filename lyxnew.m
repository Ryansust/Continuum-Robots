%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author:  Lin Yongxi
%  Module:  Full Pipeline (Data -> Net B -> Net C -> Evaluation -> Hybrid)
% =========================================================================
clc; clear; close all;
rng('default'); % 确保结果可复现

%% ========================================================================
%  Step 1: Data Loading, ROI Filtering & Cleaning
% =========================================================================
disp('--------------------------------------------------');
disp('1. Loading and preprocessing data...');

FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(FileName), error('File not found!'); end
dataTable = readtable(FileName);

% --- 1.1 提取原始信号并转换单位 ---
% 建立 track_rows 数组，全生命周期追踪 Excel 原始行号
track_rows_raw = (3:height(dataTable))'; 

% 拉力必须乘以 0.00981 转换为 N
conversion_factor = 0.00981;
F_after_raw  = (double(table2array(dataTable(3:end, 23:28))) * conversion_factor)';  
F_before_raw = (double(table2array(dataTable(3:end, 11:16))) * conversion_factor)';  

raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 

% 假设 Excel 中有 before 和 after 两个形态文本列 (根据实际列号调整)
pos_text_before_raw = dataTable{3:end, 38}; 
pos_text_after_raw  = dataTable{3:end, 29}; 

% --- 1.2 ROI 筛选 (聚焦 Node 3, 4, 5) ---
disp('   > Executing ROI filtering (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw, [3, 4, 5]);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_b_sub = pos_text_before_raw(roi_mask);
pos_text_a_sub = pos_text_after_raw(roi_mask);
track_rows_sub = track_rows_raw(roi_mask); % 同步跟踪行号

if length(raw_mag_sub) < 50, error('Insufficient data after ROI filtering.'); end

% --- 1.3 数据清洗 (NaN 与 几何畸变检测) ---
disp('   > Executing Auto-Cleaning (NaNs & large distortion detection)...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | ...
          isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);

% 手动剔除已知异常样本
known_outliers = [686]; 
if ~isempty(known_outliers)[~, loc_outliers] = ismember(known_outliers, track_rows_sub);
    loc_outliers = loc_outliers(loc_outliers > 0);
    if ~isempty(loc_outliers)
        fprintf('   ⚠ 手动剔除已知异常样本 (Excel Row: %d)\n', known_outliers);
        bad_idx(loc_outliers) = true; 
    end
end

% 解析坐标并检测动捕跳变点 (>10mm)
N_sub = length(raw_mag_sub);
P_before_ideal = zeros(21, N_sub); 
P_after_sensor = zeros(21, N_sub);
gt_F_vec = zeros(3, N_sub);

for i = 1:N_sub
    if bad_idx(i), continue; end
    
    offset_b = get_RealOffset_1S3CT(pos_text_b_sub{i});
    offset_a = get_RealOffset_1S3CT(pos_text_a_sub{i});
    
    % 【严格物理对齐】：以 Base 中点为原点 (0,0,0)
    base_center_b = (offset_b(:, 1) + offset_b(:, 2)) / 2;
    body_b_aligned = offset_b(:, 3:end) - base_center_b;
    P_before_ideal(:, i) = reshape(body_b_aligned,[], 1); 
    
    base_center_a = (offset_a(:, 1) + offset_a(:, 2)) / 2;
    body_a_aligned = offset_a(:, 3:end) - base_center_a;
    P_after_sensor(:, i) = reshape(body_a_aligned,[], 1); 
    
    % 跳变检测
    pts = reshape(P_after_sensor(:, i), 3, 7);
    for j = 2:6
        mid_point = (pts(:, j-1) + pts(:, j+1)) / 2;
        if norm(pts(:, j) - mid_point) > 0.01 % 10mm 阈值
            bad_idx(i) = true;
            break;
        end
    end
    
    % 计算外力真值矢量
    u_vec = [0;0;0];
    switch raw_dir_sub(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec = [-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag_sub(i) * u_vec;
end

% 执行最终剔除
F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
P_before_clean = P_before_ideal(:, ~bad_idx);
P_after_clean  = P_after_sensor(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
gt_F_clean = gt_F_vec(:, ~bad_idx);
track_rows_clean = track_rows_sub(~bad_idx); 

F_diff = F_after - F_before;
N = length(raw_mag);
fprintf('   > Final effective samples: %d\n', N);

%% ========================================================================
%  Step 2: Data Augmentation
% =========================================================================
disp('--------------------------------------------------');
disp('2. Executing rotational augmentation (x3)...');[aug_F_diff, aug_F_after, aug_F_before, aug_Pb, aug_Pa, aug_gt_F, aug_hgt, aug_track_rows] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before_clean, P_after_clean, gt_F_clean, raw_hgt, track_rows_clean);

% 计算真实的形变残差作为 Net C 目标
aug_Delta_P = aug_Pa - aug_Pb;

%% ========================================================================
%  Step 3: Dataset Construction & Safety Check
% =========================================================================
disp('--------------------------------------------------');
disp('3. Constructing final training sets...');

inputs_f_final   =[aug_F_after; aug_F_diff; aug_F_before]; % Net B Force
targets_f_final  = aug_gt_F;

inputs_loc_final = [aug_F_diff; aug_F_after; aug_Pb]; % Net B Loc
targets_loc_final = double(aug_hgt) / 9.0; 

% 二次安全检查
bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1);
if sum(bad_total) > 0
    fprintf('   [Warning] Removing %d bad augmented samples.\n', sum(bad_total));
    inputs_f_final(:, bad_total) =[]; targets_f_final(:, bad_total) = [];
    inputs_loc_final(:, bad_total) =[]; targets_loc_final(:, bad_total) = [];
    aug_gt_F(:, bad_total) =[]; aug_Delta_P(:, bad_total) = [];
    aug_track_rows(bad_total) =[]; aug_Pb(:, bad_total) =[];
end

% 注入微小噪声防止 Z-Score 归一化除零
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > Augmented input samples: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation
% =========================================================================
disp('--------------------------------------------------');
disp('4. Training Net B (Force Regression)...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;
[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);
if any(isnan(pred_f(:))), error('Net B Force produced NaN!'); end
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force MAE: %.4f N\n', mae_f);

%% ========================================================================
%  Step 5: Net B - Location Sensing (Weighted Loss)
% =========================================================================
disp('--------------------------------------------------');
disp('5. Training Net B (Location Classification with Weighted Loss)...');

% 筛选高力值数据
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in = inputs_loc_final(:, v_mask);
raw_tg = targets_loc_final(:, v_mask);
raw_shape_tg = aug_Delta_P(:, v_mask); % 对应 Net C 目标
node_labels = round(raw_tg * 9.0);

% 计算逆频率权重
nodes_interest = [3, 4, 5];
weights_vec = ones(1, length(node_labels));
fprintf('   > Sample Distribution:\n');
for k = nodes_interest
    idx_k = (node_labels == k);
    count_k = sum(idx_k);
    if count_k > 0
        w_k = length(node_labels) / (length(nodes_interest) * count_k);
        weights_vec(idx_k) = w_k;
        fprintf('     - Node %d: %d samples, Weight: %.2f\n', k, count_k, w_k);
    end
end
[in_norm, ps_in] = mapstd(raw_in);[tg_norm, ps_out] = mapstd(raw_tg);

net_loc = fitnet([60, 30]); % 恢复最优参数
net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = false; 
net_loc.divideParam.testRatio = 0.0; % 手动测试
[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [],[], weights_vec);

%% ========================================================================
%  Step 6: Evaluation & Visualization (Net B)
% =========================================================================
disp('--------------------------------------------------');
disp('6. Evaluating Net B Location Performance...');

pred_val = mapstd('reverse', net_loc(mapstd('apply', raw_in, ps_in)), ps_out);
pred_node = pred_val * 9.0;
real_node = raw_tg * 9.0;

pred_node(pred_node < 3) = 3; pred_node(pred_node > 5) = 5;
rmse_node = sqrt(mean((pred_node - real_node).^2));
acc_strict = sum(round(pred_node) == round(real_node)) / length(real_node);

fprintf('   > [Location] RMSE: %.2f Segment\n', rmse_node);
fprintf('   > [Location] Strict Accuracy: %.2f%%\n', acc_strict * 100);

figure('Name', 'Net B: Location Results', 'Color', 'w', 'Position',[100, 100, 1000, 400]);
subplot(1, 2, 1);
jitter = (rand(size(pred_node))-0.5)*0.15;
scatter(real_node, pred_node+jitter, 30, abs(real_node-pred_node), 'filled', 'MarkerFaceAlpha', 0.7);
colormap(jet); caxis([0 1]); hold on; plot([2, 6], [2, 6], 'k--');
title(['Regression (RMSE: ', num2str(rmse_node, '%.2f'), ')']);
xlabel('Truth'); ylabel('Pred');

subplot(1, 2, 2);
cm = confusionchart(round(real_node), round(pred_node));
cm.Title = 'Confusion Matrix (Weighted)';
cm.RowSummary = 'row-normalized'; 
sortClasses(cm, 'ascending');

%% ========================================================================
%  Step 7: Net C - Shape Reconstruction (Residual Framework)
% =========================================================================
disp('--------------------------------------------------');
disp('7. Training Net C (Residual Shape Reconstruction)...');

feat_internal = aug_F_after(:, v_mask);
feat_external = aug_gt_F(:, v_mask); 
feat_location = double(aug_hgt(:, v_mask)) / 9.0;
feat_P_before = aug_Pb(:, v_mask); 

inputs_net_c =[feat_internal; feat_external; feat_location; feat_P_before];
targets_net_c = raw_shape_tg; % 目标为 Delta_P
track_rows_net_c = aug_track_rows(v_mask);[in_c_norm, ps_in_c] = mapstd(inputs_net_c);[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

net_shape = fitnet([80, 60, 40]); 
net_shape.trainFcn = 'trainscg'; 
net_shape.trainParam.showWindow = false;
net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio   = 0.1;
net_shape.divideParam.testRatio  = 0.1;
[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8: Net C Evaluation
% =========================================================================
disp('--------------------------------------------------');
disp('8. Evaluating Net C Performance...');

test_idx = tr_c.testInd;
if isempty(test_idx), test_idx = randperm(size(inputs_net_c,2), 50); end

in_test = inputs_net_c(:, test_idx);
target_delta_test = targets_net_c(:, test_idx);
p_before_test = feat_P_before(:, test_idx);
rows_test = track_rows_net_c(test_idx);

% 预测残差并还原绝对坐标
pred_delta = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);
pred_P_after = p_before_test + pred_delta;
real_P_after = p_before_test + target_delta_test;

% Mean Shape Error
dist_errs = zeros(1, length(test_idx));
tip_dist  = zeros(1, length(test_idx));
for i = 1:length(test_idx)
    p_p = reshape(pred_P_after(:, i), 3,[]);
    p_r = reshape(real_P_after(:, i), 3,[]);
    dist_errs(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
    tip_dist(i)  = norm(p_p(:, end) - p_r(:, end));
end
mean_dist = mean(dist_errs);
tip_mae = mean(tip_dist);
fprintf('   > [Net C] Mean Shape Error: %.4f m (%.2f mm)\n', mean_dist, mean_dist*1000);
fprintf('   > [Net C] Tip MAE: %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);

%% ========================================================================
%  Step 9: Worst Case Analysis (Base Centered View)
% =========================================================================
disp('--------------------------------------------------');
disp('9. Analyzing the Top Worst Cases...');

num_worst = 5;[sorted_errors, sorted_indices] = sort(tip_dist, 'descend'); 
num_worst = min(num_worst, length(tip_dist));
worst_indices_local = sorted_indices(1:num_worst); 

for k = 1:num_worst
    loc_idx = worst_indices_local(k);
    orig_row = rows_test(loc_idx); 
    err_val = sorted_errors(k) * 1000;
    
    P_p = [[0;0;0], reshape(pred_P_after(:, loc_idx), 3,[])];
    P_r = [[0;0;0], reshape(real_P_after(:, loc_idx), 3,[])];
    
    figure('Name', sprintf('Worst Case %d - Row: %d', k, orig_row), 'Color', 'w', 'Position',[100+k*20, 100+k*20, 600, 500]);
    hold on; grid on; axis equal;
    
    plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y');
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth');
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--o', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', 'Prediction');
    
    tip_t = P_r(:, end); tip_p = P_p(:, end);
    plot3([tip_t(1), tip_p(1)],[tip_t(2), tip_p(2)],[tip_t(3), tip_p(3)], 'm-', 'LineWidth', 2.5, 'DisplayName', sprintf('Tip Error: %.1fmm', err_val));
    
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Worst Case | Excel Row: %d', orig_row), 'FontSize', 12);
    legend('Location', 'best'); view(30, 20);
end

%% ========================================================================
%  Step 10: Hybrid Sim2Real Evaluation (Net B + CSBCM)
% =========================================================================
disp('--------------------------------------------------');
disp('10. Evaluating Hybrid Framework (Net B + CSBCM)...');

tendon_count = 3;         
section_num  = 2;        
D_val        = 0.0006;         
E_val        = 1.016e+12;      
L_a_val      = 0.09;         
L_b_val      = 0.00;         
N_d_val      = 7;            
H_list_val   = linspace(0.0025, 0.0025, section_num*N_d_val+1); 
mu_val       = 0.25;          
delta_alpha_val = 0; 
G_load_val   = 4.000 * 0.00981;  

inputs_f_aligned = inputs_f_final(:, v_mask);
num_physics_test = min(10, length(test_idx));

% 【修复索引错误】随机生成在测试集中的“局部索引”
local_hybrid_ids = randperm(length(test_idx), num_physics_test);

hybrid_errors = zeros(1, num_physics_test);
figure('Name', 'Hybrid Framework', 'Color', 'w', 'Position',[100, 100, 1200, 400]);

for k = 1:num_physics_test
    % 获取局部索引和全局索引
    loc_idx = local_hybrid_ids(k);    % 在测试集(例如长度198)中的局部位置
    glob_idx = test_idx(loc_idx);     % 映射到全局完整集中的真实位置
    
    % 提取真实驱动力 (用 glob_idx 提取全局变量)
    F_tendon_real = feat_internal(:, glob_idx)' / 0.00981; 
    
    % 提取真实对齐坐标 (用 loc_idx 提取测试集变量)
    P_truth_aligned = [[0;0;0], reshape(real_P_after(:, loc_idx), 3,[])]; 
    
    % Net B 推理 (用 glob_idx)
    in_f = inputs_f_aligned(:, glob_idx);
    pred_F_ext = net_force(in_f); 
    
    in_loc = mapstd('apply', raw_in(:, glob_idx), ps_in); 
    pred_loc_val = mapstd('reverse', net_loc(in_loc), ps_out);
    pred_node = round(pred_loc_val * 9.0);
    if pred_node < 3, pred_node = 3; end; if pred_node > 5, pred_node = 5; end
    
    % 建立 Marker 到 物理 Disk 的映射 (需根据实际修改)
    switch pred_node
        case 3, touch_id_csbcm = 5;
        case 4, touch_id_csbcm = 7;
        case 5, touch_id_csbcm = 10;
        otherwise, touch_id_csbcm = 7;
    end
    
    fprintf('   > Sample %d: Net B 预测外力 [%.2f, %.2f, %.2f]N, 盘片 %d. 求解物理引擎...\n', ...
        k, pred_F_ext(1), pred_F_ext(2), pred_F_ext(3), touch_id_csbcm);
    
    try
        [P_Theo, ~, ~, ~, ~, ~] = solve_continuum_shape(tendon_count, section_num, ...
            D_val, E_val, L_a_val, L_b_val, N_d_val, H_list_val, mu_val, ...
            delta_alpha_val, G_load_val, pred_F_ext, F_tendon_real, touch_id_csbcm);
        
        tip_theo = P_Theo(:, end);
        tip_true = P_truth_aligned(:, end);
        hybrid_errors(k) = norm(tip_theo - tip_true) * 1000; 
        
        if k <= 3 
            subplot(1, 3, k); hold on; grid on; axis equal;
            quiver3(0,0,0, 0.05,0,0, 'r', 'LineWidth', 2); 
            quiver3(0,0,0, 0,0.05,0, 'g', 'LineWidth', 2); 
            quiver3(0,0,0, 0,0,0.05, 'b', 'LineWidth', 2);
            plot3(P_truth_aligned(1,:), P_truth_aligned(2,:), P_truth_aligned(3,:), 'k-s', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
            plot3(P_Theo(1,:), P_Theo(2,:), P_Theo(3,:), 'b--o', 'LineWidth', 2, 'DisplayName', 'Net B + CSBCM');
            xlabel('X'); ylabel('Y'); zlabel('Z'); set(gca, 'ZDir', 'reverse', 'YDir', 'reverse'); view(30, 20);
            title(sprintf('Hybrid Err: %.1f mm', hybrid_errors(k)));
            if k==1, legend('Location', 'best'); end
        end
    catch ME
        fprintf('   ⚠ 求解失败: %s\n', ME.message);
        hybrid_errors(k) = NaN;
    end
end

valid_errors = hybrid_errors(~isnan(hybrid_errors));
if ~isempty(valid_errors)
    fprintf('\n✅ [Hybrid] Net B + CSBCM 平均末端误差: %.2f mm\n', mean(valid_errors));
else
    disp('❌ 物理模型全部求解失败。');
end
%% ========================================================================
%  Step 11: Save Checkpoint
% =========================================================================
save('Final_System_Checkpoint.mat', 'net_force', 'net_loc', 'net_shape', ...                 
     'ps_in', 'ps_out', 'ps_in_c', 'ps_out_c', 'test_idx', 'v_mask', ...
     'inputs_f_final', 'targets_f_final', 'inputs_loc_final', 'targets_loc_final', ...
     'inputs_net_c', 'targets_net_c', 'feat_P_before');
disp('>>> All done. Checkpoint saved successfully.');

%% ========================================================================
%  Helper Function: Data Augmentation
% =========================================================================
function[aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_Pa, aug_gF, aug_h, aug_tr] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, P_after, gt_F, hgt, track_rows)
    N = size(F_diff, 2);
    R120 =[cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 =[cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    idx120 =[5, 6, 1, 2, 3, 4]; idx240 =[3, 4, 5, 6, 1, 2];
    rotP = @(P, R) reshape(R * reshape(P, 3,[]), 21, N);
    
    aug_Fd =[F_diff, F_diff(idx120,:), F_diff(idx240,:)];
    aug_Fa = [F_after, F_after(idx120,:), F_after(idx240,:)];
    aug_Fb =[F_before, F_before(idx120,:), F_before(idx240,:)];
    aug_Pb =[P_before, rotP(P_before, R120), rotP(P_before, R240)];
    aug_Pa =[P_after, rotP(P_after, R120), rotP(P_after, R240)]; 
    aug_gF =[gt_F, R120*gt_F, R240*gt_F]; 
    aug_h  = [hgt, hgt, hgt];
    aug_tr =[track_rows; track_rows; track_rows];
end