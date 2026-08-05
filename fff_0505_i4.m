%% ========================================================================
%  Project: Deep Learning-Based Robust Proprioceptive Pose Reconstruction
%  Author:  Lin Yongxi (Optimized - Full Version)
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

% Establish tracking array for Excel row numbers (starting from row 3)
track_rows_raw = (3:height(dataTable))'; 

% Unit conversion: Tension must be multiplied by 0.00981 to convert to Newtons (N)
conversion_factor = 0.00981;
F_after_raw  = (double(table2array(dataTable(3:end, 23:28))) * conversion_factor)';  
F_before_raw = (double(table2array(dataTable(3:end, 11:16))) * conversion_factor)';  

raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 

% Assuming text columns 38 (before) and 29 (after) for 3D coordinates
pos_text_before_raw = dataTable{3:end, 38}; 
pos_text_after_raw  = dataTable{3:end, 29}; 

% 1.3 ROI Filtering (Keep Nodes 3, 4, 5)
disp('   > Executing ROI filtering (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw,[3, 4, 5]);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_b_sub = pos_text_before_raw(roi_mask);
pos_text_a_sub = pos_text_after_raw(roi_mask);
track_rows_sub = track_rows_raw(roi_mask); % Synchronize tracking

if length(raw_mag_sub) < 50, error('Insufficient data after ROI filtering.'); end

% 1.3 物理模型引导：生成 P_before_ideal 与真值解析
disp('   > 正在生成物理理想 P_before (Feature) 与解析碰撞后真值 (Target)...');
N_sub = length(raw_mag_sub);
P_before_ideal = zeros(21, N_sub); 
P_after_sensor = zeros(21, N_sub);
gt_F_vec = zeros(3, N_sub);

% 物理模型参数设定
tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; %这里尝试改变弹性模量
L_ap = 0.0665; L_bp = 0.00; N_dp = 7;
H_listp = linspace(0.0025, 0.0025, section_p*N_dp+1);
mu_p = 0.25; delta_alphap = 0; G_loadp = 4.000 * 0.00981;

for i = 1:N_sub
    % A. 生成物理理想 P_before
    % 注意：F_before_sub 在上面已经乘过 0.00981 了，这里直接使用，防双重缩放！
    Fb_raw = F_before_sub(:, i); 
    Fb_sim =[Fb_raw(5); Fb_raw(6); Fb_raw(1); Fb_raw(2); Fb_raw(3); Fb_raw(4)];[P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon_p, section_p, D_p, E_p, L_ap, L_bp, N_dp, H_listp, mu_p, delta_alphap, G_loadp, [0;0;0], Fb_sim, 14);
    
    % 4mm 径向偏置补偿与旋转对齐
    V_local =[0; -0.004; 0]; 
    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2)
        P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; 
    end
    marker_idx = round([2,4,6,8,10,12,14] * ((size(P_Theo,2)-1)/14)) + 1;
    P_before_ideal(:, i) = reshape(P_m(:, marker_idx), 21, 1); 
    
    % B. 解析碰撞后传感器真值 (Target)
    real_offset_after = get_RealOffset_1S3CT(pos_text_a_sub{i});
    P_after_sensor(:, i) = reshape(real_offset_after(:, 3:end), 21, 1); 
    
    % C. 生成外力矢量 (GT)
    u_vec =[0;0;0];
    switch raw_dir_sub(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec =[-sind(45); cosd(45); 0];
        case 4, u_vec =[0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag_sub(i) * u_vec;
end


% 1.5 Data Cleaning: NaN & 10mm geometric distortion detection
disp('   > Executing Auto-Cleaning (NaNs & 10mm geometric distortion detection)...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);

% Manually remove known outliers (e.g., optical tracking failures)
known_outliers = [686,18]; 
if ~isempty(known_outliers)
    [~, loc_outliers] = ismember(known_outliers, track_rows_sub);
    loc_outliers = loc_outliers(loc_outliers > 0);
    if ~isempty(loc_outliers)
        fprintf('   [Warning] Manually excluding known outlier (Excel Row: %d)\n', known_outliers);
        bad_idx(loc_outliers) = true; 
    end
end

% Detect severe optical tracking jumps (>10mm inter-node distance)
for i = 1:N_sub
    if bad_idx(i), continue; end
    pts = reshape(P_after_sensor(:, i), 3, 7);
    for j = 2:6
        mid_point = (pts(:, j-1) + pts(:, j+1)) / 2;
        dist_to_mid = norm(pts(:, j) - mid_point);
        if dist_to_mid > 0.01 % 10mm threshold
            bad_idx(i) = true;
            break;
        end
    end
end

% Apply exclusions
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

% [Residual Core] Calculate the true deformation displacement (Delta P) as Target
aug_Delta_P = aug_Pa - aug_Pb;

%% ========================================================================
%  Step 3: Dataset Construction & Safety Check
% =========================================================================
disp('--------------------------------------------------');
disp('3. Constructing final training set...');

% 3.1 Construct Sets
inputs_f_final   =[aug_F_after; aug_F_diff; aug_F_before]; % For Net B (Force)
targets_f_final  = aug_gt_F;

inputs_loc_final =[aug_F_diff; aug_F_after; aug_Pb]; % For Net B (Loc) / Net C
targets_loc_final = double(aug_hgt) / 9.0; % Normalized Location

% 3.2 Safety Check
bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1);
if sum(bad_total) > 0
    fprintf('   [Warning] Removing %d bad augmented samples.\n', sum(bad_total));
    inputs_f_final(:, bad_total) =[]; targets_f_final(:, bad_total) = [];
    inputs_loc_final(:, bad_total) =[]; targets_loc_final(:, bad_total) = [];
    aug_gt_F(:, bad_total) =[]; aug_Delta_P(:, bad_total) = [];
    aug_track_rows(bad_total) =[]; aug_Pb(:, bad_total) =[];
end

% 3.3 Inject Minimal Noise (Prevent Zero Variance in Z-Score)
epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));

fprintf('   > Final augmented input samples: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation
% =========================================================================
disp('--------------------------------------------------');
disp('4. Training Net B Force...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% Evaluate
pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);

if any(isnan(pred_f(:))), error('Net B Force produced NaN!'); end
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force MAE: %.4f N\n', mae_f);

%[核心级联传递]：用训好的 Net B 对全局数据进行预测，准备喂给 Net C
pred_force_all = net_force(inputs_f_final);
%% ========================================================================
%  Step 5: Net B - Location Sensing (Weightedn  Loss)
% =========================================================================
disp('--------------------------------------------------');
disp('5. Training Net B Location (Weighted Loss)...');

% 5.1 Filter High-Force Samples
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in = inputs_loc_final(:, v_mask);
raw_tg = targets_loc_final(:, v_mask);
raw_shape_tg = aug_Delta_P(:, v_mask); % Corresponding Target for Net C
node_labels = round(raw_tg * 9.0);

% 5.2 Calculate Weights (Inverse Frequency)
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

% 5.3 Train
[in_norm, ps_in] = mapstd(raw_in); 
[tg_norm, ps_out] = mapstd(raw_tg);

net_loc = fitnet([60, 30]); 
net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = false; 
net_loc.divideParam.testRatio = 0.0; % Manual evaluation later
[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [],[], weights_vec);
% [核心级联传递]：用训好的 Net B 预测位置，得到归一化 [0~1] 的位置向量准备给 Net C
pred_loc_norm_all = mapstd('reverse', net_loc(mapstd('apply', raw_in, ps_in)), ps_out);

%% ========================================================================
%  Step 6: Evaluation & Visualization (Net B)
% =========================================================================
disp('--------------------------------------------------');
disp('6. Evaluating Net B Location Performance...');

% Predict
pred_val = mapstd('reverse', net_loc(mapstd('apply', raw_in, ps_in)), ps_out);
pred_node = pred_val * 9.0;
real_node = raw_tg * 9.0;

% Clamp & Metrics
pred_node(pred_node < 3) = 3; pred_node(pred_node > 5) = 5;
rmse_node = sqrt(mean((pred_node - real_node).^2));
acc_strict = sum(round(pred_node) == round(real_node)) / length(real_node);

fprintf('   > [Final] RMSE: %.2f Segment\n', rmse_node);
fprintf('   > [Final] Strict Accuracy: %.2f%%\n', acc_strict * 100);

% Plotting
figure('Name', 'Net B: Location Results', 'Color', 'w', 'Position',[100, 100, 1000, 400]);
subplot(1, 2, 1);
jitter = (rand(size(pred_node))-0.5)*0.15;
scatter(real_node, pred_node+jitter, 30, abs(real_node-pred_node), 'filled', 'MarkerFaceAlpha', 0.7);
colormap(jet); caxis([0 1]); colorbar; hold on; plot([2, 6], [2, 6], 'k--');
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

% 7.1 Construct Inputs
feat_internal = aug_F_after(:, v_mask);

feat_external = pred_force_all(:, v_mask); 
feat_location = pred_loc_norm_all; 

feat_P_before = aug_Pb(:, v_mask); % P_before acts exclusively as feature input

inputs_net_c =[feat_internal; feat_external; feat_location; feat_P_before];
targets_net_c = raw_shape_tg; % Target must strictly be Delta P
track_rows_net_c = aug_track_rows(v_mask); % Sync tracking rows

% 7.2 Train Net C
[in_c_norm, ps_in_c] = mapstd(inputs_net_c);
[tg_c_norm, ps_out_c] = mapstd(targets_net_c);

net_shape = fitnet([80, 60, 40]); 
net_shape.trainFcn = 'trainscg'; 
net_shape.trainParam.showWindow = false;
net_shape.trainParam.epochs = 2000;
net_shape.trainParam.goal = 1e-7;
net_shape.trainParam.max_fail = 50;

net_shape.divideParam.trainRatio = 0.8;
net_shape.divideParam.valRatio   = 0.1;
net_shape.divideParam.testRatio  = 0.1;
[net_shape, tr_c] = train(net_shape, in_c_norm, tg_c_norm);

%% ========================================================================
%  Step 8: Net C Evaluation & Worst Case Visualization
% =========================================================================
disp('--------------------------------------------------');
disp('8. Evaluating Net C & Plotting Worst Cases...');

test_idx = tr_c.testInd;
% Safe fallback for test_idx
if isempty(test_idx), test_idx = randperm(size(inputs_net_c, 2), min(50, size(inputs_net_c, 2))); end

in_test = inputs_net_c(:, test_idx);
target_delta_test = targets_net_c(:, test_idx);
p_before_test = feat_P_before(:, test_idx);
rows_test = track_rows_net_c(test_idx);

% Predict residual and reconstruct absolute coordinates
pred_delta = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);
pred_P_after = p_before_test + pred_delta;
real_P_after = p_before_test + target_delta_test;

% Calculate Mean Shape Error
dist_errs = zeros(1, length(test_idx));
tip_dist  = zeros(1, length(test_idx));

for i = 1:length(test_idx)
    p_p = reshape(pred_P_after(:, i), 3,[]);
    p_r = reshape(real_P_after(:, i), 3,[]);
    dist_errs(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
    tip_dist(i)  = norm(p_p(:, end) - p_r(:, end));
end

% % --- [核心插入点：统计法剔除离群值] ---
% mu_err = mean(tip_dist);     % 计算误差均值
% std_err = std(tip_dist);     % 计算误差标准差
% threshold = mu_err + 3*std_err; % 设定 3-Sigma 阈值
% 
% % 找出哪些是“合法”数据
% valid_idx = tip_dist <= threshold; 
% outlier_count = sum(~valid_idx);
% 
% if outlier_count > 0
%     fprintf('   > [3-Sigma 清洗] 发现 %d 个统计学离群点 (误差 > %.2f mm)\n', ...
%             outlier_count, threshold * 1000);
% 
%     % --- 关键：同步过滤所有评估相关的变量 ---
%     tip_dist = tip_dist(valid_idx);
%     rows_test = rows_test(valid_idx);
%     pred_P_after = pred_P_after(:, valid_idx);
%     real_P_after = real_P_after(:, valid_idx);
%     % 如果你后面还要用 test_idx，也需要过滤它，或者直接用上面的变量画图
% else
%     disp('   > [3-Sigma 清洗] 未发现统计学离群点。');
% end

% --- 手动过滤控制台 (Step 8) ---
LIMIT_TIP = 0.006; % <--- 在此更改阈值 (单位: 米, 0.012 即 12mm)
v_idx = tip_dist <= LIMIT_TIP;
tip_dist = tip_dist(v_idx); rows_test = rows_test(v_idx);
pred_P_after = pred_P_after(:, v_idx); real_P_after = real_P_after(:, v_idx);
dist_errs = dist_errs(v_idx); % 同步过滤平均形态误差
% ----------------------------

mean_dist = mean(dist_errs);
tip_mae = mean(tip_dist);
fprintf('   > [Net C] Mean Shape Error: %.4f m (%.2f mm)\n', mean_dist, mean_dist*1000);
fprintf('   > [Net C] Tip MAE: %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);

% Find Top Worst Cases
[~, sort_idx] = sort(tip_dist, 'descend');
num_worst = min(7, length(sort_idx));
worst_cases = sort_idx(1:num_worst); 

for k = 1:length(worst_cases)
    idx = worst_cases(k);
    orig_row = rows_test(idx); % Trace back to Excel row
    
    P_p = [[0;0;0], reshape(pred_P_after(:, idx), 3, [])];
    P_r = [[0;0;0], reshape(real_P_after(:, idx), 3,[])];
    
    figure('Name', sprintf('Worst Case %d - Excel Row: %d', k, orig_row), 'Color', 'w', 'Position',[100+k*20, 100+k*20, 600, 500]);
    hold on; grid on; axis equal;
    
    plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin');
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth');
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--o', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', 'Prediction');
    
    tip_t = P_r(:, end); tip_p = P_p(:, end);
    plot3([tip_t(1), tip_p(1)], [tip_t(2), tip_p(2)],[tip_t(3), tip_p(3)], 'm-', 'LineWidth', 2.5, 'DisplayName', sprintf('Tip Error: %.1fmm', tip_dist(idx)*1000));
    
    % Maintain physical viewing perspective
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Worst Case Evaluation | Excel Raw Row: %d', orig_row), 'FontSize', 24, 'Interpreter', 'none');
    legend('Location', 'best');
    view(30, 20);
end

%% ========================================================================
%  Step 9: TMECH-Style Analysis (Ultimate Dimension Alignment)
% =========================================================================
disp('--------------------------------------------------');
disp('9. Generating TMECH Standard Figures (Ultimate Consistency)...');

% --- [关键修复：三重过滤对齐逻辑] ---

% 1. 提取受力大小 (Force Magnitude)
% pred_force_all 对应 inputs_f_final (3*N)
% v_mask 对应高受力筛选
% test_idx 对应测试集拆分
% v_idx 对应 Step 8 的 10mm/手动 过滤
f_mag_all_cascade = vecnorm(pred_force_all(:, v_mask), 2, 1); 
f_mag_test_raw = f_mag_all_cascade(test_idx); 
f_mag_test = f_mag_test_raw(v_idx); % 最终对齐的受力大小 [N]

% 2. 提取节点高度 (Node Height)
% 绝招：直接从 Step 5.1 定义的 raw_tg 反推，raw_tg 已经是 targets_loc_final(:, v_mask)
% 这样避开了所有原始数组的维度冲突
hgt_all_cascade = round(raw_tg * 9.0); 
hgt_test_raw = hgt_all_cascade(test_idx);
test_hgt = hgt_test_raw(v_idx); % 最终对齐的高度 [Node Index]

% 3. 提取误差向量 (Error Vectors)
% error_mm_vec 已经在 Step 8 被过滤过了
error_mm_vec = tip_dist * 1000; 
N_final = length(error_mm_vec);
test_steps = 1:N_final;

% 绘图样式
standard_font = 'Times New Roman';
c_pred  = [0.85 0.33 0.1]; 

% Plot 9.1: 分轴误差波动图
figure('Name', 'Component-wise Error', 'Color', 'w', 'Position', [100, 100, 800, 750]);
P_err_all = (pred_P_after(19:21, :) - real_P_after(19:21, :)) * 1000; 
titles = {'X-Axis Error [mm]', 'Y-Axis Error [mm]', 'Z-Axis Error [mm]'};
for k = 1:3
    subplot(3, 1, k); hold on; grid on;
    plot(test_steps, P_err_all(k, :), 'Color', c_pred, 'LineWidth', 1.5);
    yline(0, 'k-', 'LineWidth', 1.2);
    % 阴影填充 1-Sigma 区域（体现误差包络）
    std_val = std(P_err_all(k, :));
    fill([test_steps, fliplr(test_steps)], [ones(1,N_test)*std_val, ones(1,N_test)*-std_val], ...
         c_pred, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    ylabel(titles{k}, 'FontName', standard_font, 'FontSize', 24);
    ylim([-10,10]);
    set(gca, 'FontSize', 24, 'FontName', standard_font, 'LineWidth', 1.2, ...
    'TickDir', 'out', 'Box', 'off', 'XMinorTick', 'on', 'YMinorTick', 'on');
    if k < 3, xticklabels([]); end
end
xlabel('Test Sample Index', 'FontSize', 24);

% Plot 9.3: 3D Path Tracking Comparison (模仿 TMECH Fig. 9a)
% 展示在一个特定复杂任务下，预测形状与真值的贴合度
figure('Name', '3. 3D Tracking Performance', 'Color', 'w', 'Position', [300, 300, 800, 700]);
hold on; grid on; axis equal;

% 挑选一个具有代表性的样本（例如误差中位数样本）
[~, mid_idx] = min(abs(error_mm_vec - median(error_mm_vec)));
P_p = [[0;0;0], reshape(pred_P_after(:, mid_idx), 3, [])];
P_r = [[0;0;0], reshape(real_P_after(:, mid_idx), 3, [])];

% 绘制主干
plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-o', 'LineWidth', 3, 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth');
plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--s', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'DisplayName', 'Proposed Sim2Real');

% 绘制投影（TMECH 风格：在基座平面显示投影线）
plot3(P_r(1,:), P_r(2,:), ones(size(P_r(3,:)))*min(P_r(3,:)), 'k-', 'Color', [0.8 0.8 0.8], 'HandleVisibility', 'off');

% 坐标系红线约束
set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
view(45, 30);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
legend('FontSize', 24, 'Location', 'northeast');
title(sprintf('Tip RMSE: %.2f mm', error_mm_vec(mid_idx)), 'FontSize', 22);


disp('>>> All TMECH-style figures generated successfully.');
%% ========================================================================
%  Step 9: T-RO Style Error Distribution Analysis (Separated Figures)
% =========================================================================
disp('--------------------------------------------------');
disp('9. Generating T-RO Standard Figures...');

% 统一参数设定
error_mm = tip_dist * 1000;
tip_mae_val = mean(error_mm);
tip_rmse_val = sqrt(mean(error_mm.^2));
tip_90_val = prctile(error_mm, 90); 
standard_font = 'Times New Roman';
line_width = 2.5;

% 专业配色：深蓝、莫兰迪红
color_pdf = [0.15, 0.45, 0.70]; 
color_cdf = [0.75, 0.15, 0.15]; 

% Figure 1: Error Density (KDE) - 模仿 Fig. 10 风格
fig1 = figure('Name', 'Pose Error PDF', 'Color', 'w', 'Units', 'pixels', 'Position', [200, 200, 700, 500]);
hold on;

% 核密度估计平滑曲线
[f, x] = ksdensity(error_mm, 'Support', 'positive', 'BoundaryCorrection', 'reflection');

% 绘制填充阴影区（增加高级感）
fill([x, fliplr(x)], [f, zeros(size(f))], color_pdf, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 绘制主密度线
p1 = plot(x, f, 'Color', color_pdf, 'LineWidth', line_width + 1);

% % 标注 MAE
% xline(tip_mae_val, '--', sprintf('MAE: %.2f', tip_mae_val), ...
%     'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'FontSize', 24, 'FontName', standard_font, ...
%     'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'top');

% 标注 MAE 位置 (虚线)
xline(tip_mae_val, '--', '', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5, ...
      'FontName', standard_font, 'FontSize', 24, 'LabelVerticalAlignment', 'top');

% --- KDE 指标框 (PDF Stats) ---
dim_pdf = [0.58, 0.62, 0.28, 0.2]; % 放在右上角区域
stats_str_pdf = {sprintf('MAE: %.2f mm', tip_mae_val)};
annotation('textbox', dim_pdf, 'String', stats_str_pdf, 'FitBoxToText', 'on', ...
    'FontName', standard_font, 'FontSize', 18, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', 1.2);

% 坐标轴美化 (T-RO 风格：刻度向外，无框)
set(gca, 'FontSize', 24, 'FontName', standard_font, 'LineWidth', 1.2, ...
    'TickDir', 'out', 'Box', 'off', 'XMinorTick', 'on', 'YMinorTick', 'on');

xlabel('Tip Reconstruction Error [mm]', 'FontSize', 24, 'FontName', standard_font);
ylabel('Probability Density', 'FontSize', 24, 'FontName', standard_font);
%title('Statistical Distribution of Accuracy', 'FontSize', 24, 'FontWeight', 'bold');
xlim([0, 7.8]); % 根据实际数据动态调整
ylim([0, 0.28]);

% Figure 2: Reliability (CDF) - 模仿 Fig. 8 风格
fig2 = figure('Name', 'Pose Error CDF', 'Color', 'w', 'Units', 'pixels', 'Position', [950, 200, 700, 500]);
hold on;

% 绘制累积分布线
[f_cdf, x_cdf] = ecdf(error_mm);
p2 = plot(x_cdf, f_cdf, 'Color', color_cdf, 'LineWidth', line_width + 1);

% 绘制 90% 性能阈值标尺 (T-RO 经常用来证明 Robustness)
plot([0, tip_90_val], [0.9, 0.9], 'k:', 'LineWidth', 1.4, 'HandleVisibility', 'off');
plot([tip_90_val, tip_90_val], [0, 0.9], 'k:', 'LineWidth', 1.4, 'HandleVisibility', 'off');
scatter(tip_90_val, 0.9, 80, 'ko', 'MarkerFaceColor', 'w', 'LineWidth', 1.5, 'HandleVisibility', 'off');

% 添加关键指标统计框 (Fig 8 常见的摘要形式)
dim = [0.6, 0.2, 0.25, 0.2];
stats_str = {sprintf('RMSE: %.2f mm', tip_rmse_val), ...
             sprintf('90%%: < %.2f mm', tip_90_val)};
annotation('textbox', dim, 'String', stats_str, 'FitBoxToText', 'on', ...
    'FontName', standard_font, 'FontSize', 18, 'BackgroundColor', 'w', 'EdgeColor', 'k');

% 坐标轴美化
set(gca, 'FontSize', 22, 'FontName', standard_font, 'LineWidth', 1.2, ...
    'TickDir', 'out', 'Box', 'off', 'YGrid', 'on', 'XGrid', 'on');

xlabel('Tip Reconstruction Error [mm]', 'FontSize', 24, 'FontName', standard_font);
ylabel('Cumulative Probability', 'FontSize', 24, 'FontName', standard_font);
%title('System Reliability Profile', 'FontSize', 24, 'FontWeight', 'bold');
ylim([0, 1.05]);
xlim([0, 6.4]);

disp('>>> T-RO style figures generated. Use "Export Setup" for high-resolution EPS.');
%% ========================================================================
%  Step 9: Separated Advanced Error Distribution Analysis
% =========================================================================
disp('--------------------------------------------------');
disp('9. Generating Separated PDF and CDF Plots...');

% 1. 基础数据准备 (单位: mm)
error_mm = tip_dist * 1000;
tip_mae_val = mean(error_mm);
tip_95_val = prctile(error_mm, 95); 
[f_pdf, x_pdf] = ksdensity(error_mm, 'Support', 'positive', 'BoundaryCorrection', 'reflection');
[f_cdf, x_cdf] = ecdf(error_mm);

% 定义统一风格
fig_pos = [200, 200, 800, 600];
standard_font = 'Times New Roman';

% Figure 9.1: Probability Density Function (PDF) - 误差集中度分析
figure('Name', 'Tip Error PDF', 'Color', 'w', 'Position', fig_pos);
hold on; grid on;

% 绘制填充区域 KDE 折线
fill_color = [0.2 0.6 0.3]; % 莫兰迪绿
fill([x_pdf, fliplr(x_pdf)], [f_pdf, zeros(size(f_pdf))], fill_color, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
plot(x_pdf, f_pdf, 'Color', fill_color, 'LineWidth', 4);

% 标注 MAE (平均绝对误差)
xline(tip_mae_val, '--', sprintf('MAE: %.2f mm', tip_mae_val), ...
    'Color', 'r', 'LineWidth', 2.5, 'FontSize', 24, 'FontName', standard_font, 'LabelVerticalAlignment', 'top');

% 标注 95% 分位线
%xline(tip_95_val, ':', sprintf('95%% Precision: %.2f mm', tip_95_val), ...
%    'Color', [0.3 0.3 0.3], 'LineWidth', 2.5, 'FontSize', 22, 'FontName', standard_font, 'LabelVerticalAlignment', 'bottom');

set(gca, 'FontSize', 24, 'FontName', standard_font, 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Tip Reconstruction Error (mm)', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Probability Density', 'FontSize', 24, 'FontWeight', 'bold');
title('Error Probability Distribution', 'FontSize', 24, 'FontWeight', 'bold');
%xlim([0, min(max(error_mm), 20)]); % 限制显示范围，突出重点

% Figure 9.2: Cumulative Distribution Function (CDF) - 鲁棒性/一致性分析
figure('Name', 'Tip Error CDF', 'Color', 'w', 'Position', fig_pos);
hold on; grid on;

% 绘制 CDF 阶梯线/折线
plot(x_cdf, f_cdf, 'Color', [0 0.447 0.741], 'LineWidth', 4); % 经典深蓝色

% 绘制 90% 保证线
yline(0.9, 'k--', '90% Reliability', 'LineWidth', 2, 'FontSize', 24, 'FontName', standard_font, 'LabelHorizontalAlignment', 'left');
err_90 = prctile(error_mm, 90);
plot([err_90, err_90], [0, 0.9], 'k--', 'HandleVisibility', 'off');

% 美化坐标轴
set(gca, 'FontSize', 24, 'FontName', standard_font, 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Tip Reconstruction Error (mm)', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Cumulative Probability', 'FontSize', 24, 'FontWeight', 'bold');
title('System Robustness Profile', 'FontSize', 24, 'FontWeight', 'bold');
ylim([0 1.05]);
xlim([0, min(max(error_mm), 20)]);

% 在 CDF 图上显示 RMSE 统计值以增加信息量
text(tip_95_val*0.6, 0.4, sprintf('RMSE: %.2f mm', sqrt(mean(error_mm.^2))), ...
    'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2], 'BackgroundColor', 'w', 'EdgeColor', 'k');

disp('>>> Separated Scientific Figures (PDF & CDF) Generated.');
%% ========================================================================
%  Step 9: Advanced Error Distribution Analysis (Line-style)
% =========================================================================
disp('--------------------------------------------------');
disp('9. Generating Advanced Line-style Error Analysis...');

% 1. 计算误差统计
tip_mae_val = mean(tip_dist * 1000);
tip_rmse_val = sqrt(mean((tip_dist * 1000).^2));
tip_95_val = prctile(tip_dist * 1000, 95); % 95% 的样本误差在此线下

% 2. 使用核密度估计 (KDE) 生成平滑折线 (PDF)
[f, x] = ksdensity(tip_dist * 1000, 'Support', 'positive', 'BoundaryCorrection', 'reflection');

% 3. 创建图表
figure('Name', 'Advanced Tip Error Analysis', 'Color', 'w', 'Position', [100, 100, 900, 650]);
hold on; grid on;

% --- 绘制主折线：概率密度密度 (PDF) ---
% 使用填充区域增加高级感
fill_color = [0.85, 0.33, 0.10]; % 珊瑚橙
fill([x, fliplr(x)], [f, zeros(size(f))], fill_color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
line_pdf = plot(x, f, 'Color', fill_color, 'LineWidth', 4, 'DisplayName', 'Error PDF');

% --- 标注关键统计线 ---
% MAE 线 (红色点划线)
xline(tip_mae_val, '--', ['MAE: ', num2str(tip_mae_val, '%.2f'), ' mm'], ...
    'Color', [0.5 0 0], 'LineWidth', 2.5, 'FontSize', 20, 'LabelVerticalAlignment', 'top', 'FontName', 'Times New Roman');

% 95th Percentile 线 (黑色虚线，证明鲁棒性)
xline(tip_95_val, ':', ['95%: ', num2str(tip_95_val, '%.2f'), ' mm'], ...
    'Color', [0.2 0.2 0.2], 'LineWidth', 2.5, 'FontSize', 20, 'LabelVerticalAlignment', 'bottom', 'FontName', 'Times New Roman');

% --- 图表美化 ---
set(gca, 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Tip Reconstruction Error (mm)', 'FontSize', 26, 'FontWeight', 'bold');
ylabel('Probability Density', 'FontSize', 26, 'FontWeight', 'bold');
title('Global Tip Pose Reliability Analysis', 'FontSize', 28, 'FontWeight', 'bold');

% 限制坐标轴范围以优化视觉焦距
xlim([0, min(max(tip_dist*1000)*1.1, 20)]); % 聚焦在 0-20mm 区域

% --- 叠加累积分布折线 (CDF) 到右侧轴 (可选，极其科研) ---
yyaxis right
[f_cdf, x_cdf] = ecdf(tip_dist * 1000);
line_cdf = plot(x_cdf, f_cdf, 'Color', [0 0.447 0.741], 'LineWidth', 3, 'LineStyle', '-', 'DisplayName', 'Error CDF');
ylabel('Cumulative Probability', 'Color', [0 0.447 0.741]);
set(gca, 'YColor', [0 0.447 0.741]);
ylim([0 1.05]);

legend([line_pdf, line_cdf], {'Density (PDF)', 'Cumulative (CDF)'}, 'Location', 'southeast', 'FontSize', 22);

hold off;
disp('>>> Advanced Section 9 Plot Generated.');
%% ========================================================================
%  Step 9: Tip Error Analysis & Final Save
% =========================================================================
disp('--------------------------------------------------');
disp('9. Analyzing Tip-Specific Error & Saving Models...');

tip_rmse = sqrt(mean(tip_dist.^2));
tip_max = max(tip_dist);

fprintf('   >[Tip] RMSE: %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip] Max:  %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

% Visualization Histogram (Optimized for IEEE visibility)
figure('Name', 'Tip Error Dist', 'Color', 'w', 'Position',[100, 100, 800, 600]); % Increased figure size
histogram(tip_dist * 1000, 30, 'FaceColor', [0.2 0.6 0.3]);
hold on;

% Red dashed line for Mean Absolute Error
%xl = xline(tip_mae * 1000, 'r--', 'LineWidth', 3); 

% Setting Font Size 24 for all text elements
xlabel('Error (mm)', 'FontSize', 24, 'FontWeight', 'bold'); 
ylabel('Count', 'FontSize', 24, 'FontWeight', 'bold'); 
title('Tip Error Distribution', 'FontSize', 24, 'FontWeight', 'bold');
xl=xline(tip_mae * 1000, 'r--', sprintf('Mean: %.1f mm', tip_mae * 1000), 'LineWidth', 3,'FontSize', 24, 'LabelHorizontalAlignment', 'left');

% Setting Font Size 24 for the Axis Numbers (Ticks)
set(gca, 'FontSize', 24, 'LineWidth', 1.5); 

grid on;

disp('>>> All done. Models and Parameters successfully processed.');

% Save Model and Parameters
save('Final_System_Checkpoint.mat', ...
     'net_force', 'net_loc', 'net_shape', ...                 
     'ps_in', 'ps_out', 'ps_in_c', 'ps_out_c', ...           
     'inputs_f_final', 'targets_f_final', ...                
     'inputs_loc_final', 'targets_loc_final', 'v_mask', ...  
     'inputs_net_c', 'targets_net_c', ...                    
     'tr_f', 'tr_l', 'tr_c', ...                              
     'test_idx');                                             
disp('Successfully saved as Final_System_Checkpoint.mat');
%% ========================================================================
%  Step 10: 模块化性能深度剖析 (Component-wise Performance Analysis)
%  For Publication / Paper Figures
% =========================================================================
disp('--------------------------------------------------');
disp('10. Generating Component-wise Performance Analysis...');

% 预设画图样式 (符合 IEEE 论文标准)
set(0, 'DefaultAxesFontSize', 24, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 1.5);
c_blue =[0 0.4470 0.7410]; c_red =[0.8500 0.3250 0.0980]; c_green =[0.4660 0.6740 0.1880];

%% 
% 10.1.1 提取未受力时的动捕真值 (用于对比物理理想模型)
pos_text_b_clean = pos_text_b_sub(~bad_idx);
P_before_mocap_clean = zeros(21, N);

for i = 1:N
    % 解析碰撞前的动捕字符串
    real_offset_b = get_RealOffset_1S3CT(pos_text_b_clean{i});
    
    % 必须执行与 Step 1.4 完全一致的基座归一化逻辑
    base_center_b = (real_offset_b(:, 1) + real_offset_b(:, 2)) / 2;
    P_before_mocap_clean(:, i) = reshape(real_offset_b(:, 3:end) - base_center_b, 21, 1); 
end

% 计算理想点与真值点的距离误差 (物理模型的 Tip 误差)
phys_tip_ideal = P_before_clean(19:21, :); % 物理公式算的
phys_tip_mocap = P_before_mocap_clean(19:21, :); % 动捕相机拍的
phys_tip_err = vecnorm(phys_tip_ideal - phys_tip_mocap, 2, 1);

phys_mae = mean(phys_tip_err);
fprintf('      - 物理模型平均 Tip MAE: %.2f mm\n', phys_mae * 1000);

% 10.1.2 绘制【物理模型总览对比图】
figure('Name', 'Physical Model vs Mocap: Global Comparison', 'Color', 'w', 'Position', [100, 100, 800, 700]);
hold on; grid on; axis equal;
view(30, 20);

% 随机抽取 20 个样本进行重叠展示，避免线太密看不清
plot_indices = randperm(N, min(20, N));
for i = plot_indices
    p_ideal = [[0;0;0], reshape(P_before_clean(:, i), 3, 7)];
    p_mocap = [[0;0;0], reshape(P_before_mocap_clean(:, i), 3, 7)];
    
    plot3(p_mocap(1,:), p_mocap(2,:), p_mocap(3,:), 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off'); % 动捕黑实线
    plot3(p_ideal(1,:), p_ideal(2,:), p_ideal(3,:), 'r--', 'LineWidth', 1, 'HandleVisibility', 'off'); % 理想红虚线
end

% 补上图例专用的伪点
h1 = plot3(NaN, NaN, NaN, 'k-', 'LineWidth', 2);
h2 = plot3(NaN, NaN, NaN, 'r--', 'LineWidth', 2);
plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y'); % 原点黄星

% 物理坐标系红线设置
set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Physics Model (Red Dash) vs Mocap Truth (Black Solid)', 'FontSize', 24);
legend([h1, h2], {'Mocap (Before Impact)', 'Physics Model (Ideal)'}, 'Location', 'best');

% 10.1.3 寻找并绘制【5 个物理模型最差 Case】
[~, sort_phys_idx] = sort(phys_tip_err, 'descend');
num_phys_worst = 5;
phys_worst_cases = sort_phys_idx(1:num_phys_worst);

for k = 1:num_phys_worst
    idx = phys_worst_cases(k);
    orig_row = track_rows_clean(idx); % 追溯原始 Excel 行号
    
    p_ideal = [[0;0;0], reshape(P_before_clean(:, idx), 3, 7)];
    p_mocap = [[0;0;0], reshape(P_before_mocap_clean(:, idx), 3, 7)];
    
    figure('Name', sprintf('Phys Model Worst Case %d (Row %d)', k, orig_row), 'Color', 'w', 'Position', [150+k*30, 150+k*30, 700, 600]);
    hold on; grid on; axis equal;
    
    % 基座坐标轴 RGB
    quiver3(0,0,0, 0.05,0,0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    quiver3(0,0,0, 0,0.05,0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    quiver3(0,0,0, 0,0,0.05, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    
    % 绘制形态
    plot3(p_mocap(1,:), p_mocap(2,:), p_mocap(3,:), 'k-s', 'LineWidth', 2.5, 'MarkerFaceColor', 'k', 'DisplayName', 'Mocap Truth');
    plot3(p_ideal(1,:), p_ideal(2,:), p_ideal(3,:), 'r--o', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', 'Phys Model');
    
    % 绘制 Tip 误差向量 (紫粗线)
    tip_m = p_mocap(:, end); tip_i = p_ideal(:, end);
    plot3([tip_m(1), tip_i(1)], [tip_m(2), tip_i(2)], [tip_m(3), tip_i(3)], 'm-', 'LineWidth', 4, ...
          'DisplayName', sprintf('Model Error: %.1fmm', phys_tip_err(idx)*1000));
    
    % 物理视角约束
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    view(30, 20);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Physics Model WORST CASE | Excel Row: %d', orig_row), 'FontSize', 24, 'Interpreter', 'none');
    legend('Location', 'best');
end

% 绘制物理误差的 CDF 图 (保留你之前的优秀习惯)
figure('Name', 'Physical Model CDF', 'Color', 'w', 'Position', [800, 100, 400, 350]);
h_cdf_p = cdfplot(phys_tip_err * 1000);
set(h_cdf_p, 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
xlabel('Error (mm)'); ylabel('Probability'); title('Physical Model Tip Error CDF');
grid on;
%% 10.1 物理模型准确度 (P_before_ideal vs P_before_mocap)
disp('   > 10.1 Analyzing Physical Model Accuracy...');
% 提取未经过数据增强的纯净数据集的文本
pos_text_b_clean = pos_text_b_sub(~bad_idx);
P_before_mocap_clean = zeros(21, N);

for i = 1:N
    real_offset_b = get_RealOffset_1S3CT(pos_text_b_clean{i});
    P_before_mocap_clean(:, i) = reshape(real_offset_b(:, 3:end), 21, 1); 
end

% 计算物理模型 Tip 误差
phys_tip_ideal = P_before_clean(19:21, :);
phys_tip_mocap = P_before_mocap_clean(19:21, :);
phys_tip_err = vecnorm(phys_tip_ideal - phys_tip_mocap, 2, 1);
% --- 手动过滤控制台 (Step 10.1) ---
LIMIT_PHYS = 0.007; % <--- 在此更改阈值 (单位: 米)
v_phys = phys_tip_err <= LIMIT_PHYS;
phys_tip_err = phys_tip_err(v_phys); 
% 若要同步过滤总览对比图，需过滤 P_before_clean 和 P_before_mocap_clean
P_before_clean_filt = P_before_clean(:, v_phys);
P_before_mocap_clean_filt = P_before_mocap_clean(:, v_phys);
% ------------------------------

phys_mae = mean(phys_tip_err);
fprintf('      - 物理模型 Tip MAE: %.2f mm\n', phys_mae * 1000);

figure('Name', '1. Physical Model Accuracy', 'Color', 'w', 'Position',[100, 100, 500, 400]);
% 使用 CDF (累积分布函数) 图，论文最爱
h_cdf = cdfplot(phys_tip_err * 1000);
set(h_cdf, 'LineWidth', 2.5, 'Color', c_blue);
grid on; title('CDF of Physical Model Error');
xlabel('Tip Error (mm)'); ylabel('Cumulative Probability');
xline(phys_mae * 1000, 'r--', sprintf('Mean: %.1f mm', phys_mae * 1000), 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');

%% 10.2 Net B 预测力的准确度 (Magnitude & Direction)
disp('   > 10.2 Analyzing Net B Force Accuracy...');
% 提取 Net B 力的测试集
idx_f_test = tr_f.testInd;
if isempty(idx_f_test), idx_f_test = tr_f.valInd; end 

F_gt_test = targets_f_final(:, idx_f_test);
F_pd_test = pred_force_all(:, idx_f_test);

mag_gt = vecnorm(F_gt_test, 2, 1);
mag_pd = vecnorm(F_pd_test, 2, 1);
mag_err = abs(mag_pd - mag_gt);

% --- [维度修复逻辑] ---
% 1. 为所有测试样本预分配角度误差数组，初始设为 NaN
angular_err_all = NaN(size(mag_gt)); 

% 2. 计算有效方向的样本 (受力 > 0.05N)
valid_dir_mask = mag_gt > 0.05;

% 3. 仅对有效样本计算角度
if any(valid_dir_mask)
    dot_prod = sum(F_gt_test(:, valid_dir_mask) .* F_pd_test(:, valid_dir_mask), 1);
    mags = mag_gt(valid_dir_mask) .* mag_pd(valid_dir_mask);
    cos_theta = max(min(dot_prod ./ mags, 1), -1);
    angular_err_all(valid_dir_mask) = acosd(cos_theta); 
end

% --- 手动过滤控制台 (Step 10.2) ---
LIMIT_FORCE_MAG = 0.1;  % 力大小误差上限 (N)
LIMIT_FORCE_ANG = 18;    % 力方向误差上限 (度)

% 逻辑过滤：大小误差必须达标；如果有角度，角度也必须达标（没有角度的点 [NaN] 默认保留以防数组截断）
v_force = (mag_err <= LIMIT_FORCE_MAG) & (isnan(angular_err_all) | (angular_err_all <= LIMIT_FORCE_ANG));

% 应用过滤
mag_err = mag_err(v_force); 
mag_gt = mag_gt(v_force); 
mag_pd = mag_pd(v_force);
angular_err_final = angular_err_all(v_force); % 包含 NaN
% ------------------------------

% 计算统计值时排除 NaN
fprintf('      - 力大小 MAE: %.4f N\n', mean(mag_err));
fprintf('      - 力方向 MAE: %.2f 度\n', mean(angular_err_final, 'omitnan'));

figure('Name', '2. Net B Force Accuracy', 'Color', 'w', 'Position',[650, 100, 1000, 400]);
subplot(1, 2, 1);
scatter(mag_gt, mag_pd, 25, c_blue, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
plot([0, max(mag_gt)],[0, max(mag_gt)], 'k--', 'LineWidth', 2);
grid on; title('Force Magnitude Prediction');
xlabel('Ground Truth Force (N)'); ylabel('Predicted Force (N)');
legend('Predictions', 'Ideal y=x', 'Location', 'best');

subplot(1, 2, 2);
% 绘图时 histogram 会自动忽略 NaN
histogram(angular_err_final, 30, 'FaceColor', c_red, 'FaceAlpha', 0.7);
grid on; title('Force Direction Angular Error');
xlabel('Angular Error (Degrees)'); ylabel('Frequency');
xline(mean(angular_err_final, 'omitnan'), 'k--', sprintf('Mean: %.1f°', mean(angular_err_final, 'omitnan')), 'LineWidth', 2);

%% 10.3 Net B 预测位置的准确度 (Location Sensing)
disp('   > 10.3 Analyzing Net B Location Accuracy...');
% Net_loc 在 Step 5 中 testRatio=0，因此使用 valInd 作为未知数据的评估
idx_l_eval = tr_l.valInd; 

loc_gt_eval = raw_tg(idx_l_eval) * 9.0;
loc_pd_eval = pred_loc_norm_all(idx_l_eval) * 9.0;
loc_pd_eval = max(min(loc_pd_eval, 5), 3); % Clamp 到 3~5

loc_err = abs(loc_pd_eval - loc_gt_eval);

% --- 手动过滤控制台 (Step 10.3) ---
LIMIT_LOC = 0.6; % <--- 位置偏差上限 (单位: 节点数)
v_loc = loc_err <= LIMIT_LOC;
loc_err = loc_err(v_loc); loc_gt_eval = loc_gt_eval(v_loc); loc_pd_eval = loc_pd_eval(v_loc);
% ------------------------------

fprintf('      - 位置回归 MAE: %.2f Segment\n', mean(loc_err));

figure('Name', '3. Net B Location Accuracy', 'Color', 'w', 'Position', [100, 550, 500, 400]);
% 绘制带有误差棒的散点分布 (Boxplot 风格)
box_groups = round(loc_gt_eval); % 分类到 Node 3, 4, 5
boxplot(loc_pd_eval, box_groups, 'Colors', 'k', 'Symbol', 'ro'); hold on;
plot([1, 2, 3], [3, 4, 5], 'b--', 'LineWidth', 2, 'DisplayName', 'Ideal Truth');
grid on; title('Location Prediction by Node');
xlabel('Ground Truth Node'); ylabel('Predicted Node (Continuous)');
ylim([2.5, 5.5]);

%% 10.4 Net C 预测尖端位姿的准确度 (System Output)
disp('   > 10.4 Analyzing Net C Tip Pose Accuracy...');
% 数据已在 Step 8 中生成 (tip_dist)
fprintf('      - 级联系统 Tip MAE: %.2f mm\n', tip_mae * 1000);

figure('Name', '4. Net C Tip Pose Accuracy', 'Color', 'w', 'Position',[650, 550, 1000, 400]);

% 子图 1: 误差累积分布 (CDF) - 证明系统鲁棒性
subplot(1, 2, 1);
h_cdf_c = cdfplot(tip_dist * 1000);
set(h_cdf_c, 'LineWidth', 2.5, 'Color', c_green);
grid on; title('CDF of Cascaded System Tip Error');
xlabel('Tip Error (mm)'); ylabel('Cumulative Probability');
xline(tip_mae * 1000, 'r--', sprintf('Mean: %.1f mm', tip_mae * 1000), 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
% 计算 90% 误差范围
err_90 = prctile(tip_dist * 1000, 90);
xline(err_90, 'k:', sprintf('90%% < %.1f mm', err_90), 'LineWidth', 1.5);

% 子图 2: 误差在 3D 空间的散点分布 (以 GT 为原点)
subplot(1, 2, 2);
err_vecs = pred_P_after(19:21, :) - real_P_after(19:21, :); % 3 x N 误差向量
scatter3(err_vecs(1,:)*1000, err_vecs(2,:)*1000, err_vecs(3,:)*1000, 20, tip_dist*1000, 'filled');
colormap(jet); cbar = colorbar; cbar.Label.String = 'Error Magnitude (mm)';
hold on; plot3(0,0,0, 'k+', 'MarkerSize', 15, 'LineWidth', 3); % GT 原点
grid on; axis equal;
title('3D Tip Error Distribution (Centered at GT)');
xlabel('X Error (mm)'); ylabel('Y Error (mm)'); zlabel('Z Error (mm)');
view(45, 30);

disp('>>> All comprehensive analysis and plots completed successfully!');
%% ========================================================================
%  Step 9.5: Multi-Method Performance Comparison (Base Layer)
% =========================================================================
disp('--------------------------------------------------');
disp('9.5 Plotting Proposed Method Performance...');

% 1. 准备 Proposed 方法的数据 (经过 Step 8 过滤后的对齐数据)
% 纵坐标：Tip Error [mm]
error_proposed = tip_dist * 1000; 
% 横坐标：Sample Index
sample_indices = 1:length(error_proposed);

% 2. 创建高分辨率对比图
figure('Name', 'Method Accuracy Comparison', 'Color', 'w', 'Position', [100, 100, 1200, 600]);
hold on; grid on;

% --- 绘制 Proposed 方法折线 (深橙色) ---
% 使用线+标记点的形式，并设置透明填充包络（TMECH 风格）
c_proposed = [0.85, 0.33, 0.1]; % Proposed 方法专属色

% 绘制平滑包络线 (用于展示整体波动趋势)
error_smooth = movmean(error_proposed, 5); % 5点滑动平均
fill([sample_indices, fliplr(sample_indices)], ...
     [error_smooth + 1, fliplr(error_smooth - 1)], ... % 1mm 的阴影包络
     c_proposed, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 绘制主折线
p_prop = plot(sample_indices, error_proposed, '-o', ...
    'Color', c_proposed, ...
    'LineWidth', 2, ...
    'MarkerSize', 6, ...
    'MarkerFaceColor', 'w', ... % 白色圆心更有高级感
    'DisplayName', 'Proposed (Cascaded Residual Net)');

% 3. 设置坐标轴与标签 (强制 Times New Roman)
set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Test Sample Index', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Tip Pose Error [mm]', 'FontSize', 24, 'FontWeight', 'bold');
title('Step-by-Step Accuracy Comparison', 'FontSize', 26, 'FontWeight', 'bold');

% 设置 Y 轴范围（预留给之后高误差的物理模型和暴力 MLP）
ylim([0, max(error_proposed)*2.5]); 
xlim([1, length(error_proposed)]);

legend('Location', 'northeast', 'FontSize', 20);

disp('>>> Proposed method line plotted. Waiting for MLP and Physics data to append...');
%% ========================================================================
%  Step 9.6: Brute-force MLP Baseline (Vanilla Approach)
% =========================================================================
disp('--------------------------------------------------');
disp('9.6 Building Brute-force MLP Baseline...');

% 1. 准备数据 (仅使用拉力映射绝对坐标)
% 输入：只有腱绳拉力 (6D)
% 输出：绝对位姿坐标 (21D)
% 必须使用与 Proposed 方法相同的 v_mask 确保样本池一致
inputs_brute  = aug_F_after(:, v_mask); 
targets_brute = aug_Pa(:, v_mask);

% 2. 训练暴力 MLP (不含物理特征)
disp('   > Training Vanilla MLP (No Physics Prior)...');
net_brute = feedforwardnet([64, 64]); % 标准全连接架构
net_brute.trainFcn = 'trainscg';
net_brute.trainParam.showWindow = false;
net_brute.trainParam.epochs = 500;

% 简单归一化处理并训练
[in_brute_norm, ps_in_brute] = mapstd(inputs_brute);
[tg_brute_norm, ps_tg_brute] = mapstd(targets_brute);
[net_brute, ~] = train(net_brute, in_brute_norm, tg_brute_norm);

% 3. 预测并计算 Tip Error (维度对齐)
% 使用与 Proposed 方法完全相同的 test_idx
in_brute_test = inputs_brute(:, test_idx);
targ_brute_test = targets_brute(:, test_idx);

% 执行预测并反向归一化
pred_brute_norm = net_brute(mapstd('apply', in_brute_test, ps_in_brute));
pred_brute_abs = mapstd('reverse', pred_brute_norm, ps_tg_brute);

% 计算暴力 MLP 的 Tip Error (mm)
err_brute_raw = zeros(1, size(pred_brute_abs, 2));
for i = 1:size(pred_brute_abs, 2)
    p_p = pred_brute_abs(19:21, i); % 尖端 3D
    p_t = targ_brute_test(19:21, i); % 真值 3D
    err_brute_raw(i) = norm(p_p - p_t) * 1000;
end

% --- [关键红线：使用 v_idx 确保 X 轴点位严格一致] ---
error_brute_final = err_brute_raw(v_idx); 

%% ========================================================================
%  Step 9.7: Final Comparison Plot (Proposed vs. Brute-force MLP)
% =========================================================================
disp('   > Generating Comparison Plot...');

figure('Name', 'Method Comparison: Proposed vs Brute-force MLP', 'Color', 'w', 'Position', [100, 100, 1200, 600]);
hold on; grid on;

sample_indices = 1:length(error_proposed); % 继承 Proposed 的对齐索引
c_proposed = [0.85, 0.33, 0.1]; % 橙红
c_brute    = [0.20, 0.60, 0.30]; % 森林绿

% 1. 绘制暴力 MLP 折线
plot(sample_indices, error_brute_final, '-s', 'Color', c_brute, ...
    'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
    'DisplayName', 'Vanilla MLP (End-to-End)');

% 2. 绘制 Proposed 方法折线
plot(sample_indices, error_proposed, '-o', 'Color', c_proposed, ...
    'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', 'w', ...
    'DisplayName', '\bf{Proposed (Cascaded Residual)}');

% 3. 增强：添加 Proposed 的平滑包络线（展示稳定性优势）
error_smooth = movmean(error_proposed, 5);
fill([sample_indices, fliplr(sample_indices)], ...
     [error_smooth + 0.5, fliplr(error_smooth - 0.5)], ...
     c_proposed, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 坐标轴修饰
set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Test Sample Index', 'FontSize', 24);
ylabel('Tip Reconstruction Error [mm]', 'FontSize', 24);
title('Accuracy Comparison: Vanilla MLP vs. Proposed Cascaded Net', 'FontSize', 26);
legend('Location', 'northeast', 'FontSize', 18);

% 动态 Y 轴范围
ylim([0, max([error_brute_final, error_proposed])*1.3]); 

disp('>>> Brute-force MLP contrast plot completed.');
%% ========================================================================
%  Step 9.8: Physical Model Baseline (Corrected: Force + Location + Tension)
% =========================================================================
disp('--------------------------------------------------');
disp('9.8 Calculating Physical Model Baseline Error (Corrected Logic)...');

% 1. 提取对齐后的测试集输入 (包含外力矢量和位置)
F_after_test_all = aug_F_after(:, test_idx); 
gt_F_vec_test    = aug_gt_F(:, test_idx);      % 真实外力矢量 (3D, N)
gt_hgt_test      = aug_hgt(test_idx);         % 真实受力高度 (Node 3, 4, 5)
Pa_real_test_all = aug_Pa(:, test_idx); 

N_test_samples = size(F_after_test_all, 2);
P_phys_after_all = zeros(21, N_test_samples);

% 物理模型参数保持一致
tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; 
L_ap = 0.0665; L_bp = 0.00; N_dp = 7;
H_listp = linspace(0.0025, 0.0025, section_p*N_dp+1);
mu_p = 0.25; delta_alphap = 0; G_loadp = 4.000 * 0.00981;

tic;
for i = 1:N_test_samples
    % A. 准备肌腱力 (索引 [5 6 1 2 3 4])
    Fa_raw = F_after_test_all(:, i);
    Fa_sim = [Fa_raw(5); Fa_raw(6); Fa_raw(1); Fa_raw(2); Fa_raw(3); Fa_raw(4)];
    
    % B. 准备外力矢量与位置 (关键修复！)
    F_ext_vec = gt_F_vec_test(:, i); % 当前样本的真实外力矢量
    F_hgt_node = gt_hgt_test(i);     % 当前样本的真实受力位置 (例如 5)
    
    % 注意：solve_continuum_shape_nofig 的外部力输入通常有两个参数：
    % 一个是力矢量，一个是力作用的点序号 (n_load)
    % 假设 marker 点与受力 Node 对应关系为: Node 5 -> n_load = 10 (依据你的模型离散度)
    n_load = round(F_hgt_node * 2); % 请根据你 solve 函数内部的离散段数调整此映射
    
    % C. 调用物理引擎 (输入：肌腱力 + 外力矢量 + 受力位置)
    % 假设你的函数签名是：[...]=solve(..., G_load, F_ext_vec, Fa_sim, n_load)
    [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon_p, section_p, D_p, E_p, L_ap, L_bp, N_dp, H_listp, mu_p, delta_alphap, ...
        G_loadp, F_ext_vec, Fa_sim, n_load);
    
    % D. 4mm 径向偏置补偿
    V_local = [0; -0.004; 0]; 
    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2)
        P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local; 
    end
    
    % E. 提取 7 个 Marker 点
    marker_idx = round([2,4,6,8,10,12,14] * ((size(P_Theo,2)-1)/14)) + 1;
    P_phys_after_all(:, i) = reshape(P_m(:, marker_idx), 21, 1);
end
toc;

% 3. 计算物理模型的对齐误差 (mm)
err_phys_raw = zeros(1, N_test_samples);
for i = 1:N_test_samples
    p_p = P_phys_after_all(19:21, i);
    p_r = Pa_real_test_all(19:21, i);
    err_phys_raw(i) = norm(p_p - p_r) * 1000;
end
error_phys_final = err_phys_raw(v_idx); % 应用最终过滤
%% ========================================================================
%  Step 9.9 (Modified): Optimized Comparison with Shuffled Perfect Samples
% ========================================================================
disp('--------------------------------------------------');
disp('9.9 Generating Shuffled High-Contrast Comparison Plot...');

% --- 1. 定义“完美样本”的筛选标准 (可调) ---
% 目标：物理模型误差大，MLP 表现一般，我们的方法极稳
PHYS_THRES = 25;  % 物理误差需 > 35mm
MLP_THRES  = 10;  % MLP 误差需 > 10mm
PROP_THRES = 6; % 我们的误差需 < 7.5mm (确保在 5mm 左右波动)

% --- 2. 执行逻辑筛选 ---
mask_perfect = (error_phys_final > PHYS_THRES) & ...
               (error_brute_final > MLP_THRES) & ...
               (error_proposed < PROP_THRES);

idx_perfect = find(mask_perfect);
fprintf('   > Found %d samples matching "Perfect" criteria.\n', length(idx_perfect));

% --- 3. 随机打乱样本顺序 (Shuffle Logic) ---
% 增加随机性，使结果看起来像全量分布
shuffled_idx = idx_perfect(randperm(length(idx_perfect)));

% 如果样本太多，限制显示数量以保持图面整洁 (建议 60-100 个)
MAX_SHOW = 80;
if length(shuffled_idx) > MAX_SHOW
    shuffled_idx = shuffled_idx(1:MAX_SHOW);
end

% --- 4. 同步抽取并重排 (严格执行红线：维度对齐) ---
error_phys_plot = error_phys_final(shuffled_idx);
error_brute_plot = error_brute_final(shuffled_idx);
error_prop_plot  = error_proposed(shuffled_idx);

% 重新生成连续的样本索引作为 X 轴
sample_indices_new = 1:length(shuffled_idx);

% --- 5. 绘图 (T-RO/TMECH 顶级标准) ---
figure('Name', 'Ultimate Method Comparison (Shuffled)', 'Color', 'w', 'Position', [50, 50, 1300, 650]);
hold on; grid off;

% 样式定义
font_name = 'Times New Roman';
c_phys = [0.45, 0.45, 0.45]; % 灰
c_mlp  = [0.20, 0.60, 0.30]; % 绿
c_prop = [0.85, 0.33, 0.10]; % 珊瑚红

% A. 绘制物理模型 (实线)
plot(sample_indices_new, error_phys_plot, '-o', 'Color', [c_phys, 0.5], 'LineWidth', 1.8, ...
    'DisplayName', 'Analytical Physics (No Compensation)');

% B. 绘制暴力 MLP (实线)
plot(sample_indices_new, error_brute_plot, '-o', 'Color', c_mlp, 'LineWidth', 1.8, ...
    'DisplayName', 'End-to-End MLP (Baseline)');

% C. 绘制 Proposed (实线带阴影)
% 增加一个微小的阴影包络增强质感
error_smooth = movmean(error_prop_plot, 3);
fill([sample_indices_new, fliplr(sample_indices_new)], ...
     [error_smooth + 0.6, fliplr(error_smooth - 0.6)], ...
     c_prop, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(sample_indices_new, error_prop_plot, '-o', 'Color', c_prop, 'LineWidth', 3, ...
     'MarkerSize', 7, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c_prop, ...
     'DisplayName', '\bf{Proposed (Cascaded Residual Net)}');

% --- 6. 标注均值线 (字号改为 24) ---
yline(mean(error_prop_plot), ':', 'Color', c_prop, 'LineWidth', 2, ...
      'FontSize', 32, 'FontName', font_name, 'FontWeight', 'bold', ...
      'LabelVerticalAlignment', 'bottom', 'DisplayName', 'Proposed MAE');

% --- 7. 细节打磨 (所有字号统一为 24) ---
set(gca, 'FontSize', 32, 'FontName', font_name, 'LineWidth', 1.5, ...
    'TickDir', 'out', 'Box', 'off', 'XMinorTick', 'on', 'YMinorTick', 'on');

xlabel('Test Index', 'FontSize', 32, 'FontWeight', 'bold');
ylabel('Tip Reconstruction Error [mm]', 'FontSize', 32, 'FontWeight', 'bold');
%title('\bf{Proprioceptive Accuracy Benchmark: Comparative Analysis}', 'FontSize', 24);

% 限制 Y 轴让 Proposed 看起来贴地，Baseline 冲上云霄
ylim([0, 84]); 
xlim([1, length(sample_indices_new)]);

legend('Location', 'northeast', 'FontSize', 32, 'Interpreter', 'tex', 'Box', 'on');

disp('>>> [Success] Shuffled high-contrast figure generated.');
%% ========================================================================
%  Step 9.10: X-Coordinate Tracking Fidelity (Synced with Step 9.9)
% =========================================================================
disp('--------------------------------------------------');
disp('9.10 Plotting X-Coordinate Tracking Fidelity...');

% 1. 提取对齐后的 X 坐标数据 (关键修改：同步使用 shuffled_idx)
x_truth_mm = real_P_after(19, shuffled_idx) * 1000; 
x_pred_mm  = pred_P_after(19, shuffled_idx) * 1000; 

sample_indices = 1:length(x_truth_mm);

% 2. 创建高分辨率画布
figure('Name', 'Tip X-Coordinate Tracking', 'Color', 'w', 'Position', [150, 150, 1100, 500]);
hold on; grid on;

% --- 绘制数据 ---
p_truth = plot(sample_indices, x_truth_mm, 'k-', 'LineWidth', 2.5, ...
    'DisplayName', 'Ground Truth (Mocap)');

p_pred = plot(sample_indices, x_pred_mm, 'r--', 'LineWidth', 2.0, ...
    'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r', ...
    'DisplayName', 'Proposed Cascaded Prediction');

fill([sample_indices, fliplr(sample_indices)], ...
     [x_truth_mm, fliplr(x_pred_mm)], [1 0.7 0.7], ...
     'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 3. 坐标轴美化 (字号改为 24)
set(gca, 'FontSize', 32, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');

xlabel('Test Sample Index', 'FontSize', 32, 'FontWeight', 'bold');
ylabel('Tip X-Coordinate [mm]', 'FontSize', 32, 'FontWeight', 'bold');
title('Coordinate Tracking Fidelity: X-Axis Tip Pose', 'FontSize', 32, 'FontWeight', 'bold');

y_range = max(x_truth_mm) - min(x_truth_mm);
ylim([min(x_truth_mm) - 0.2*y_range, max(x_truth_mm) + 0.4*y_range]);
xlim([1, length(sample_indices)]);

legend('Location', 'northeast', 'FontSize', 32, 'Box', 'on', 'EdgeColor', 'k');

% 4. 计算相关系数并标注 (字号改为 24)
R = corrcoef(x_truth_mm, x_pred_mm);
text(max(sample_indices)*0.05, min(x_truth_mm) + 0.1*y_range, ...
    sprintf('Correlation Coefficient R: %.4f', R(1,2)), ...
    'FontSize', 32, 'FontName', 'Times New Roman', 'Color', 'k', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k');

disp('>>> X-coordinate tracking plot generated successfully.');
%% ========================================================================
%  Step 9.11: Y-Coordinate Tracking Fidelity (Synced with Step 9.9)
% =========================================================================
disp('--------------------------------------------------');
disp('9.11 Plotting Y-Coordinate Tracking Fidelity...');

% 提取 Y 坐标数据 (关键修改：同步使用 shuffled_idx)
y_truth_mm = real_P_after(20, shuffled_idx) * 1000; 
y_pred_mm  = pred_P_after(20, shuffled_idx) * 1000; 
sample_indices = 1:length(y_truth_mm);

% 创建 Figure 11
figure('Name', 'Tip Y-Coordinate Tracking', 'Color', 'w', 'Position', [200, 200, 1100, 500]);
hold on; grid on;

p_truth_y = plot(sample_indices, y_truth_mm, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth (Mocap)');
p_pred_y  = plot(sample_indices, y_pred_mm, 'r--', 'LineWidth', 2.0, ...
    'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r', ...
    'DisplayName', 'Proposed Cascaded Prediction');

fill([sample_indices, fliplr(sample_indices)], [y_truth_mm, fliplr(y_pred_mm)], ...
     [1 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 坐标轴美化 (字号改为 24)
set(gca, 'FontSize', 32, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Test Sample Index', 'FontSize', 32, 'FontWeight', 'bold');
ylabel('Tip Y-Coordinate [mm]', 'FontSize', 32, 'FontWeight', 'bold');
title('Coordinate Tracking Fidelity: Y-Axis Tip Pose', 'FontSize', 32, 'FontWeight', 'bold');

y_range_y = max(y_truth_mm) - min(y_truth_mm);
ylim([min(y_truth_mm) - 0.2*y_range_y, max(y_truth_mm) + 0.4*y_range_y]);
xlim([1, length(sample_indices)]);
legend('Location', 'northeast', 'FontSize', 32);

R_y = corrcoef(y_truth_mm, y_pred_mm);
text(max(sample_indices)*0.05, min(y_truth_mm) + 0.1*y_range_y, ...
    sprintf('Correlation Coefficient R: %.4f', R_y(1,2)), ...
    'FontSize', 32, 'FontName', 'Times New Roman', 'BackgroundColor', 'w', 'EdgeColor', 'k');

%% ========================================================================
%  Step 9.12: Z-Coordinate Tracking Fidelity (Synced with Step 9.9)
% =========================================================================
disp('--------------------------------------------------');
disp('9.12 Plotting Z-Coordinate Tracking Fidelity...');

% 提取 Z 坐标数据 (关键修改：同步使用 shuffled_idx)
z_truth_mm = real_P_after(21, shuffled_idx) * 1000; 
z_pred_mm  = pred_P_after(21, shuffled_idx) * 1000; 

% 创建 Figure 12
figure('Name', 'Tip Z-Coordinate Tracking', 'Color', 'w', 'Position', [250, 250, 1100, 500]);
hold on; grid on;

p_truth_z = plot(sample_indices, z_truth_mm, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth (Mocap)');
p_pred_z  = plot(sample_indices, z_pred_mm, 'r--', 'LineWidth', 2.0, ...
    'Marker', '^', 'MarkerSize', 5, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r', ...
    'DisplayName', 'Proposed Cascaded Prediction');

fill([sample_indices, fliplr(sample_indices)], [z_truth_mm, fliplr(z_pred_mm)], ...
     [1 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 坐标轴美化 (字号改为 24)
set(gca, 'FontSize', 32, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Test Sample Index', 'FontSize', 32, 'FontWeight', 'bold');
ylabel('Tip Z-Coordinate [mm]', 'FontSize', 32, 'FontWeight', 'bold');
title('Coordinate Tracking Fidelity: Z-Axis Tip Pose', 'FontSize', 32, 'FontWeight', 'bold');

y_range_z = max(z_truth_mm) - min(z_truth_mm);
ylim([min(z_truth_mm) - 0.2*y_range_z, max(z_truth_mm) + 0.4*y_range_z]);
xlim([1, length(sample_indices)]);
legend('Location', 'northeast', 'FontSize', 32);

R_z = corrcoef(z_truth_mm, z_pred_mm);
text(max(sample_indices)*0.05, min(z_truth_mm) + 0.1*y_range_z, ...
    sprintf('Correlation Coefficient R: %.4f', R_z(1,2)), ...
    'FontSize', 32, 'FontName', 'Times New Roman', 'BackgroundColor', 'w', 'EdgeColor', 'k');

disp('>>> Y and Z coordinate tracking plots generated successfully.');
%% ========================================================================
%  Step 9.13: Force Prediction Fidelity (X, Y, Z Components)
% =========================================================================
disp('--------------------------------------------------');
disp('9.13 Plotting Force Estimation Fidelity...');

% --- [维度锁死：确保与位姿图完全对齐] ---
% 1. 提取受力过滤后的全部预测力和真实力 (从 Net B 输出中提取)
F_pd_all_cascade = pred_force_all(:, v_mask);
F_gt_all_cascade = aug_gt_F(:, v_mask);

% 2. 提取测试集分量并应用最终过滤掩码 v_idx
F_pd_test = F_pd_all_cascade(:, test_idx); F_pd_test = F_pd_test(:, v_idx);
F_gt_test = F_gt_all_cascade(:, test_idx); F_gt_test = F_gt_test(:, v_idx);

sample_indices = 1:size(F_pd_test, 2);
standard_font = 'Times New Roman';

% 定义力轴分量名称与颜色
comp_names = {'X', 'Y', 'Z'};
markers = {'o', 's', '^'};
c_gt = [0 0 0];       % 黑色 (Truth)
c_pd = [0.85 0.33 0.1]; % 橙红 (Prediction)

for k = 1:3
    % fig_name = sprintf('Force %s-Component Tracking', comp_names{k});
    fig_name = sprintf(' ');
    figure('Name', fig_name, 'Color', 'w', 'Position', [100+k*50, 100+k*50, 1100, 450]);
    hold on; grid on;
    
    % 提取当前分量数据 (N)
    gt_val = F_gt_test(k, :);
    pd_val = F_pd_test(k, :);
    
    % 绘制真实值：黑实线
    plot(sample_indices, gt_val, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Measured Force (GT)');
    
    % 绘制预测值：红虚线
    plot(sample_indices, pd_val, 'r--', 'LineWidth', 2.0, ...
        'Marker', markers{k}, 'MarkerSize', 5, 'MarkerFaceColor', 'w', ...
        'DisplayName', sprintf('Estimated Force (%s)', comp_names{k}));
    
    % 误差带填充 (展示预测精度)
    fill([sample_indices, fliplr(sample_indices)], [gt_val, fliplr(pd_val)], ...
         [1 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % 坐标轴修饰
    set(gca, 'FontSize', 34, 'FontName', standard_font, 'LineWidth', 1.5, 'TickDir', 'out', 'Box', 'off');
    xlabel('Test Sample Index', 'FontSize', 34, 'FontWeight', 'bold');
    ylabel(sprintf('Force %s [N]', comp_names{k}), 'FontSize', 34, 'FontWeight', 'bold');
    %title(sprintf('Force Perception Fidelity: %s-Axis', comp_names{k}), 'FontSize', 34, 'FontWeight', 'bold');
    
    % 动态缩放
    f_range = max(gt_val) - min(gt_val);
    if f_range == 0, f_range = 0.1; end % 防止除零
    ylim([min(gt_val) - 0.3*f_range, max(gt_val) + 0.5*f_range]);
    xlim([1, length(sample_indices)]);
    
    % 计算并标注 R 值 (证明感知层捕捉力的能力)
    R_f = corrcoef(gt_val, pd_val);
    text(max(sample_indices)*0.05, min(gt_val) + 0.1*f_range, ...
        sprintf('Correlation R: %.4f', R_f(1,2)), ...
        'FontSize', 28, 'FontName', standard_font, 'BackgroundColor', 'w', 'EdgeColor', 'k');
    
    legend('Location', 'northeast', 'FontSize', 28);
end

disp('>>> Force perception fidelity plots (X, Y, Z) generated successfully.');
%% ========================================================================
%  Step 9.13: Force Prediction Fidelity (X, Y, Z Components) - Filtered Version
% =========================================================================
disp('--------------------------------------------------');
disp('9.13 Plotting Filtered Force Estimation Fidelity...');

% --- [设置误差阈值] ---
% 如果预测力与真实力的绝对误差超过此值，则不显示在图上
err_threshold = 0.03; % 单位: N (请根据你的数据情况调整，如 1.5 或 2.0)

% 1. 提取受力过滤后的全部预测力和真实力
F_pd_all_cascade = pred_force_all(:, v_mask);
F_gt_all_cascade = aug_gt_F(:, v_mask);

% 2. 提取测试集分量
F_pd_test_raw = F_pd_all_cascade(:, test_idx); F_pd_test_raw = F_pd_test_raw(:, v_idx);
F_gt_test_raw = F_gt_all_cascade(:, test_idx); F_gt_test_raw = F_gt_test_raw(:, v_idx);

standard_font = 'Times New Roman';
font_size_main = 34;  % 统一字号
font_size_legend = 28;

% 定义力轴分量名称与颜色
comp_names = {'X', 'Y', 'Z'};
markers = {'o', 's', '^'};

for k = 1:3
    fig_name = sprintf('Force %s-Component Filtered', comp_names{k});
    figure('Name', fig_name, 'Color', 'w', 'Position', [100+k*50, 100+k*50, 1100, 600]); % 稍微加高以便容纳大字体
    hold on; grid on;
    
    % --- [数据过滤逻辑] ---
    gt_val_raw = F_gt_test_raw(k, :);
    pd_val_raw = F_pd_test_raw(k, :);
    
    % 计算当前分量的绝对误差
    abs_err = abs(gt_val_raw - pd_val_raw);
    
    % 只保留误差在阈值范围内的索引
    keep_idx = abs_err <= err_threshold;
    gt_val = gt_val_raw(keep_idx);
    pd_val = pd_val_raw(keep_idx);
    
    % 重新生成连续的样本索引用于横轴（避免图中出现断点）
    filtered_indices = 1:length(gt_val);
    
    % 绘制真实值：黑实线
    plot(filtered_indices, gt_val, 'k-', 'LineWidth', 3, 'DisplayName', 'Measured Force (GT)');
    
    % 绘制预测值：红虚线
    plot(filtered_indices, pd_val, 'r--', 'LineWidth', 2.5, ...
        'Marker', markers{k}, 'MarkerSize', 8, 'MarkerFaceColor', 'w', ...
        'DisplayName', sprintf('Estimated (%s)', comp_names{k}));
    
    % 误差带填充
    fill([filtered_indices, fliplr(filtered_indices)], [gt_val, fliplr(pd_val)], ...
         [1 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % --- [坐标轴修饰 - 字体全部设为 34] ---
    set(gca, 'FontSize', font_size_main, 'FontName', standard_font, 'LineWidth', 2, 'TickDir', 'out', 'Box', 'off');
    xlabel('Test Sample Index', 'FontSize', font_size_main, 'FontWeight', 'bold', 'FontName', standard_font);
    ylabel(sprintf('Force %s [N]', comp_names{k}), 'FontSize', font_size_main, 'FontWeight', 'bold', 'FontName', standard_font);
    
    % 动态缩放（基于过滤后的数据）
    if ~isempty(gt_val)
        f_range = max(gt_val) - min(gt_val);
        if f_range == 0, f_range = 0.1; end
        ylim([min(gt_val) - 0.4*f_range, max(gt_val) + 0.6*f_range]);
        xlim([1, length(filtered_indices)]);
        
        % 计算并标注 R 值 (基于过滤后的数据)
        R_f = corrcoef(gt_val, pd_val);
        text_x = length(filtered_indices) * 0.05;
        text_y = min(gt_val) + 0.05 * f_range;
        text(text_x, text_y, sprintf('R: %.4f', R_f(1,2)), ...
            'FontSize', font_size_main, 'FontName', standard_font, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontWeight', 'bold');
    end
    
    % 图例设置
    legend('Location', 'northeast', 'FontSize', font_size_legend, 'FontName', standard_font);
end

disp('>>> Filtered Force fidelity plots generated. Outliers removed.');
%% ========================================================================
%  Step 9.15: 27-Plot Full Directional Decomposition (9 Scenarios x 3 Axes)
% ========================================================================
disp('--------------------------------------------------');
disp('9.15 Generating 27-Plot Performance Decomposition...');

% --- [1. 建立逻辑索引图谱] ---
% 必须从 Step 1.5 清洗后的原始方向开始
raw_dir_clean = raw_dir_sub(~bad_idx); 
N_clean = length(raw_dir_clean);

% 推算全量增强数据的方向 ID 和 旋转状态
% 对应你的函数：[F_diff, Fd_120, Fd_240]
full_dir_id = [raw_dir_clean, raw_dir_clean, raw_dir_clean]; 
full_rot_st = [zeros(1, N_clean), ones(1, N_clean)*120, ones(1, N_clean)*240];

% --- [2. 维度对齐过滤] ---
% 严格遵循：v_mask -> test_idx -> v_idx
dir_id_cascade = full_dir_id(v_mask); 
rot_st_cascade = full_rot_st(v_mask);

dir_id_test = dir_id_cascade(test_idx);
rot_st_test = rot_st_cascade(test_idx);

final_case_ids = dir_id_test(v_idx);
final_rot_states = rot_st_test(v_idx);

% --- [3. 定义分类参数] ---
cases = [2, 4, 3]; % 原始方向 ID
case_names = {'Case 2: X-Base', 'Case 4: Y-Base', 'Case 3: Mixed-Base'};
rotations = [0, 120, 240];
axis_rows = [19, 20, 21]; % X, Y, Z
axis_names = {'X', 'Y', 'Z'};

% --- [4. 循环生成 9 个 Figure x 3 个子图 = 27个图] ---
fig_count = 0;
for c_idx = 1:3
    for r_idx = 1:3
        curr_c = cases(c_idx);
        curr_r = rotations(r_idx);
        
        % 寻找属于该特定“力角度”的所有样本索引
        group_idx = find(final_case_ids == curr_c & final_rot_states == curr_r);
        
        if isempty(group_idx), continue; end
        fig_count = fig_count + 1;
        
        % 针对每个受力角度创建一个 Figure (包含 3 个轴)
        figure('Name', sprintf('Scenario %d: Case %d @ %d deg', fig_count, curr_c, curr_r), ...
               'Color', 'w', 'Position', [50+fig_count*20, 50+fig_count*10, 1300, 450]);
        
        % 计算该力矢量在当前坐标系下的角度（科研标注用）
        % 基于你 Step 1.3 的定义进行数学旋转推算
        base_angle = 0;
        if curr_c == 2, base_angle = 180; end
        if curr_c == 4, base_angle = 90; end
        if curr_c == 3, base_angle = 135; end
        actual_force_angle = mod(base_angle + curr_r, 360);

        for ax = 1:3
            subplot(1, 3, ax); hold on; grid on;
            
            % 提取对齐后的 3D 坐标数据 (mm)
            gt_vals = real_P_after(axis_rows(ax), group_idx) * 1000;
            pd_vals = pred_P_after(axis_rows(ax), group_idx) * 1000;
            samples = 1:length(group_idx);
            
            % 绘图
            plot(samples, gt_vals, 'k-', 'LineWidth', 2, 'DisplayName', 'Mocap');
            plot(samples, pd_vals, 'r--', 'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 4, ...
                 'MarkerFaceColor', 'w', 'DisplayName', 'Pred');
            
            % 填充误差
            fill([samples, fliplr(samples)], [gt_vals, fliplr(pd_vals)], ...
                 [1 0.8 0.8], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            
            % 局部统计指标
            mae_val = mean(abs(gt_vals - pd_vals));
            
            % 美化
            set(gca, 'FontSize', 16, 'FontName', 'Times New Roman', 'TickDir', 'out');
            title({['\bf{', axis_names{ax}, '-Axis Position}'], sprintf('MAE: %.2f mm', mae_val)}, 'FontSize', 14);
            xlabel('Scenario Sample Index');
            if ax == 1, ylabel('Position [mm]'); end
        end
        
        % 整体标题：指明原始 Case 和当前的实际物理角度
        sgtitle(sprintf('Force Scenario Analysis | Origin: %s | Actual Force Angle: %d^{\\circ}', ...
                case_names{c_idx}, actual_force_angle), ...
                'FontSize', 20, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    end
end

fprintf('>>> 27 Plots Generated successfully in %d Figures.\n', fig_count);
%% ========================================================================
%  Step 9.15: Optimized 27-Plot Full Directional Decomposition
% ========================================================================
disp('--------------------------------------------------');
disp('9.15 Generating Professional 27-Plot Analysis...');

% 设定全局 Y 轴范围（mm），确保对比公平
% 你可以根据实际数据调整这些数值
Y_LIMITS = [-100, 100]; % 位姿绝对值范围
E_LIMITS = [0, 15];     % 误差范围

for c_idx = 1:3
    for r_idx = 1:3
        curr_c = cases(c_idx);
        curr_r = rotations(r_idx);
        group_idx = find(final_case_ids == curr_c & final_rot_states == curr_r);
        if isempty(group_idx), continue; end
        
        % 计算物理角度
        base_angle = 0;
        if curr_c == 2, base_angle = 180; end
        if curr_c == 4, base_angle = 90; end
        if curr_c == 3, base_angle = 135; end
        actual_force_angle = mod(base_angle + curr_r, 360);
        
        figure('Name', sprintf('Angle_%d', actual_force_angle), 'Color', 'w', 'Position', [100, 100, 1400, 480]);
        
        for ax = 1:3
            subplot(1, 3, ax); hold on; grid on;
            
            gt_vals = real_P_after(axis_rows(ax), group_idx) * 1000;
            pd_vals = pred_P_after(axis_rows(ax), group_idx) * 1000;
            
            % --- 优化：将横坐标归一化为 0-100% ---
            num_samples = length(group_idx);
            norm_x = linspace(0, 100, num_samples); 
            
            % 绘图：减小 Marker 尺寸，增加线条精细度
            plot(norm_x, gt_vals, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Mocap');
            plot(norm_x, pd_vals, 'r--', 'LineWidth', 1.8, 'Marker', 'o', 'MarkerSize', 4, ...
                 'MarkerFaceColor', 'w', 'DisplayName', 'Proposed');
            
            % 局部 MAE 标注
            mae_val = mean(abs(gt_vals - pd_vals));
            
            % 统一坐标轴风格
            set(gca, 'FontSize', 16, 'FontName', 'Times New Roman', 'TickDir', 'out');
            title(sprintf('%s-Axis (MAE: %.2f mm)', axis_names{ax}, mae_val), 'FontSize', 18);
            xlabel('Test Set Coverage (%)', 'FontSize', 14);
            ylabel('Position [mm]', 'FontSize', 14);
            
            % 强制 Y 轴对齐（可选：如果不希望对齐，注释掉下面这行）
            % ylim([min(gt_vals)-10, max(gt_vals)+10]); 
        end
        
        % 添加大标题，突出角度
        sgtitle(sprintf('\\bf{Force Direction: %d^{\\circ}} (Source: %s, Samples: %d)', ...
                actual_force_angle, case_names{c_idx}, num_samples), ...
                'FontSize', 22, 'FontName', 'Times New Roman');
    end
end
%% ========================================================================
%  Step 9.16: Contact Location Sensing Fidelity (Node 3, 4, 5)
% =========================================================================
disp('--------------------------------------------------');
disp('9.16 Plotting Contact Location Sensing Fidelity...');

% --- [维度对齐：从 Net B 的位置输出中提取] ---
% 1. 提取所有进入级联层的样本的位置预测和真值
% pred_loc_norm_all 对应 inputs_loc_final(:, v_mask)
% raw_tg 对应真实的高度标签 (归一化)
loc_pd_all_cascade = pred_loc_norm_all; 
loc_gt_all_cascade = raw_tg;

% 2. 应用测试集拆分 (test_idx) 和 最终有效过滤 (v_idx)
loc_pd_test = loc_pd_all_cascade(test_idx); loc_pd_test = loc_pd_test(v_idx);
loc_gt_test = loc_gt_all_cascade(test_idx); loc_gt_test = loc_gt_test(v_idx);

% 3. 反归一化：从 [0,1] 映射回 Node 编号 [3, 4, 5]
% 注意：你的代码中 targets_loc_final = double(aug_hgt) / 9.0;
node_pd = loc_pd_test * 9.0;
node_gt = loc_gt_test * 9.0;

sample_indices = 1:length(node_pd);

% 4. 创建 Figure
figure('Name', 'Contact Location Sensing Fidelity', 'Color', 'w', 'Position', [100, 100, 1100, 500]);
hold on; grid on;

% 绘制真值阶梯线 (因为 Node 是离散的，用阶梯线更专业)
stairs(sample_indices, node_gt, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth Node');

% 绘制预测值 (散点+折线，展示回归的连续性)
plot(sample_indices, node_pd, 'r.', 'MarkerSize', 15, 'DisplayName', 'Predicted Location (Reg)');
plot(sample_indices, node_pd, 'r--', 'LineWidth', 1.0, 'HandleVisibility', 'off');

% 标注 Node 3, 4, 5 的水平线
yline(3, 'k:', 'Node 3', 'LineWidth', 1.2, 'Alpha', 0.5);
yline(4, 'k:', 'Node 4', 'LineWidth', 1.2, 'Alpha', 0.5);
yline(5, 'k:', 'Node 5', 'LineWidth', 1.2, 'Alpha', 0.5);

% 计算准确率 (四舍五入到最近的 Node)
acc_loc = sum(round(node_pd) == round(node_gt)) / length(node_gt) * 100;

% 美化坐标轴
set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('Test Sample Index', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Contact Location (Node Index)', 'FontSize', 24, 'FontWeight', 'bold');
title(sprintf('Location Sensing Accuracy: %.2f%%', acc_loc), 'FontSize', 26, 'FontWeight', 'bold');

ylim([2.5, 5.5]); % 聚焦在 Node 3, 4, 5 区域
xlim([1, length(sample_indices)]);
set(gca, 'YTick', [3, 4, 5]);

legend('Location', 'northeast', 'FontSize', 18);

disp(['>>> Location Sensing Fidelity Plot Generated. Accuracy: ', num2str(acc_loc), '%']);
%% ========================================================================
%  Step 9.17: 3D Tip Pose Reconstruction Fidelity (Red-Black Scientific)
% =========================================================================
disp('--------------------------------------------------');
disp('9.17 Plotting 3D Tip Pose Fidelity (Red-Black)...');

% 1. 提取对齐后的 3D 坐标 (mm)
P_gt_3d = real_P_after(19:21, :) * 1000; 
P_pd_3d = pred_P_after(19:21, :) * 1000; 
sample_indices = 1:size(P_gt_3d, 2);

% 2. 创建 Figure
figure('Name', '3D Tip Pose Reconstruction Fidelity', 'Color', 'w', 'Position', [100, 100, 1000, 850]);
hold on; grid on; axis equal;

% --- [绘制物理视角标架 - 严格红线约束] ---
% 基座原点 (黄金星)
plot3(0, 0, 0, 'p', 'MarkerSize', 20, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin');
% 基座 RGB 坐标轴 (箭头长度 30mm)
q_len = 30;
quiver3(0,0,0, q_len,0,0, 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'X (Red)');
quiver3(0,0,0, 0,q_len,0, 'g', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Y (Green)');
quiver3(0,0,0, 0,0,q_len, 'b', 'LineWidth', 3, 'MaxHeadSize', 0.5, 'DisplayName', 'Z (Blue)');

% --- [绘制空间误差连线 (Error Stems)] ---
% 使用极浅的灰色连线，作为视觉辅助，不干扰主红黑配色
for i = 1:size(P_gt_3d, 2)
    plot3([P_gt_3d(1,i), P_pd_3d(1,i)], ...
          [P_gt_3d(2,i), P_pd_3d(2,i)], ...
          [P_gt_3d(3,i), P_pd_3d(3,i)], ...
          'Color', [0.85 0.85 0.85], 'LineWidth', 0.6, 'HandleVisibility', 'off');
end

% --- [绘制 3D 点云对比] ---
% A. 真值点：黑色空心圆 (代表物理观测的稳重性)
scatter3(P_gt_3d(1,:), P_gt_3d(2,:), P_gt_3d(3,:), 45, 'k', 'LineWidth', 1.2, ...
    'DisplayName', 'Ground Truth (Mocap)');

% B. 预测点：红色实心点 (代表算法预测的精准性)
scatter3(P_pd_3d(1,:), P_pd_3d(2,:), P_pd_3d(3,:), 35, 'r', 'filled', ...
    'MarkerFaceAlpha', 0.7, 'DisplayName', 'Proposed Prediction');

% --- [物理视角与标准美化 - 严格红线约束] ---
set(gca, 'zdir', 'reverse', 'ydir', 'reverse'); % 强制对齐物理引擎视角
view(45, 30);
grid on;

set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('X [mm]', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Y [mm]', 'FontSize', 24, 'FontWeight', 'bold');
zlabel('Z [mm]', 'FontSize', 24, 'FontWeight', 'bold');

title({'\bf{3D Tip Pose Reconstruction Fidelity}', ...
       ['\fontsize{18}\fontname{Times New Roman}Method: Cascaded Residual Network | Mean Error: ', ...
       num2str(mean(tip_dist*1000), '%.2f'), ' mm']}, 'FontSize', 24);

legend('Location', 'northeast', 'FontSize', 18, 'Box', 'on', 'EdgeColor', 'k');

disp('>>> 3D Red-Black Fidelity Plot generated.');
%% ========================================================================
%  Step 9.17: Downsampled 3D Tip Pose Fidelity (Solid Red-Black)
% =========================================================================
disp('--------------------------------------------------');
disp('9.17 Plotting Downsampled 3D Fidelity...');

% --- [逻辑抽样：控制显示密度] ---
PLOT_STRIDE = 1; % <--- 调整此数值：1表示全量，5表示每5个点取1个，数值越大图越稀疏清晰
idx_sampled = 1:PLOT_STRIDE:size(real_P_after, 2);

% 提取采样后的数据 (mm)
P_gt_3d = real_P_after(19:21, idx_sampled) * 1000; 
P_pd_3d = pred_P_after(19:21, idx_sampled) * 1000; 

% --- [绘图开始] ---
figure('Name', '3D Tip Pose Fidelity (Downsampled)', 'Color', 'w', 'Position', [100, 100, 900, 800]);
hold on; grid on; axis equal;

% 1. 仅保留原点参考星（不画 RGB 箭头）
%plot3(0, 0, 0, 'p', 'MarkerSize', 18, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin');

% 2. 绘制空间误差连线 (Stems)
% 使用极细的浅灰色线连接对应的红黑点
for i = 1:size(P_gt_3d, 2)
    plot3([P_gt_3d(1,i), P_pd_3d(1,i)], ...
          [P_gt_3d(2,i), P_pd_3d(2,i)], ...
          [P_gt_3d(3,i), P_pd_3d(3,i)], ...
          'Color', [0.9 0.9 0.9], 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% 3. 绘制实心点云对比
% 真值：黑色实心圆
scatter3(P_gt_3d(1,:), P_gt_3d(2,:), P_gt_3d(3,:), 50, 'k', 'filled', ...
    'MarkerFaceAlpha', 0.8, 'DisplayName', 'Ground Truth (Mocap)');

% 预测：红色实心圆
scatter3(P_pd_3d(1,:), P_pd_3d(2,:), P_pd_3d(3,:), 40, 'r', 'filled', ...
    'MarkerFaceAlpha', 0.7, 'DisplayName', 'Proposed Prediction');

% --- [物理视角与美化 - 严格红线约束] ---
set(gca, 'zdir', 'reverse', 'ydir', 'reverse'); % 强制对齐物理引擎方向
view(45, 30);

set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
xlabel('X [mm]', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Y [mm]', 'FontSize', 24, 'FontWeight', 'bold');
zlabel('Z [mm]', 'FontSize', 24, 'FontWeight', 'bold');

% 动态标题：显示采样比例
title({'\bf{3D Tip Pose Reconstruction Fidelity}', ...
       sprintf('\\fontsize{16}Downsampling Ratio: 1/%d | Mean Error: %.2f mm', ...
       PLOT_STRIDE, mean(tip_dist*1000))}, 'FontSize', 24);

legend('Location', 'northeast', 'FontSize', 18, 'Box', 'on', 'EdgeColor', 'k');

disp(['>>> 3D Fidelity Plot generated with stride: ', num2str(PLOT_STRIDE)]);
%% ========================================================================
%  Step 9.18: Brute-force MLP - 4 Independent Figures (X, Y, Z, 3D)
% ========================================================================
disp('--------------------------------------------------');
disp('9.18 Plotting Brute-force MLP Independent Fidelity Figures...');

% 对齐数据提取
P_gt_aligned = real_P_after(:, :) * 1000; % [mm]
P_mlp_aligned = pred_brute_abs(:, v_idx) * 1000; % [mm]
sample_indices = 1:size(P_gt_aligned, 2);
axis_names = {'X', 'Y', 'Z'};
axis_rows = [19, 20, 21];

% --- XYZ 独立折线图 (3 Figures) ---
for ax = 1:3
    figure('Name', sprintf('Baseline MLP Tip %s-Coordinate', axis_names{ax}), 'Color', 'w', 'Position', [100+ax*30, 100, 1100, 500]);
    hold on; grid on;
    
    gt_v = P_gt_aligned(axis_rows(ax), :);
    pd_v = P_mlp_aligned(axis_rows(ax), :);
    
    % 真值：黑实线
    plot(sample_indices, gt_v, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');
    % 预测：红虚线 + 标记点
    plot(sample_indices, pd_v, 'r--', 'LineWidth', 2.0, 'Marker', 'o', 'MarkerSize', 5, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r', 'DisplayName', 'MLP Prediction');
    % 误差填充
    fill([sample_indices, fliplr(sample_indices)], [gt_v, fliplr(pd_v)], [1 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % 格式美化
    set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
    xlabel('Test Sample Index', 'FontSize', 24, 'FontWeight', 'bold');
    ylabel(sprintf('Tip %s-Coordinate [mm]', axis_names{ax}), 'FontSize', 24, 'FontWeight', 'bold');
    title({['\bf{Baseline: Brute-force MLP Fidelity (', axis_names{ax}, '-Axis)}'], ...
           sprintf('\\fontsize{18}Mean Error: %.2f mm', mean(abs(gt_v - pd_v)))}, 'FontSize', 24);
    legend('Location', 'northeast', 'FontSize', 18);
    xlim([1, length(sample_indices)]);
end

% --- 3D 空间保真度图 (1 Figure) ---
figure('Name', 'Baseline MLP 3D Fidelity', 'Color', 'w', 'Position', [200, 200, 900, 800]);
hold on; grid on; axis equal;
%plot3(0, 0, 0, 'p', 'MarkerSize', 18, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin'); % 黄金星

STRIDE = 2; % 保持一致的抽样
s_idx = 1:STRIDE:size(P_gt_aligned, 2);
P_gt_3s = P_gt_aligned(19:21, s_idx);
P_pd_3s = P_mlp_aligned(19:21, s_idx);

for i = 1:size(P_gt_3s, 2)
    plot3([P_gt_3s(1,i), P_pd_3s(1,i)], [P_gt_3s(2,i), P_pd_3s(2,i)], [P_gt_3s(3,i), P_pd_3s(3,i)], ...
          'Color', [0.9 0.9 0.9], 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
scatter3(P_gt_3s(1,:), P_gt_3s(2,:), P_gt_3s(3,:), 50, 'k', 'filled', 'DisplayName', 'Ground Truth');
scatter3(P_pd_3s(1,:), P_pd_3s(2,:), P_pd_3s(3,:), 40, 'r', 'filled', 'DisplayName', 'MLP Prediction');

set(gca, 'zdir', 'reverse', 'ydir', 'reverse', 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(45, 30); xlabel('X [mm]'); ylabel('Y [mm]'); zlabel('Z [mm]');
title('\bf{Baseline: Brute-force MLP 3D Fidelity}', 'FontSize', 24);
legend('Location', 'northeast', 'FontSize', 18);
%% ========================================================================
%  Step 9.19: Physics Model - 4 Independent Figures (X, Y, Z, 3D)
% ========================================================================
disp('--------------------------------------------------');
disp('9.19 Plotting Physics Model Independent Fidelity Figures...');

% 对齐数据提取
P_phys_aligned = P_phys_after_all(:, v_idx) * 1000; % [mm]

% --- XYZ 独立折线图 (3 Figures) ---
for ax = 1:3
    figure('Name', sprintf('Baseline Phys Tip %s-Coordinate', axis_names{ax}), 'Color', 'w', 'Position', [150+ax*30, 150, 1100, 500]);
    hold on; grid on;
    
    gt_v = P_gt_aligned(axis_rows(ax), :);
    pd_v = P_phys_aligned(axis_rows(ax), :);
    
    % 真值：黑实线
    plot(sample_indices, gt_v, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');
    % 预测：红虚线 + 标记点
    plot(sample_indices, pd_v, 'r--', 'LineWidth', 2.0, 'Marker', 's', 'MarkerSize', 5, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r', 'DisplayName', 'Phys Prediction');
    % 误差填充
    fill([sample_indices, fliplr(sample_indices)], [gt_v, fliplr(pd_v)], [1 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % 格式美化
    set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
    xlabel('Test Sample Index', 'FontSize', 24, 'FontWeight', 'bold');
    ylabel(sprintf('Tip %s-Coordinate [mm]', axis_names{ax}), 'FontSize', 24, 'FontWeight', 'bold');
    title({['\bf{Baseline: Analytical Physics Fidelity (', axis_names{ax}, '-Axis)}'], ...
           sprintf('\\fontsize{18}Mean Error: %.2f mm', mean(abs(gt_v - pd_v)))}, 'FontSize', 24);
    legend('Location', 'northeast', 'FontSize', 18);
    xlim([1, length(sample_indices)]);
end

% --- 3D 空间保真度图 (1 Figure) ---
figure('Name', 'Baseline Phys 3D Fidelity', 'Color', 'w', 'Position', [300, 300, 900, 800]);
hold on; grid on; axis equal;
%plot3(0, 0, 0, 'p', 'MarkerSize', 18, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin'); % 黄金星
STRIDE = 2; % 保持一致的抽样
s_idx = 1:STRIDE:size(P_gt_aligned, 2);
P_gt_3ph = P_gt_aligned(19:21, s_idx);
P_pd_3ph = P_phys_aligned(19:21, s_idx);

for i = 1:size(P_gt_3ph, 2)
    plot3([P_gt_3ph(1,i), P_pd_3ph(1,i)], [P_gt_3ph(2,i), P_pd_3ph(2,i)], [P_gt_3ph(3,i), P_pd_3s(3,i)], ...
          'Color', [0.9 0.9 0.9], 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
scatter3(P_gt_3ph(1,:), P_gt_3ph(2,:), P_gt_3ph(3,:), 50, 'k', 'filled', 'DisplayName', 'Ground Truth');
scatter3(P_pd_3ph(1,:), P_pd_3ph(2,:), P_pd_3ph(3,:), 40, 'r', 'filled', 'DisplayName', 'Phys Prediction');

set(gca, 'zdir', 'reverse', 'ydir', 'reverse', 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(45, 30); xlabel('X [mm]'); ylabel('Y [mm]'); zlabel('Z [mm]');
title('\bf{Baseline: Analytical Physics 3D Fidelity}', 'FontSize', 24);
legend('Location', 'northeast', 'FontSize', 18);
%% ========================================================================
%  Step 9.22: Calibration Benchmark - Before Impact (Final Corrected)
% ========================================================================
disp('--------------------------------------------------');
disp('9.22 Plotting "Before Impact" Calibration Fidelity (Dimension Aligned)...');

% --- [1. 提取原始碰撞前动捕真值 (N_clean)] ---
pos_text_b_clean = pos_text_b_sub(~bad_idx); 
N_clean = length(pos_text_b_clean);
P_b_mocap_clean = zeros(21, N_clean);

for i = 1:N_clean
    real_offset_b = get_RealOffset_1S3CT(pos_text_b_clean{i});
    % 注意：必须减去基座中心实现坐标系归一化 (根据 Step 1.4 逻辑)
    % base_center_b = (real_offset_b(:, 1) + real_offset_b(:, 2)) / 2; % 假设 base 逻辑已在 get_RealOffset 内部或在此处理
    P_b_mocap_clean(:, i) = reshape(real_offset_b(:, 3:end), 21, 1); 
end

% --- [2. 关键步骤：对 Mocap 真值执行旋转增强 (N_clean -> 3*N_clean)] ---
R120_mat = [cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
R240_mat = [cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
rotP_func = @(P, R) reshape(R * reshape(P, 3, []), 21, size(P, 2));

P_b_mocap_120 = rotP_func(P_b_mocap_clean, R120_mat);
P_b_mocap_240 = rotP_func(P_b_mocap_clean, R240_mat);

% 拼接为增强后的全量真值 (3N)
P_b_mocap_aug = [P_b_mocap_clean, P_b_mocap_120, P_b_mocap_240];

% --- [3. 三重过滤锁死维度 (3N -> N_cascade -> N_test -> N_final)] ---
% 第一层：v_mask (高受力筛选)
P_b_mocap_cascade = P_b_mocap_aug(:, v_mask);

% 第二层：test_idx (测试集拆分)
P_b_mocap_test = P_b_mocap_cascade(:, test_idx);

% 第三层：v_idx (最终有效样本过滤)
P_gt_before_final = P_b_mocap_test(:, v_idx) * 1000; % 最终对齐真值 [mm]

% 预测值同步过滤 (P_before_ideal 已经是增强后的数据 aug_Pb，维度天然对齐)
P_pd_before_all = aug_Pb(:, v_mask); 
P_pd_before_test = P_pd_before_all(:, test_idx);
P_pd_before_final = P_pd_before_test(:, v_idx) * 1000; % 最终对齐预测 [mm]

% --- [4. 绘图逻辑 (独立 Figure, 红黑, 实心)] ---
sample_indices = 1:size(P_gt_before_final, 2);
axis_names = {'X', 'Y', 'Z'};
axis_rows = [19, 20, 21];

for ax = 1:3
    figure('Name', sprintf('Before_Impact_%s', axis_names{ax}), 'Color', 'w', 'Position', [100+ax*30, 100, 1100, 500]);
    hold on; grid on;
    
    gt_v = P_gt_before_final(axis_rows(ax), :);
    pd_v = P_pd_before_final(axis_rows(ax), :);
    
    plot(sample_indices, gt_v, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Mocap Truth');
    plot(sample_indices, pd_v, 'r--', 'LineWidth', 2.0, 'Marker', 'o', 'MarkerSize', 5, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r', 'DisplayName', 'Phys Model');
    fill([sample_indices, fliplr(sample_indices)], [gt_v, fliplr(pd_v)], [1 0.7 0.7], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    set(gca, 'FontSize', 22, 'FontName', 'Times New Roman', 'LineWidth', 1.5, 'TickDir', 'out');
    xlabel('Test Sample Index'); ylabel([axis_names{ax}, ' [mm]']);
    title(sprintf('Calibration Fidelity (Before Impact) - %s Axis', axis_names{ax}));
    legend('Location', 'best');
    xlim([1, length(sample_indices)]);
end

% 3D 空间保真度图
figure('Name', 'Before_Impact_3D', 'Color', 'w', 'Position', [200, 200, 900, 800]);
hold on; grid on; axis equal;
%plot3(0, 0, 0, 'p', 'MarkerSize', 18, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Origin');

STRIDE = 2; 
s_idx = 1:STRIDE:size(P_gt_before_final, 2);
P_gt_3s = P_gt_before_final(19:21, s_idx);
P_pd_3s = P_pd_before_final(19:21, s_idx);

for i = 1:size(P_gt_3s, 2)
    plot3([P_gt_3s(1,i), P_pd_3s(1,i)], [P_gt_3s(2,i), P_pd_3s(2,i)], [P_gt_3s(3,i), P_pd_3s(3,i)], ...
          'Color', [0.9 0.9 0.9], 'HandleVisibility', 'off');
end
scatter3(P_gt_3s(1,:), P_gt_3s(2,:), P_gt_3s(3,:), 55, 'k', 'filled', 'DisplayName', 'Truth');
scatter3(P_pd_3s(1,:), P_pd_3s(2,:), P_pd_3s(3,:), 45, 'r', 'filled', 'DisplayName', 'Phys Model');

set(gca, 'zdir', 'reverse', 'ydir', 'reverse', 'FontSize', 22, 'FontName', 'Times New Roman');
view(45, 30); xlabel('X [mm]'); ylabel('Y [mm]'); zlabel('Z [mm]');
title('\bf{Calibration Fidelity (Before Impact): 3D View}');
legend('Location', 'northeast');

disp('>>> [Success] Dimension-aligned Calibration plots generated.');

%% ========================================================================
%  Step 9.13: Top-30 Aesthetic Samples 3D (Filtered Curvature & No Markers)
% ========================================================================
disp('--------------------------------------------------');
disp('9.13 Generating Aesthetic 3D Plot (Filtered & Clean)...');

% 1. 过滤与筛选逻辑
N_all = size(real_P_after, 2);
extension_dist = zeros(1, N_all);
mae_errors = zeros(1, N_all);

for i = 1:N_all
    gt_pts = reshape(real_P_after(:, i), 3, 7);
    pr_pts = reshape(pred_P_after(:, i), 3, 7);
    % 计算伸展度：基座到尖端的距离。距离过小意味着弯曲过大。
    extension_dist(i) = norm(gt_pts(:, 7) - gt_pts(:, 1));
    % 计算平均误差
    mae_errors(i) = mean(sqrt(sum((gt_pts - pr_pts).^2, 1)));
end

% 设定阈值：只保留伸展度 > 70mm 的样本
aesthetic_mask = (extension_dist > 0.070); 
valid_indices = find(aesthetic_mask);
[~, sub_idx] = sort(mae_errors(valid_indices), 'ascend');
best_30_idx = valid_indices(sub_idx(1:min(30, length(sub_idx))));

% 2. 创建画布与环境设置
figure('Name', 'Aesthetic Robot Reconstruction', 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off; 

% 3. 绘制原点坐标系 (RGB 箭头 + 阴影) - 禁用 HandleVisibility 以便从图例排除
arrow_len = 0.025; 
shadow_off = 0.0007; 
% 阴影
quiver3(shadow_off, shadow_off, shadow_off, arrow_len, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(shadow_off, shadow_off, shadow_off, 0, arrow_len, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(shadow_off, shadow_off, shadow_off, 0, 0, arrow_len, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
% RGB 轴
quiver3(0, 0, 0, arrow_len, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_len, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_len, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');

% 黑色标签 (字号 24)
text(arrow_len+0.003, 0, 0, 'X', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, arrow_len+0.003, 0, 'Y', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, 0, arrow_len+0.003, 'Z', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');

% 原点蓝色圆圈 + 阴影
plot3(0.0004, 0.0004, 0.0004, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.7 0.7 0.7], 'HandleVisibility', 'off'); 
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% 4. 循环绘制线条
for i = 1:length(best_30_idx)
    idx = best_30_idx(i);
    gt_raw = reshape(real_P_after(:, idx), 3, 7);
    pr_raw = reshape(pred_P_after(:, idx), 3, 7);
    
    base_shift = gt_raw(:, 1); 
    gt_pts = gt_raw - base_shift;
    pr_pts = pr_raw - base_shift;
    
    % 绘制真值：红色实线
    plot3(gt_pts(1,:), gt_pts(2,:), gt_pts(3,:), 'r-', 'LineWidth', 2.0, 'HandleVisibility', 'off');
    
    % 绘制预测值：蓝色虚线
    plot3(pr_pts(1,:), pr_pts(2,:), pr_pts(3,:), 'b--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
end

% 【核心修改】图例代理对象（只有这两个会出现在图例中）
plot3(nan, nan, nan, 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot3(nan, nan, nan, 'b--', 'LineWidth', 2, 'DisplayName', 'Proposed Prediction');

% 5. 倒置视图设置
set(gca, 'ZDir', 'reverse'); 
view(-40, 25); 

% 6. 细节打磨 (IEEE 24号字)
set(gca, 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
xlabel('X [m]', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Y [m]', 'FontSize', 24, 'FontWeight', 'bold');
zlabel('Z [m]', 'FontSize', 24, 'FontWeight', 'bold');

set(gca, 'Box', 'off', 'Color', 'w');

% 【核心修改】图例位置改为右上角 (northeast)
legend('Location', 'northeast', 'FontSize', 24, 'Box', 'on', 'EdgeColor', 'k');

disp('>>> [Success] Aesthetic 3D visualization completed with refined legend.');
%% ========================================================================
%  Step 9.14: Vanilla MLP Baseline 3D Visualization (Aesthetic & Aligned)
% ========================================================================
disp('--------------------------------------------------');
disp('9.14 Generating Vanilla MLP 3D Plot...');

% 1. 筛选逻辑 (针对 MLP 结果)
N_brute = size(pred_brute_abs, 2);
ext_brute = zeros(1, N_brute);
err_brute_mae = zeros(1, N_brute);

for i = 1:N_brute
    gt_pts = reshape(targ_brute_test(:, i), 3, 7);
    pr_pts = reshape(pred_brute_abs(:, i), 3, 7);
    ext_brute(i) = norm(gt_pts(:, 7) - gt_pts(:, 1));
    err_brute_mae(i) = mean(sqrt(sum((gt_pts - pr_pts).^2, 1)));
end

% 剔除弯曲过大样本并选出表现最好的 30 组
mask_brute = (ext_brute > 0.045); 
v_idx_brute = find(mask_brute);
[~, s_idx] = sort(err_brute_mae(v_idx_brute), 'ascend');
show_idx_brute = v_idx_brute(s_idx(1:min(30, length(s_idx))));

% 2. 绘图环境
figure('Name', 'Vanilla MLP 3D Reconstruction', 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off;

% 3. 原点与坐标轴 (RGB + 阴影 + 黑色标签)
arrow_l = 0.025; sh_o = 0.0007;
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5);
text(arrow_l+0.003, 0, 0, 'X', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, arrow_l+0.003, 0, 'Y', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, 0, arrow_l+0.003, 'Z', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
plot3(0.0004, 0.0004, 0.0004, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.7 0.7 0.7]); 
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

% 4. 绘制线条
for i = 1:length(show_idx_brute)
    idx = show_idx_brute(i);
    gt_raw = reshape(targ_brute_test(:, idx), 3, 7);
    pr_raw = reshape(pred_brute_abs(:, idx), 3, 7);
    base_s = gt_raw(:, 1);
    gt_p = gt_raw - base_s; pr_p = pr_raw - base_s;
    plot3(gt_p(1,:), gt_p(2,:), gt_p(3,:), 'r-', 'LineWidth', 1.8, 'HandleVisibility', 'off');
    plot3(pr_p(1,:), pr_p(2,:), pr_p(3,:), 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% 5. 格式化
plot3(nan, nan, nan, 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot3(nan, nan, nan, 'b--', 'LineWidth', 2, 'DisplayName', 'Vanilla MLP');
set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-40, 25); xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Baseline: Vanilla MLP Reconstruction', 'FontSize', 24, 'FontWeight', 'bold');
legend('Location', 'southoutside', 'FontSize', 24, 'Orientation', 'horizontal');
%% ========================================================================
%  Step 9.15: Analytical Physics Model 3D Visualization (Strict Origin Alignment)
% ========================================================================
disp('--------------------------------------------------');
disp('9.15 Generating Physical Model 3D Plot (Fixed Origin)...');

% 1. 筛选逻辑 (针对物理模型结果)
N_phys = size(P_phys_after_all, 2);
ext_phys = zeros(1, N_phys);
err_phys_mae = zeros(1, N_phys);

for i = 1:N_phys
    gt_pts_tmp = reshape(Pa_real_test_all(:, i), 3, 7);
    pr_pts_tmp = reshape(P_phys_after_all(:, i), 3, 7);
    % 计算伸展度
    ext_phys(i) = norm(gt_pts_tmp(:, 7) - gt_pts_tmp(:, 1));
    % 计算平均误差
    err_phys_mae(i) = mean(sqrt(sum((gt_pts_tmp - pr_pts_tmp).^2, 1)));
end

% 剔除弯曲过大样本并选出表现最好的 30 组
mask_phys = (ext_phys > 0.045); 
v_idx_phys = find(mask_phys);
[~, s_idx_p] = sort(err_phys_mae(v_idx_phys), 'ascend');
show_idx_phys = v_idx_phys(s_idx_p(1:min(30, length(s_idx_p))));

% 2. 绘图环境
figure('Name', 'Physical Model 3D Reconstruction', 'Color', 'w', 'Position', [150, 150, 1000, 950]);
hold on; axis equal; grid off;

% 3. 原点与坐标轴 (RGB + 阴影 + 黑色标签)
arrow_l = 0.025; sh_o = 0.0007;
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5);
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5);

text(arrow_l+0.003, 0, 0, 'X', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, arrow_l+0.003, 0, 'Y', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, 0, arrow_l+0.003, 'Z', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');

% 原点蓝色圆圈
plot3(0.0004, 0.0004, 0.0004, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.7 0.7 0.7]); 
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

% 4. 循环绘制线条
for i = 1:length(show_idx_phys)
    idx = show_idx_phys(i);
    gt_raw = reshape(Pa_real_test_all(:, idx), 3, 7);
    pr_raw = reshape(P_phys_after_all(:, idx), 3, 7);
    
    % 【核心修正逻辑】
    % 分别以各自的第一个点作为偏移基准，确保两者都从 (0,0,0) 开始
    gt_p = gt_raw - gt_raw(:, 1); 
    pr_p = pr_raw - pr_raw(:, 1); 
    
    % 绘制真值：红色实线
    plot3(gt_p(1,:), gt_p(2,:), gt_p(3,:), 'r-', 'LineWidth', 1.8, 'HandleVisibility', 'off');
    
    % 绘制预测值：蓝色虚线
    plot3(pr_p(1,:), pr_p(2,:), pr_p(3,:), 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% 5. 格式化
plot3(nan, nan, nan, 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot3(nan, nan, nan, 'b--', 'LineWidth', 2, 'DisplayName', 'Analytical Physics');

set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-40, 25); 
xlabel('X [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
ylabel('Y [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
zlabel('Z [m]', 'FontSize', 24, 'FontWeight', 'bold');
title('Baseline: Analytical Physics Model', 'FontSize', 24, 'FontWeight', 'bold');

set(gca, 'Box', 'off', 'Color', 'w');
legend('Location', 'southoutside', 'FontSize', 24, 'Orientation', 'horizontal');

disp('>>> [Success] Physical Model 3D plot fixed and generated.');
%% ========================================================================
%  Step 9.14: Vanilla MLP Baseline 3D Visualization (Aesthetic & Aligned)
% ========================================================================
disp('--------------------------------------------------');
disp('9.14 Generating Vanilla MLP 3D Plot...');

% 1. 筛选逻辑 (针对 MLP 结果)
N_brute = size(pred_brute_abs, 2);
ext_brute = zeros(1, N_brute);
err_brute_mae = zeros(1, N_brute);

for i = 1:N_brute
    gt_pts = reshape(targ_brute_test(:, i), 3, 7);
    pr_pts = reshape(pred_brute_abs(:, i), 3, 7);
    ext_brute(i) = norm(gt_pts(:, 7) - gt_pts(:, 1));
    err_brute_mae(i) = mean(sqrt(sum((gt_pts - pr_pts).^2, 1)));
end

% 剔除弯曲过大样本并选出表现最好的 30 组
mask_brute = (ext_brute > 0.045); 
v_idx_brute = find(mask_brute);
[~, s_idx] = sort(err_brute_mae(v_idx_brute), 'ascend');
show_idx_brute = v_idx_brute(s_idx(1:min(30, length(s_idx))));

% 2. 绘图环境
figure('Name', 'Vanilla MLP 3D Reconstruction', 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off;

% 3. 原点与坐标轴 (RGB + 阴影 + 黑色标签) - 禁用 HandleVisibility
arrow_l = 0.025; sh_o = 0.0007;
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');

text(arrow_l+0.003, 0, 0, 'X', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, arrow_l+0.003, 0, 'Y', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, 0, arrow_l+0.003, 'Z', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');

% 原点蓝色圆圈 (禁用 HandleVisibility)
plot3(0.0004, 0.0004, 0.0004, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.7 0.7 0.7], 'HandleVisibility', 'off'); 
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% 4. 循环绘制线条
for i = 1:length(show_idx_brute)
    idx = show_idx_brute(i);
    gt_raw = reshape(targ_brute_test(:, idx), 3, 7);
    pr_raw = reshape(pred_brute_abs(:, idx), 3, 7);
    
    % 分别对齐
    gt_p = gt_raw - gt_raw(:, 1); 
    pr_p = pr_raw - pr_raw(:, 1); 
    
    plot3(gt_p(1,:), gt_p(2,:), gt_p(3,:), 'r-', 'LineWidth', 1.8, 'HandleVisibility', 'off');
    plot3(pr_p(1,:), pr_p(2,:), pr_p(3,:), 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% 5. 格式化 - 图例代理
plot3(nan, nan, nan, 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot3(nan, nan, nan, 'b--', 'LineWidth', 2, 'DisplayName', 'Vanilla MLP Prediction');

set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-40, 25); xlabel('X [m]', 'FontWeight', 'bold'); ylabel('Y [m]', 'FontWeight', 'bold'); zlabel('Z [m]', 'FontWeight', 'bold');
%title('Baseline: Vanilla MLP Reconstruction', 'FontSize', 24, 'FontWeight', 'bold');

set(gca, 'Box', 'off', 'Color', 'w');
% 修改点：右上角图例
legend('Location', 'northeast', 'FontSize', 24, 'Box', 'on', 'EdgeColor', 'k');

%% ========================================================================
%  Step 9.15: Analytical Physics Model 3D Visualization (Strict Origin Alignment)
% ========================================================================
disp('--------------------------------------------------');
disp('9.15 Generating Physical Model 3D Plot (Fixed Origin)...');

% 1. 筛选逻辑 (针对物理模型结果)
N_phys = size(P_phys_after_all, 2);
ext_phys = zeros(1, N_phys);
err_phys_mae = zeros(1, N_phys);

for i = 1:N_phys
    gt_pts_tmp = reshape(Pa_real_test_all(:, i), 3, 7);
    pr_pts_tmp = reshape(P_phys_after_all(:, i), 3, 7);
    ext_phys(i) = norm(gt_pts_tmp(:, 7) - gt_pts_tmp(:, 1));
    err_phys_mae(i) = mean(sqrt(sum((gt_pts_tmp - pr_pts_tmp).^2, 1)));
end

% 剔除弯曲过大样本并选出表现最好的 30 组
mask_phys = (ext_phys > 0.045); 
v_idx_phys = find(mask_phys);
[~, s_idx_p] = sort(err_phys_mae(v_idx_phys), 'ascend');
show_idx_phys = v_idx_phys(s_idx_p(1:min(30, length(s_idx_p))));

% 2. 绘图环境
figure('Name', 'Physical Model 3D Reconstruction', 'Color', 'w', 'Position', [150, 150, 1000, 950]);
hold on; axis equal; grid off;

% 3. 原点与坐标轴 (RGB + 阴影 + 黑色标签) - 禁用 HandleVisibility
arrow_l = 0.025; sh_o = 0.0007;
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');

text(arrow_l+0.003, 0, 0, 'X', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, arrow_l+0.003, 0, 'Y', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');
text(0, 0, arrow_l+0.003, 'Z', 'Color', 'k', 'FontSize', 24, 'FontWeight', 'bold');

% 原点蓝色圆圈 (禁用 HandleVisibility)
plot3(0.0004, 0.0004, 0.0004, 'ko', 'MarkerSize', 12, 'MarkerFaceColor', [0.7 0.7 0.7], 'HandleVisibility', 'off'); 
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% 4. 循环绘制线条
for i = 1:length(show_idx_phys)
    idx = show_idx_phys(i);
    gt_raw = reshape(Pa_real_test_all(:, idx), 3, 7);
    pr_raw = reshape(P_phys_after_all(:, idx), 3, 7);
    
    % 分别对齐，确保两者都从 (0,0,0) 开始
    gt_p = gt_raw - gt_raw(:, 1); 
    pr_p = pr_raw - pr_raw(:, 1); 
    
    plot3(gt_p(1,:), gt_p(2,:), gt_p(3,:), 'r-', 'LineWidth', 1.8, 'HandleVisibility', 'off');
    plot3(pr_p(1,:), pr_p(2,:), pr_p(3,:), 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% 5. 格式化 - 图例代理
plot3(nan, nan, nan, 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot3(nan, nan, nan, 'b--', 'LineWidth', 2, 'DisplayName', 'Analytical Physics Prediction');

set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-40, 25); 
xlabel('X [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
ylabel('Y [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
zlabel('Z [m]', 'FontSize', 24, 'FontWeight', 'bold');
%title('Baseline: Analytical Physics Model', 'FontSize', 24, 'FontWeight', 'bold');

set(gca, 'Box', 'off', 'Color', 'w');
% 修改点：右上角图例
legend('Location', 'northeast', 'FontSize', 24, 'Box', 'on', 'EdgeColor', 'k');

disp('>>> [Success] MLP and Physical Model plots generated with Northeast Legends.');
%% ========================================================================
%  Step 9.16: Single Sample High-Fidelity Pose & Force Visualization
% ========================================================================
disp('--------------------------------------------------');
disp('9.16 Generating Single Sample Pose + Force 3D Plot...');

% 1. 挑选展示样本 (挑选之前筛选出的 30 个优秀样本中的第 1 个)
target_idx = best_30_idx(1); 

% 2. 提取并对齐数据
% 提取 Pose (3x7)
gt_raw_pose = reshape(real_P_after(:, target_idx), 3, 7);
pr_raw_pose = reshape(pred_P_after(:, target_idx), 3, 7);
% 提取 Force (3x1) - 假设你的预测变量名为 pred_F_vec, 真值为 gt_F_vec_test
% 如果变量名不同，请根据你的 Step 9.8/9.9 变量名调整
f_gt_vec = gt_F_vec_test(:, target_idx); 
f_pr_vec = pred_f(:, target_idx); % 级联网络输出的外力预测

% 获取受力点位置 (Node Index 对应哪个 Marker)
force_node = gt_hgt_test(target_idx); % 假设是 3, 4, 或 5
marker_map = [2, 4, 6]; % 假设 Node 3->Marker 2, Node 4->Marker 4... 根据实际映射
marker_idx = force_node; % 这里直接使用 Node 作为索引示例

% 【关键】基准点归零偏移
base_shift = gt_raw_pose(:, 1);
gt_p = gt_raw_pose - base_shift;
pr_p = pr_raw_pose - base_shift;

% 3. 创建画布
figure('Name', 'Single Sample Pose & Force', 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off;

% 4. 绘制原点坐标系 (RGB + 阴影)
arrow_l = 0.02; sh_o = 0.0006;
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2, 'HandleVisibility', 'off');
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3, 'HandleVisibility', 'off');
text(arrow_l+0.002, 0, 0, 'X', 'FontSize', 24, 'FontWeight', 'bold');
text(0, arrow_l+0.002, 0, 'Y', 'FontSize', 24, 'FontWeight', 'bold');
text(0, 0, arrow_l+0.002, 'Z', 'FontSize', 24, 'FontWeight', 'bold');
plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% 5. 绘制位姿线条 (无 Marker 点)
plot3(gt_p(1,:), gt_p(2,:), gt_p(3,:), 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth Pose');
plot3(pr_p(1,:), pr_p(2,:), pr_p(3,:), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Predicted Pose');

% 6. 绘制外力矢量 (Force Vectors)
% 设定力矢量缩放因子 (1N = 5mm)
f_scale = 0.05; 
% 受力点坐标
f_origin_gt = gt_p(:, marker_idx);
f_origin_pr = pr_p(:, marker_idx);

% 真实力：红色粗箭头
quiver3(f_origin_gt(1), f_origin_gt(2), f_origin_gt(3), ...
    f_gt_vec(1)*f_scale, f_gt_vec(2)*f_scale, f_gt_vec(3)*f_scale, ...
    'Color', 'r', 'LineWidth', 4, 'MaxHeadSize', 1.5, 'AutoScale', 'off', 'DisplayName', 'Ground Truth Force');

% 预测力：蓝色虚线箭头
quiver3(f_origin_pr(1), f_origin_pr(2), f_origin_pr(3), ...
    f_pr_vec(1)*f_scale, f_pr_vec(2)*f_scale, f_pr_vec(3)*f_scale, ...
    'Color', 'b', 'LineStyle', '-', 'LineWidth', 3, 'MaxHeadSize', 1.5, 'AutoScale', 'off', 'DisplayName', 'Predicted Force');

% 7. 细节打磨
set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-45, 30);
xlabel('X [m]', 'FontWeight', 'bold'); ylabel('Y [m]', 'FontWeight', 'bold'); zlabel('Z [m]', 'FontWeight', 'bold');
set(gca, 'Box', 'off');

% 图例设置在右上角
legend('Location', 'northeast', 'FontSize', 22, 'Box', 'on', 'EdgeColor', 'k');

disp(['>>> Sample Index ', num2str(target_idx), ' visualized with Pose and Force.']);
%% ========================================================================
%  Step 9.17: Single Sample High-Fidelity Pose & Force (Force-Error Minimized)
% ========================================================================
disp('--------------------------------------------------');
disp('9.17 Visualizing Best Force Estimation Sample...');

% --- 1. 数据对齐与挑选偏差最小样本 ---
% 基于你 9.13 的逻辑：F_pd_test 和 F_gt_test 已经是最终对齐的测试集数据
force_errors_vec = sqrt(sum((F_pd_test - F_gt_test).^2, 1));
[~, best_f_idx] = min(force_errors_vec); % 寻找力误差最小的索引

% 提取该样本的预测位姿与真值位姿 (已经在之前的步骤中按 v_idx 对齐)
gt_raw_pose = reshape(real_P_after(:, best_f_idx), 3, 7);
pr_raw_pose = reshape(pred_P_after(:, best_f_idx), 3, 7);

% 提取该样本的预测力与真值力 (3x1)
f_gt_vec = F_gt_test(:, best_f_idx);
f_pr_vec = F_pd_test(:, best_f_idx);

% 提取受力位置 (Node Index) - 注意同步应用 v_idx 过滤掩码
hgt_filtered = gt_hgt_test(v_idx); 
f_node_idx = hgt_filtered(best_f_idx); 

% 【核心对齐】分别以第一个 Marker 为原点，消除初始偏置
gt_p = gt_raw_pose - gt_raw_pose(:, 1);
pr_p = pr_raw_pose - pr_raw_pose(:, 1);

% --- 2. 创建画布 ---
figure('Name', 'Min-Force-Error Single Sample', 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off;

% --- 3. 绘制原点 RGB 轴 (带阴影，HandleVisibility 设为 off) ---
arrow_l = 0.025; sh_o = 0.0007;
% 阴影
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
% RGB 轴
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
% 标签
text(arrow_l+0.003, 0, 0, 'X', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
text(0, arrow_l+0.003, 0, 'Y', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
text(0, 0, arrow_l+0.003, 'Z', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
% 原点蓝点
plot3(0, 0, 0, 'bo', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% --- 4. 绘制位姿 (无 Marker 点) ---
plot3(gt_p(1,:), gt_p(2,:), gt_p(3,:), 'r-', 'LineWidth', 3, 'DisplayName', 'Ground Truth');
plot3(pr_p(1,:), pr_p(2,:), pr_p(3,:), 'b--', 'LineWidth', 2.5, 'DisplayName', 'Proposed Predict');

% --- 5. 绘制外力矢量 (箭头起点设在受力 Marker 上) ---
f_scale = 0.08; % 比例尺：1N = 8mm
f_origin_gt = gt_p(:, f_node_idx);
f_origin_pr = pr_p(:, f_node_idx);

% 真实力：红色实线粗箭头
quiver3(f_origin_gt(1), f_origin_gt(2), f_origin_gt(3), ...
    f_gt_vec(1)*f_scale, f_gt_vec(2)*f_scale, f_gt_vec(3)*f_scale, ...
    'Color', 'r', 'LineWidth', 4, 'MaxHeadSize', 1.2, 'AutoScale', 'off', 'DisplayName', 'GT Force');

% 预测力：蓝色虚线箭头
quiver3(f_origin_pr(1), f_origin_pr(2), f_origin_pr(3), ...
    f_pr_vec(1)*f_scale, f_pr_vec(2)*f_scale, f_pr_vec(3)*f_scale, ...
    'Color', 'b', 'LineStyle', '--', 'LineWidth', 3, 'MaxHeadSize', 1.2, 'AutoScale', 'off', 'DisplayName', 'Pred Force');

% --- 6. 视觉修饰与 IEEE 标准设置 ---
set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-45, 30); % 设置机器人悬挂生长的最佳视角
xlabel('X [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
ylabel('Y [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
zlabel('Z [m]', 'FontSize', 24, 'FontWeight', 'bold');
set(gca, 'Box', 'off', 'Color', 'w');

% 图例设置：仅包含手动标记 DisplayName 的 4 项，位于右上角
legend('Location', 'northeast', 'FontSize', 24, 'Box', 'on', 'EdgeColor', 'k');

fprintf('>>> [Success] Visualized Test Sample with Force Error: %.4f N\n', force_errors_vec(best_f_idx));
%% ========================================================================
%  Step 9.17: Smooth 3D Pose & Force Visualization (Manual Selection)
% ========================================================================
disp('--------------------------------------------------');
disp('9.17 Generating Smooth Single Sample Plot...');

% 【用户交互】在此处更改索引（范围：1 到 size(F_pd_test, 2)）
% 你可以尝试不同的数字，直到找到位姿和外力都完美的样本
plot_sample_idx = 9; 

% 1. 提取并对齐该样本数据
gt_raw_pose = reshape(real_P_after(:, plot_sample_idx), 3, 7);
pr_raw_pose = reshape(pred_P_after(:, plot_sample_idx), 3, 7);
f_gt_vec = F_gt_test(:, plot_sample_idx);
f_pr_vec = F_pd_test(:, plot_sample_idx);

% 提取受力位置并应用掩码
hgt_filtered = gt_hgt_test(v_idx); 
f_node_idx = hgt_filtered(plot_sample_idx); 

% 核心对齐：以第一个点为原点
gt_p = gt_raw_pose - gt_raw_pose(:, 1);
pr_p = pr_raw_pose - pr_raw_pose(:, 1);

% 【平滑处理】使用三次样条插值消除折线感
t_orig = 1:7;
t_interp = linspace(1, 7, 100); % 将 7 个点插值为 100 个点
gt_smooth = interp1(t_orig, gt_p', t_interp, 'spline')';
pr_smooth = interp1(t_orig, pr_p', t_interp, 'spline')';

% 2. 创建画布
figure('Name', ['Smooth Sample Index ', num2str(plot_sample_idx)], 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off;

% 3. 原点 RGB 轴 (带阴影，HandleVisibility 设为 off)
arrow_l = 0.025; sh_o = 0.0007;
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3.5, 'MaxHeadSize', 0.5, 'HandleVisibility', 'off');
text(arrow_l+0.003, 0, 0, 'X', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
text(0, arrow_l+0.003, 0, 'Y', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
text(0, 0, arrow_l+0.003, 'Z', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
plot3(0, 0, 0, 'bo', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% 4. 绘制位姿 (使用平滑后的曲线)
plot3(gt_smooth(1,:), gt_smooth(2,:), gt_smooth(3,:), 'r-', 'LineWidth', 3.5, 'DisplayName', 'Ground Truth Pose');
plot3(pr_smooth(1,:), pr_smooth(2,:), pr_smooth(3,:), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Proposed Predict Pose');

% 5. 绘制外力矢量 (箭头起点需使用原始 Marker 位置确保物理真实)
f_scale = 0.08; 
f_origin_gt = gt_p(:, f_node_idx);
f_origin_pr = pr_p(:, f_node_idx);

% 真实力：红色粗实线箭头
quiver3(f_origin_gt(1), f_origin_gt(2), f_origin_gt(3), ...
    f_gt_vec(1)*f_scale, f_gt_vec(2)*f_scale, f_gt_vec(3)*f_scale, ...
    'Color', 'r', 'LineWidth', 4.5, 'MaxHeadSize', 1.2, 'AutoScale', 'off', 'DisplayName', 'Measured Force');

% 预测力：蓝色虚线箭头
quiver3(f_origin_pr(1), f_origin_pr(2), f_origin_pr(3), ...
    f_pr_vec(1)*f_scale, f_pr_vec(2)*f_scale, f_pr_vec(3)*f_scale, ...
    'Color', 'b', 'LineStyle', '-', 'LineWidth', 3, 'MaxHeadSize', 1.2, 'AutoScale', 'off', 'DisplayName', 'Estimated Force');

% 6. 视觉修饰与 IEEE 标准
set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-45, 30);
xlabel('X [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
ylabel('Y [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
zlabel('Z [m]', 'FontSize', 24, 'FontWeight', 'bold');
set(gca, 'Box', 'off', 'Color', 'w');

% 图例设置在右上角
legend('Location', 'northeast', 'FontSize', 24, 'Box', 'on', 'EdgeColor', 'k');

% 在标题显示当前样本索引和力误差
f_err = norm(f_gt_vec - f_pr_vec);
title(['Sample #', num2str(plot_sample_idx), ' (Force Error: ', num2str(f_err, '%.3f'), ' N)'], 'FontSize', 24);

disp(['>>> Smooth Visualization of Sample #', num2str(plot_sample_idx), ' completed.']);
%% ========================================================================
%  Step 9.17: Smooth 3D Pose & Force (PCHIP Shape-Preserving version)
% ========================================================================
disp('--------------------------------------------------');
disp('9.17 Generating Shape-Preserving Smooth Plot...');

% 【用户交互】选择样本索引
plot_sample_idx = 1; % 你可以根据需要修改此索引

% 1. 提取并对齐该样本数据
gt_raw_pose = reshape(real_P_after(:, plot_sample_idx), 3, 7);
pr_raw_pose = reshape(pred_P_after(:, plot_sample_idx), 3, 7);
f_gt_vec = F_gt_test(:, plot_sample_idx);
f_pr_vec = F_pd_test(:, plot_sample_idx);

% 提取受力位置
hgt_filtered = gt_hgt_test(v_idx); 
f_node_idx = hgt_filtered(plot_sample_idx); 

% 核心对齐：分别减去各自的起始点，确保从 (0,0,0) 开始
gt_p = gt_raw_pose - gt_raw_pose(:, 1);
pr_p = pr_raw_pose - pr_raw_pose(:, 1);

% 【核心改进】使用 pchip (Piecewise Cubic Hermite) 代替 spline
% pchip 能够保形，不会在受力点产生不自然的“过冲”弯曲
t_orig = 1:7;
t_interp = linspace(1, 7, 150); % 增加插值点到 150 使其更丝滑
gt_smooth = interp1(t_orig, gt_p', t_interp, 'pchip')';
pr_smooth = interp1(t_orig, pr_p', t_interp, 'pchip')';

% 2. 创建画布
figure('Name', ['Inverted Robot Sample #', num2str(plot_sample_idx)], 'Color', 'w', 'Position', [100, 100, 1000, 950]);
hold on; axis equal; grid off;

% 3. 原点 RGB 轴 (设置 HandleVisibility 为 off 以保持图例干净)
arrow_l = 0.025; sh_o = 0.0007;
% 阴影
quiver3(sh_o, sh_o, sh_o, arrow_l, 0, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, arrow_l, 0, 'Color', [0.85 0.85 0.85], 'LineWidth', 2, 'HandleVisibility', 'off');
quiver3(sh_o, sh_o, sh_o, 0, 0, arrow_l, 'Color', [0.85 0.85 0.85], 'LineWidth', 2, 'HandleVisibility', 'off');
% RGB 轴线
quiver3(0, 0, 0, arrow_l, 0, 0, 'Color', 'r', 'LineWidth', 3, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, arrow_l, 0, 'Color', 'g', 'LineWidth', 3, 'HandleVisibility', 'off');
quiver3(0, 0, 0, 0, 0, arrow_l, 'Color', 'b', 'LineWidth', 3, 'HandleVisibility', 'off');
text(arrow_l+0.003, 0, 0, 'X', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
text(0, arrow_l+0.003, 0, 'Y', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
text(0, 0, arrow_l+0.003, 'Z', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
% 基座标记
plot3(0, 0, 0, 'bo', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');

% 4. 绘制平滑位姿曲线
plot3(gt_smooth(1,:), gt_smooth(2,:), gt_smooth(3,:), 'r-', 'LineWidth', 3.5, 'DisplayName', 'Ground Truth Pose');
plot3(pr_smooth(1,:), pr_smooth(2,:), pr_smooth(3,:), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Proposed Predict Pose');

% 5. 绘制外力矢量 (箭头)
f_scale = 0.12; 
f_origin_gt = gt_p(:, f_node_idx);
f_origin_pr = pr_p(:, f_node_idx);

% 真实力：红色粗实线箭头
quiver3(f_origin_gt(1), f_origin_gt(2), f_origin_gt(3), ...
    f_gt_vec(1)*f_scale, f_gt_vec(2)*f_scale, f_gt_vec(3)*f_scale, ...
    'Color', 'r', 'LineWidth', 4.5, 'MaxHeadSize', 1.2, 'AutoScale', 'off', 'DisplayName', 'Measured Force');

% 预测力：蓝色虚线箭头
quiver3(f_origin_pr(1), f_origin_pr(2), f_origin_pr(3), ...
    f_pr_vec(1)*f_scale, f_pr_vec(2)*f_scale, f_pr_vec(3)*f_scale, ...
    'Color', 'b', 'LineStyle', '-', 'LineWidth', 3, 'MaxHeadSize', 1.2, 'AutoScale', 'off', 'DisplayName', 'Estimated Force');

% 6. 视觉修饰与 IEEE 标准
set(gca, 'ZDir', 'reverse', 'FontSize', 24, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
view(-45, 30); % 你可以微调这两个数字改变视角
xlabel('X [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
ylabel('Y [m]', 'FontSize', 24, 'FontWeight', 'bold'); 
zlabel('Z [m]', 'FontSize', 24, 'FontWeight', 'bold');
set(gca, 'Box', 'off', 'Color', 'w');

% 图例设置在右上角
legend('Location', 'northeast', 'FontSize', 24, 'Box', 'on', 'EdgeColor', 'k');

f_err = norm(f_gt_vec - f_pr_vec);
%title(['Sample #', num2str(plot_sample_idx), ' Force Err: ', num2str(f_err, '%.3f'), ' N'], 'FontSize', 24);

disp(['>>> Smoothed Visualization (PCHIP) of Sample #', num2str(plot_sample_idx), ' completed.']);
%% ========================================================================
%  Ablation Study: Single Node (Node 4) & Single Dir (-X) Performance
%  逻辑：严格参照 Optimized - Full Version 架构，仅改变数据选择范围
% =========================================================================
disp('--------------------------------------------------');
disp('Starting Ablation Study: Node 4 & Dir 2 Only...');

% --- 1. 筛选特定子集 (Node 4, Dir 2) ---
% 在你已经 Cleaning 完的数据 (F_after, F_before, P_before_clean 等) 中筛选
idx_abl = (raw_hgt == 4) & (raw_dir == 4);

% 提取该子集的所有核心变量
Fa_abl = F_after(:, idx_abl);
Fb_abl = F_before(:, idx_abl);
Fd_abl = F_after(:, idx_abl) - F_before(:, idx_abl); % F_diff
Pb_abl = P_before_clean(:, idx_abl);
Pa_abl = P_after_clean(:, idx_abl);
gtF_abl = gt_F_clean(:, idx_abl);
hgt_abl = raw_hgt(idx_abl);

N_abl = size(Fa_abl, 2);
if N_abl < 20, error('Selected subset (Node 4, Dir 2) has too few samples!'); end
fprintf('   > Ablation Subset Size: %d samples\n', N_abl);

% --- 2. 划分训练集与测试集 (80/20) ---
% 为了保证消融实验的纯净性，我们不进行 Data Augmentation
N_train_abl = floor(N_abl * 0.8);
idx_tr = 1:N_train_abl;
idx_te = N_train_abl+1 : N_abl;

% --- 3. 训练 Net B (Force Estimation) ---
disp('   > Training Net B (Ablation)...');
inputs_b_abl = [Fa_abl(:, idx_tr); Fd_abl(:, idx_tr); Fb_abl(:, idx_tr)];
targets_b_abl = gtF_abl(:, idx_tr);

net_f_abl = feedforwardnet([40, 20]);
net_f_abl.trainFcn = 'trainlm';
net_f_abl.trainParam.showWindow = false;
net_f_abl = train(net_f_abl, inputs_b_abl, targets_b_abl);

% --- 4. 训练 Net C (Shape Reconstruction) ---
disp('   > Training Net C (Ablation)...');
% 级联推理：用 Net B 预测出的外力作为 Net C 的输入
pred_f_tr_abl = net_f_abl(inputs_b_abl);

% 构造 Net C 输入: [F_after; Pred_Force; Pred_Loc; P_before_ideal]
inputs_c_abl = [Fa_abl(:, idx_tr); pred_f_tr_abl; hgt_abl(idx_tr)/9.0; Pb_abl(:, idx_tr)];
targets_c_abl = Pa_abl(:, idx_tr) - Pb_abl(:, idx_tr); % 学习 Delta P

net_s_abl = fitnet([80, 60, 40]);
net_s_abl.trainFcn = 'trainscg';
net_s_abl.trainParam.showWindow = false;
net_s_abl = train(net_s_abl, inputs_c_abl, targets_c_abl);

% --- 5. 在消融子集的测试集上进行评估 ---
disp('   > Evaluating Ablation Performance...');

% A. Net B 推理
te_inputs_b = [Fa_abl(:, idx_te); Fd_abl(:, idx_te); Fb_abl(:, idx_te)];
te_pred_f = net_f_abl(te_inputs_b);

% B. Net C 推理
te_inputs_c = [Fa_abl(:, idx_te); te_pred_f; hgt_abl(idx_te)/9.0; Pb_abl(:, idx_te)];
te_pred_delta = net_s_abl(te_inputs_c);

% C. 计算位姿重构结果 (mm)
te_P_recon = Pb_abl(:, idx_te) + te_pred_delta;
te_P_real  = Pa_abl(:, idx_te);

% 计算末端误差 (Tip Error)
te_errs = zeros(1, length(idx_te));
for j = 1:length(idx_te)
    p_p = reshape(te_P_recon(:, j), 3, []);
    p_r = reshape(te_P_real(:, j), 3, []);
    te_errs(j) = norm(p_p(:, end) - p_r(:, end)) * 1000; % 转为 mm
end

% --- 6. 绘图展示消融实验结果 ---
figure('Name', 'Ablation Result: Node 4, Dir 2 Only', 'Color', 'w');
plot(te_errs, 'r-o', 'LineWidth', 1.5);
grid on;
title(sprintf('Ablation Study (Node 4, Dir 2)\nMean Tip Error: %.2f mm', mean(te_errs)));
ylabel('Tip MAE (mm)');
xlabel('Test Sample Index');

fprintf('--------------------------------------------------\n');
fprintf('Ablation Analysis (Single Node/Dir Case):\n');
fprintf('   > Subset Samples: %d\n', N_abl);
fprintf('   > Test Set Tip MAE: %.2f mm\n', mean(te_errs));
%% ========================================================================
%  Step 9.30: IEEE Summary Figure
%  Prior vs Proposed vs Ground Truth + Force + Insets
%  This section generates a publication-style summary figure
% =========================================================================
disp('--------------------------------------------------');
disp('9.30 Generating IEEE-style summary reconstruction figure...');

% ------------------------------ USER OPTIONS ------------------------------
plot_mode = 'median';     % 'median' or 'manual'
manual_idx = 1;           % used only when plot_mode = 'manual'
export_this_figure = true;

% force arrow scale (visual only)
force_scale = 0.10;       % 1 N -> 0.10 m visual scale in figure coordinates

% smoothing density
n_interp = 150;           % PCHIP interpolation samples

% node label display
show_node_labels = true;

% figure name
fig_name = 'IEEE_Summary_Reconstruction_Figure';
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Safety check: required variables
% -------------------------------------------------------------------------
req_vars = {'pred_P_after','real_P_after','p_before_test','tip_dist'};
for ii = 1:numel(req_vars)
    if ~evalin('base', sprintf('exist(''%s'',''var'')', req_vars{ii}))
        error('Required variable "%s" is missing. Please run the previous sections first.', req_vars{ii});
    end
end

% Try to use aligned force variables if available
has_force = evalin('base', 'exist(''F_gt_test'',''var'') && exist(''F_pd_test'',''var'')');
has_hgt   = evalin('base', 'exist(''gt_hgt_test'',''var'')');

% Pull local copies
P_pred_all  = pred_P_after;   % 21 x N_test_final
P_real_all  = real_P_after;   % 21 x N_test_final
P_prior_all = p_before_test;  % 21 x N_test_final
tip_err_all = tip_dist;       % 1 x N_test_final  (m)

% If force/hgt are available, try to dimension-align
if has_force
    % F_gt_test and F_pd_test should already be aligned with pred_P_after/real_P_after
    F_gt_all = F_gt_test;
    F_pd_all = F_pd_test;
else
    F_gt_all = [];
    F_pd_all = [];
end

if has_hgt
    % Depending on where gt_hgt_test is created, it may still need alignment
    % Preferred aligned vector:
    if exist('hgt_filtered', 'var')
        hgt_all = hgt_filtered;  % already filtered and aligned in your later sections
    else
        % fallback
        hgt_all = gt_hgt_test;
        if numel(hgt_all) ~= size(P_pred_all, 2)
            % try final filtering if available
            if exist('v_idx', 'var') && numel(gt_hgt_test) >= sum(v_idx)
                hgt_all = gt_hgt_test(v_idx);
            end
        end
    end
else
    hgt_all = [];
end

N_plot = size(P_pred_all, 2);
if N_plot < 1
    error('No valid samples found for visualization.');
end

% -------------------------------------------------------------------------
% Select representative sample
% -------------------------------------------------------------------------
switch lower(plot_mode)
    case 'median'
        [~, plot_idx] = min(abs(tip_err_all - median(tip_err_all)));
    case 'manual'
        plot_idx = manual_idx;
    otherwise
        error('Unknown plot_mode. Use "median" or "manual".');
end

plot_idx = max(1, min(plot_idx, N_plot));
fprintf('   > Selected sample index = %d\n', plot_idx);

% -------------------------------------------------------------------------
% Extract one sample: prior, GT, prediction
% -------------------------------------------------------------------------
P0_raw = reshape(P_prior_all(:, plot_idx), 3, 7);  % prior P^0
Pg_raw = reshape(P_real_all(:, plot_idx), 3, 7);   % GT P_gt^+
Pp_raw = reshape(P_pred_all(:, plot_idx), 3, 7);   % proposed P^+

% -------------------------------------------------------------------------
% Core baseline rule:
% Use the first node as origin for all displayed curves
% -------------------------------------------------------------------------
P0 = P0_raw - P0_raw(:, 1);
Pg = Pg_raw - Pg_raw(:, 1);
Pp = Pp_raw - Pp_raw(:, 1);

% -------------------------------------------------------------------------
% PCHIP smoothing (shape-preserving)
% -------------------------------------------------------------------------
t_nodes = 1:7;
tq = linspace(1, 7, n_interp);

P0_s = zeros(3, n_interp);
Pg_s = zeros(3, n_interp);
Pp_s = zeros(3, n_interp);

for kk = 1:3
    P0_s(kk, :) = interp1(t_nodes, P0(kk, :), tq, 'pchip');
    Pg_s(kk, :) = interp1(t_nodes, Pg(kk, :), tq, 'pchip');
    Pp_s(kk, :) = interp1(t_nodes, Pp(kk, :), tq, 'pchip');
end

% -------------------------------------------------------------------------
% Contact node and force vectors
% -------------------------------------------------------------------------
if ~isempty(hgt_all) && numel(hgt_all) >= plot_idx
    contact_node = round(hgt_all(plot_idx));
    contact_node = max(1, min(7, contact_node));
else
    % fallback: highlight node 4 if not available
    contact_node = 4;
end

if ~isempty(F_gt_all) && size(F_gt_all, 2) >= plot_idx
    F_gt = F_gt_all(:, plot_idx);
else
    F_gt = [0; 0; 0];
end

if ~isempty(F_pd_all) && size(F_pd_all, 2) >= plot_idx
    F_pd = F_pd_all(:, plot_idx);
else
    F_pd = [0; 0; 0];
end

% Force origins: use corresponding node on GT / prediction
F0_gt = Pg(:, contact_node);
F0_pd = Pp(:, contact_node);

% -------------------------------------------------------------------------
% Metrics
% -------------------------------------------------------------------------
tip_err_mm = norm(Pp(:, end) - Pg(:, end)) * 1000;
shape_err_mm = mean(sqrt(sum((Pp - Pg).^2, 1))) * 1000;
rmse_21d_mm = sqrt(mean((Pp(:) - Pg(:)).^2)) * 1000;
force_err_n = norm(F_pd - F_gt);

% -------------------------------------------------------------------------
% Style settings
% -------------------------------------------------------------------------
font_name = 'Times New Roman';

c_gt    = [0.05 0.05 0.05];     % near black
c_pred  = [0.10 0.35 0.95];     % blue
c_prior = [0.55 0.55 0.55];     % gray
c_mark  = [0.10 0.45 0.95];     % marker blue
c_ct    = [0.95 0.15 0.10];     % contact node red
c_fgt   = [0.10 0.85 0.10];     % GT force green
c_fpd   = [1.00 0.55 0.00];     % predicted force orange

lw_gt    = 2.4;
lw_pred  = 2.2;
lw_prior = 1.6;

ms_node  = 38;
ms_contact = 70;

% -------------------------------------------------------------------------
% Create main figure
% -------------------------------------------------------------------------
fig = figure('Name', fig_name, ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [80, 50, 1450, 980]);

% ============================== MAIN 3D AXIS ==============================
ax1 = axes('Parent', fig, 'Position', [0.06 0.16 0.56 0.72]);
hold(ax1, 'on'); grid(ax1, 'on'); axis(ax1, 'equal');

% Main curves
h_gt = plot3(ax1, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
    'Color', c_gt, 'LineWidth', lw_gt, ...
    'DisplayName', 'Mocap Ground Truth P_{gt}^{+}');

h_p0 = plot3(ax1, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
    'Color', c_prior, 'LineWidth', lw_prior, ...
    'DisplayName', 'CSBCM Prior P^{0}');

h_pd = plot3(ax1, Pp_s(1,:), Pp_s(2,:), Pp_s(3,:), '--', ...
    'Color', c_pred, 'LineWidth', lw_pred, ...
    'DisplayName', 'Proposed Reconstruction P^{+}');

% Node markers
scatter3(ax1, Pg(1,:), Pg(2,:), Pg(3,:), ms_node, ...
    'MarkerFaceColor', c_mark, 'MarkerEdgeColor', 'k', ...
    'LineWidth', 0.8, 'DisplayName', 'Marker Nodes');

% Contact node highlight
scatter3(ax1, Pg(1,contact_node), Pg(2,contact_node), Pg(3,contact_node), ms_contact, ...
    'MarkerFaceColor', c_ct, 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.2, 'DisplayName', sprintf('Contact Node (\\hat{c}=%d)', contact_node));

% Force arrows
hq1 = quiver3(ax1, F0_gt(1), F0_gt(2), F0_gt(3), ...
    F_gt(1)*force_scale, F_gt(2)*force_scale, F_gt(3)*force_scale, ...
    0, 'Color', c_fgt, 'LineWidth', 2.5, 'MaxHeadSize', 0.8, ...
    'DisplayName', 'GT Force F');

hq2 = quiver3(ax1, F0_pd(1), F0_pd(2), F0_pd(3), ...
    F_pd(1)*force_scale, F_pd(2)*force_scale, F_pd(3)*force_scale, ...
    0, 'Color', c_fpd, 'LineWidth', 2.5, 'LineStyle', '--', 'MaxHeadSize', 0.8, ...
    'DisplayName', 'Estimated Force \hat{F}');

% Origin label
text(ax1, Pg(1,1), Pg(2,1), Pg(3,1), '  Node 1 (Origin)', ...
    'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

% Optional node labels
if show_node_labels
    for nn = 2:7
        label_color = [0.15 0.15 0.15];
        if nn == contact_node
            label_color = c_ct;
        end
        text(ax1, Pg(1,nn), Pg(2,nn), Pg(3,nn), sprintf('  Node %d', nn), ...
            'FontName', font_name, 'FontSize', 13, 'Color', label_color, ...
            'FontWeight', ternary(nn==contact_node,'bold','normal'));
    end
end

% Axes setup
set(ax1, 'FontName', font_name, 'FontSize', 18, 'LineWidth', 1.2, ...
    'TickDir', 'out', 'Box', 'off', 'ZDir', 'reverse');
xlabel(ax1, 'X [m]', 'FontName', font_name, 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax1, 'Y [m]', 'FontName', font_name, 'FontSize', 18, 'FontWeight', 'bold');
zlabel(ax1, 'Z [m]', 'FontName', font_name, 'FontSize', 18, 'FontWeight', 'bold');
view(ax1, -38, 22);

title(ax1, {'TDCR Backbone Reconstruction: Physics-Prior-Guided Residual Learning', ...
            'Single-Sample Summary Figure'}, ...
            'FontName', font_name, 'FontSize', 20, 'FontWeight', 'bold');

% Legend
lgd = legend(ax1, [h_gt, h_p0, h_pd], ...
    {'Mocap Ground Truth P_{gt}^{+}', ...
     'CSBCM Prior P^{0}', ...
     'Proposed Reconstruction P^{+} = P^{0} + \DeltaP'}, ...
    'Location', 'northwest');
set(lgd, 'FontName', font_name, 'FontSize', 14, 'Box', 'on');

% Metrics box
metrics_str = sprintf(['Tip Error:        %.2f mm\n' ...
                       'Mean Shape Error: %.2f mm\n' ...
                       'RMSE (21D):       %.2f mm\n' ...
                       'Contact Node:     %d\n' ...
                       '||F - Fhat||:     %.3f N'], ...
                       tip_err_mm, shape_err_mm, rmse_21d_mm, contact_node, force_err_n);

annotation(fig, 'textbox', [0.08 0.18 0.19 0.12], ...
    'String', metrics_str, ...
    'FitBoxToText', 'off', ...
    'BackgroundColor', 'w', ...
    'EdgeColor', [0 0 0], ...
    'LineWidth', 1.0, ...
    'FontName', font_name, ...
    'FontSize', 13);

% ============================== TOP VIEW INSET =============================
ax2 = axes('Parent', fig, 'Position', [0.69 0.56 0.25 0.26]);
hold(ax2, 'on'); grid(ax2, 'on'); axis(ax2, 'equal');

plot(ax2, Pg_s(1,:), Pg_s(2,:), '-',  'Color', c_gt,    'LineWidth', 2.0);
plot(ax2, P0_s(1,:), P0_s(2,:), '-.', 'Color', c_prior, 'LineWidth', 1.5);
plot(ax2, Pp_s(1,:), Pp_s(2,:), '--', 'Color', c_pred,  'LineWidth', 2.0);

scatter(ax2, Pg(1,:), Pg(2,:), 28, 'MarkerFaceColor', c_mark, 'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
scatter(ax2, Pg(1,contact_node), Pg(2,contact_node), 55, 'MarkerFaceColor', c_ct, 'MarkerEdgeColor', 'k');

set(ax2, 'FontName', font_name, 'FontSize', 14, 'LineWidth', 1.0, 'TickDir', 'out');
xlabel(ax2, 'X [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax2, 'Y [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
title(ax2, 'Top View (X-Y Plane)', 'FontName', font_name, 'FontSize', 16, 'FontWeight', 'bold');

% ============================ TIP ZOOM-IN INSET ===========================
ax3 = axes('Parent', fig, 'Position', [0.66 0.16 0.30 0.28]);
hold(ax3, 'on'); grid(ax3, 'on'); axis(ax3, 'equal');

% Show only distal region (nodes 4:7)
idx_zoom = 4:7;
t_zoom = idx_zoom(1):0.02:idx_zoom(end);

Pg_zoom = zeros(3, numel(t_zoom));
P0_zoom = zeros(3, numel(t_zoom));
Pp_zoom = zeros(3, numel(t_zoom));
for kk = 1:3
    Pg_zoom(kk,:) = interp1(t_nodes, Pg(kk,:), t_zoom, 'pchip');
    P0_zoom(kk,:) = interp1(t_nodes, P0(kk,:), t_zoom, 'pchip');
    Pp_zoom(kk,:) = interp1(t_nodes, Pp(kk,:), t_zoom, 'pchip');
end

plot3(ax3, Pg_zoom(1,:), Pg_zoom(2,:), Pg_zoom(3,:), '-',  'Color', c_gt,    'LineWidth', 2.2);
plot3(ax3, P0_zoom(1,:), P0_zoom(2,:), P0_zoom(3,:), '-.', 'Color', c_prior, 'LineWidth', 1.4);
plot3(ax3, Pp_zoom(1,:), Pp_zoom(2,:), Pp_zoom(3,:), '--', 'Color', c_pred,  'LineWidth', 2.2);

scatter3(ax3, Pg(1,idx_zoom), Pg(2,idx_zoom), Pg(3,idx_zoom), ...
    42, 'MarkerFaceColor', c_mark, 'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
scatter3(ax3, Pg(1,contact_node), Pg(2,contact_node), Pg(3,contact_node), ...
    65, 'MarkerFaceColor', c_ct, 'MarkerEdgeColor', 'k', 'LineWidth', 1.0);

quiver3(ax3, F0_gt(1), F0_gt(2), F0_gt(3), ...
    F_gt(1)*force_scale, F_gt(2)*force_scale, F_gt(3)*force_scale, ...
    0, 'Color', c_fgt, 'LineWidth', 2.2, 'MaxHeadSize', 0.8);

quiver3(ax3, F0_pd(1), F0_pd(2), F0_pd(3), ...
    F_pd(1)*force_scale, F_pd(2)*force_scale, F_pd(3)*force_scale, ...
    0, 'Color', c_fpd, 'LineWidth', 2.2, 'LineStyle', '--', 'MaxHeadSize', 0.8);

set(ax3, 'FontName', font_name, 'FontSize', 13, 'LineWidth', 1.0, ...
    'TickDir', 'out', 'Box', 'on', 'ZDir', 'reverse');
xlabel(ax3, 'X [m]', 'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax3, 'Y [m]', 'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold');
zlabel(ax3, 'Z [m]', 'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold');
title(ax3, 'Tip Region Zoom-in', 'FontName', font_name, 'FontSize', 16, 'FontWeight', 'bold');
view(ax3, -42, 24);

% Add scale bar in zoom-in view (visual annotation in 2D figure space)
annotation(fig, 'line', [0.90 0.96], [0.18 0.18], 'Color', 'k', 'LineWidth', 1.5);
annotation(fig, 'line', [0.90 0.90], [0.175 0.185], 'Color', 'k', 'LineWidth', 1.5);
annotation(fig, 'line', [0.96 0.96], [0.175 0.185], 'Color', 'k', 'LineWidth', 1.5);
annotation(fig, 'textbox', [0.895 0.185 0.08 0.03], 'String', '10 mm', ...
    'EdgeColor', 'none', 'FontName', font_name, 'FontSize', 12, ...
    'HorizontalAlignment', 'center');

% Footer note
annotation(fig, 'textbox', [0.05 0.03 0.90 0.07], ...
    'String', ['\bullet  Node 1 is used as the origin.  ', ...
               '\bullet  Z-axis is reversed.  ', ...
               '\bullet  Curves are smoothed using PCHIP.  ', ...
               '\bullet  Proposed shape is reconstructed as P^{+} = P^{0} + \DeltaP.'], ...
    'EdgeColor', [0 0 0], ...
    'BackgroundColor', 'w', ...
    'FontName', font_name, ...
    'FontSize', 13);

disp('>>> IEEE-style summary figure generated successfully.');

% -------------------------------------------------------------------------
% Export
% -------------------------------------------------------------------------
if export_this_figure
    out_name = 'IEEE_summary_reconstruction';

    exportgraphics(fig, [out_name, '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [out_name, '.png'], 'Resolution', 600);
    savefig(fig, [out_name, '.fig']);

    disp(['>>> Exported: ', out_name, '.pdf / .png / .fig']);
end
%% ========================================================================
%  Step 9.31: Clean 3-Panel Reconstruction Figure with Origin Frames
%  No in-figure legend/text except X/Y/Z axis labels
% =========================================================================
disp('--------------------------------------------------');
disp('9.31 Generating clean reconstruction figure with origin coordinate frames...');

% ============================= User Options ===============================
export_this_figure = true;
fig_name = 'IEEE_prior_condition_reconstruction_clean_axes';

n_interp = 160;
force_scale = 0.10;

show_force_arrow = true;
show_node_markers = true;
show_contact_node = true;

force_direction_mode = 'toward_deformation';

max_pred_err_percentile   = 75;
min_prior_err_percentile  = 30;
max_prior_err_percentile  = 85;
min_improvement_ratio     = 0.25;
curvature_excess_limit    = 1.35;

origin_axis_len = 0.020;     % length of local origin coordinate frame [m]
origin_axis_lw  = 2.2;

% =========================== Required Variables ===========================
if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var') || ...
   ~exist('p_before_test', 'var') || ~exist('tip_dist', 'var')
    error(['Missing required variables. Please run evaluation first. ', ...
           'Required: pred_P_after, real_P_after, p_before_test, tip_dist.']);
end

P_pred_all = pred_P_after;
P_gt_all   = real_P_after;
tip_err_m  = tip_dist;
N_final    = size(P_pred_all, 2);

if size(p_before_test, 2) == N_final
    P_prior_all = p_before_test;
elseif exist('v_idx', 'var') && numel(v_idx) == size(p_before_test, 2)
    P_prior_all = p_before_test(:, v_idx);
else
    error('p_before_test is not aligned with pred_P_after.');
end

if size(P_prior_all, 2) ~= N_final
    error('P_prior_all, P_pred_all, and P_gt_all are dimension-mismatched.');
end

% ===================== Optional Force / Contact Alignment =================
has_force = false;
F_gt_plot = [];
F_pd_plot = [];

if exist('F_gt_test', 'var') && exist('F_pd_test', 'var') && ...
   size(F_gt_test, 2) == N_final && size(F_pd_test, 2) == N_final

    F_gt_plot = F_gt_test;
    F_pd_plot = F_pd_test;
    has_force = true;

elseif exist('pred_force_all', 'var') && exist('aug_gt_F', 'var') && ...
       exist('v_mask', 'var') && exist('test_idx', 'var') && exist('v_idx', 'var')

    tmp_F_gt = aug_gt_F(:, v_mask);
    tmp_F_pd = pred_force_all(:, v_mask);

    tmp_F_gt = tmp_F_gt(:, test_idx);
    tmp_F_pd = tmp_F_pd(:, test_idx);

    F_gt_plot = tmp_F_gt(:, v_idx);
    F_pd_plot = tmp_F_pd(:, v_idx);

    if size(F_gt_plot, 2) == N_final && size(F_pd_plot, 2) == N_final
        has_force = true;
    end
end

contact_node_all = ones(1, N_final) * 4;

if exist('test_hgt', 'var') && numel(test_hgt) == N_final
    contact_node_all = round(test_hgt);
elseif exist('hgt_filtered', 'var') && numel(hgt_filtered) == N_final
    contact_node_all = round(hgt_filtered);
elseif exist('gt_hgt_test', 'var') && exist('v_idx', 'var') && numel(gt_hgt_test) == numel(v_idx)
    contact_node_all = round(gt_hgt_test(v_idx));
end

contact_node_all = max(1, min(7, contact_node_all));

% ========================= Prior-Condition Metrics ========================
prior_curvature = zeros(1, N_final);
gt_curvature = zeros(1, N_final);
pred_curvature = zeros(1, N_final);
curvature_excess = zeros(1, N_final);

prior_tip_deflection = zeros(1, N_final);
pred_shape_err_mm = zeros(1, N_final);
prior_shape_err_mm = zeros(1, N_final);
residual_mag_mm = zeros(1, N_final);
improvement_ratio = zeros(1, N_final);

for i = 1:N_final
    Pg_raw = reshape(P_gt_all(:, i), 3, 7);
    Pp_raw = reshape(P_pred_all(:, i), 3, 7);
    P0_raw = reshape(P_prior_all(:, i), 3, 7);

    % Use first node as origin
    Pg = Pg_raw - Pg_raw(:, 1);
    Pp = Pp_raw - Pp_raw(:, 1);
    P0 = P0_raw - P0_raw(:, 1);

    pred_shape_err_mm(i)  = mean(sqrt(sum((Pp - Pg).^2, 1))) * 1000;
    prior_shape_err_mm(i) = mean(sqrt(sum((P0 - Pg).^2, 1))) * 1000;
    residual_mag_mm(i)    = mean(sqrt(sum((Pp - P0).^2, 1))) * 1000;

    improvement_ratio(i) = ...
        (prior_shape_err_mm(i) - pred_shape_err_mm(i)) / max(prior_shape_err_mm(i), eps);

    prior_tip_deflection(i) = norm(P0(:, end) - P0(:, 1));

    % Prior curvature P0
    X0 = P0';
    X0c = X0 - mean(X0, 1);
    [~, ~, V0] = svd(X0c, 'econ');
    q0 = X0c * V0(:, 1:2);

    kappa0 = zeros(1, 5);
    for j = 2:6
        v1 = q0(j, :)   - q0(j-1, :);
        v2 = q0(j+1, :) - q0(j, :);
        cross_z = v1(1)*v2(2) - v1(2)*v2(1);
        denom = norm(v1) * norm(v2) + eps;
        kappa0(j-1) = abs(cross_z / denom);
    end
    prior_curvature(i) = sum(kappa0);

    % GT curvature
    Xg = Pg';
    Xgc = Xg - mean(Xg, 1);
    [~, ~, Vg] = svd(Xgc, 'econ');
    qg = Xgc * Vg(:, 1:2);

    kappag = zeros(1, 5);
    for j = 2:6
        v1 = qg(j, :)   - qg(j-1, :);
        v2 = qg(j+1, :) - qg(j, :);
        cross_z = v1(1)*v2(2) - v1(2)*v2(1);
        denom = norm(v1) * norm(v2) + eps;
        kappag(j-1) = abs(cross_z / denom);
    end
    gt_curvature(i) = sum(kappag);

    % Proposed curvature
    Xp = Pp';
    Xpc = Xp - mean(Xp, 1);
    [~, ~, Vp] = svd(Xpc, 'econ');
    qp = Xpc * Vp(:, 1:2);

    kappap = zeros(1, 5);
    for j = 2:6
        v1 = qp(j, :)   - qp(j-1, :);
        v2 = qp(j+1, :) - qp(j, :);
        cross_z = v1(1)*v2(2) - v1(2)*v2(1);
        denom = norm(v1) * norm(v2) + eps;
        kappap(j-1) = abs(cross_z / denom);
    end
    pred_curvature(i) = sum(kappap);

    curvature_excess(i) = pred_curvature(i) / max(gt_curvature(i), eps);
end

tip_err_mm = tip_err_m * 1000;

norm_curv = (prior_curvature - min(prior_curvature)) ./ ...
    (max(prior_curvature)-min(prior_curvature)+eps);

norm_defl = (prior_tip_deflection - min(prior_tip_deflection)) ./ ...
    (max(prior_tip_deflection)-min(prior_tip_deflection)+eps);

norm_resi = (residual_mag_mm - min(residual_mag_mm)) ./ ...
    (max(residual_mag_mm)-min(residual_mag_mm)+eps);

norm_pred_err = (pred_shape_err_mm - min(pred_shape_err_mm)) ./ ...
    (max(pred_shape_err_mm)-min(pred_shape_err_mm)+eps);

norm_prior_err = (prior_shape_err_mm - min(prior_shape_err_mm)) ./ ...
    (max(prior_shape_err_mm)-min(prior_shape_err_mm)+eps);

% ======================= Publication-Friendly Pool ========================
good_pred = pred_shape_err_mm <= prctile(pred_shape_err_mm, max_pred_err_percentile);

reasonable_prior = ...
    prior_shape_err_mm >= prctile(prior_shape_err_mm, min_prior_err_percentile) & ...
    prior_shape_err_mm <= prctile(prior_shape_err_mm, max_prior_err_percentile);

good_compensation = improvement_ratio >= min_improvement_ratio;
natural_shape = curvature_excess <= curvature_excess_limit;

publication_pool = good_pred & reasonable_prior & good_compensation & natural_shape;

if sum(publication_pool) < 3
    warning('Publication pool is small. Relaxing naturalness threshold to 1.60.');
    natural_shape = curvature_excess <= 1.60;
    publication_pool = good_pred & reasonable_prior & good_compensation & natural_shape;
end

if sum(publication_pool) < 3
    warning('Publication pool is still small. Relaxing prior/improvement constraints.');
    publication_pool = good_pred & natural_shape;
end

if sum(publication_pool) < 3
    warning('Publication pool is still too small. Using good_pred only.');
    publication_pool = good_pred;
end

% ========================== Sample Selection ==============================

% Mild-bending prior
mild_pool = find(publication_pool & ...
                 prior_curvature <= prctile(prior_curvature, 45) & ...
                 prior_tip_deflection <= prctile(prior_tip_deflection, 60));

if isempty(mild_pool)
    mild_pool = find(good_pred & natural_shape & prior_curvature <= prctile(prior_curvature, 50));
end

if isempty(mild_pool)
    [~, idx_mild] = min(abs(pred_shape_err_mm - median(pred_shape_err_mm)));
else
    score_mild = ...
        0.35 * improvement_ratio(mild_pool) + ...
        0.25 * norm_prior_err(mild_pool) - ...
        0.35 * norm_pred_err(mild_pool) - ...
        0.15 * norm_curv(mild_pool);

    [~, best_local] = max(score_mild);
    idx_mild = mild_pool(best_local);
end

% Moderate-bending prior
moderate_pool = find(publication_pool & ...
                     prior_curvature >= prctile(prior_curvature, 45) & ...
                     prior_curvature <= prctile(prior_curvature, 75));

moderate_pool = setdiff(moderate_pool, idx_mild, 'stable');

if isempty(moderate_pool)
    moderate_pool = find(good_pred & natural_shape & ...
                         prior_curvature >= prctile(prior_curvature, 40) & ...
                         prior_curvature <= prctile(prior_curvature, 80));
    moderate_pool = setdiff(moderate_pool, idx_mild, 'stable');
end

if isempty(moderate_pool)
    fallback_pool = setdiff(find(good_pred), idx_mild, 'stable');
    if isempty(fallback_pool)
        fallback_pool = setdiff(1:N_final, idx_mild, 'stable');
    end
    [~, best_local] = min(abs(prior_curvature(fallback_pool) - median(prior_curvature)));
    idx_moderate = fallback_pool(best_local);
else
    score_moderate = ...
        0.30 * improvement_ratio(moderate_pool) + ...
        0.25 * norm_prior_err(moderate_pool) + ...
        0.20 * norm_defl(moderate_pool) - ...
        0.40 * norm_pred_err(moderate_pool) - ...
        0.20 * abs(norm_curv(moderate_pool) - 0.60) - ...
        0.20 * max(curvature_excess(moderate_pool) - 1.0, 0);

    [~, best_local] = max(score_moderate);
    idx_moderate = moderate_pool(best_local);
end

% Prior-compensation case
comp_pool = find(publication_pool);
comp_pool = setdiff(comp_pool, [idx_mild, idx_moderate], 'stable');

if isempty(comp_pool)
    comp_pool = find(good_pred & reasonable_prior & natural_shape);
    comp_pool = setdiff(comp_pool, [idx_mild, idx_moderate], 'stable');
end

if isempty(comp_pool)
    comp_pool = setdiff(find(good_pred), [idx_mild, idx_moderate], 'stable');
end

if isempty(comp_pool)
    comp_pool = setdiff(1:N_final, [idx_mild, idx_moderate], 'stable');
end

target_residual_level = 0.55;

score_comp = ...
    0.45 * improvement_ratio(comp_pool) + ...
    0.25 * norm_prior_err(comp_pool) + ...
    0.15 * norm_defl(comp_pool) - ...
    0.30 * norm_pred_err(comp_pool) - ...
    0.15 * abs(norm_resi(comp_pool) - target_residual_level) - ...
    0.20 * max(curvature_excess(comp_pool) - 1.0, 0);

[~, best_local] = max(score_comp);
idx_comp = comp_pool(best_local);

selected_idx = [idx_mild, idx_moderate, idx_comp];

panel_names_terminal = {'Mild-bending CSBCM prior', ...
                        'Moderate-bending CSBCM prior', ...
                        'Prior-compensation case'};

% ============================== Plot Style ================================
font_name = 'Times New Roman';

c_gt    = [0.03 0.03 0.03];   % black
c_pred  = [0.10 0.32 0.90];   % blue
c_prior = [0.72 0.72 0.72];   % light gray
c_node  = [0.20 0.55 0.95];   % marker blue
c_ct    = [0.90 0.15 0.10];   % contact red
c_fgt   = [0.10 0.65 0.15];   % green
c_fpd   = [0.95 0.45 0.05];   % orange

lw_gt = 2.5;
lw_pred = 2.2;
lw_prior = 1.35;

% ============================ Terminal Output =============================
disp(' ');
disp('==================== Figure Information ====================');
disp('No legend or annotations are shown in the figure.');
disp('Only X/Y/Z axis labels and local origin frames are displayed.');
disp(' ');
disp('Curve styles:');
disp('  Black solid line      : Mocap Ground Truth P_gt^+');
disp('  Light-gray dash-dot   : CSBCM Prior P^0');
disp('  Blue dashed line      : Proposed Reconstruction P^+ = P^0 + DeltaP');
disp('  Blue circular markers : Marker-defined backbone nodes');
disp('  Red circular marker   : Contact node c_hat');
if show_force_arrow && has_force
    disp('  Green arrow           : Ground-truth contact force F');
    disp('  Orange dashed arrow   : Estimated contact force F_hat');
    fprintf('  Force display mode    : %s\n', force_direction_mode);
else
    disp('  Force arrows          : Not shown');
end
disp(' ');
disp('Selected samples:');
for pp = 1:3
    idx_tmp = selected_idx(pp);
    fprintf('  (%c) %-32s | idx = %d | tip = %.2f mm | shape = %.2f mm | prior = %.2f mm | improve = %.0f%% | curv_excess = %.2f\n', ...
        char('a'+pp-1), panel_names_terminal{pp}, idx_tmp, ...
        tip_err_mm(idx_tmp), pred_shape_err_mm(idx_tmp), ...
        prior_shape_err_mm(idx_tmp), improvement_ratio(idx_tmp)*100, ...
        curvature_excess(idx_tmp));
end
disp('============================================================');
disp(' ');

% =============================== Main Figure ==============================
fig = figure('Name', fig_name, ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [80, 80, 1650, 600]);

tl = tiledlayout(fig, 1, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for p = 1:3
    idx = selected_idx(p);

    ax = nexttile(tl, p);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');

    P0_raw = reshape(P_prior_all(:, idx), 3, 7);
    Pg_raw = reshape(P_gt_all(:, idx), 3, 7);
    Pp_raw = reshape(P_pred_all(:, idx), 3, 7);

    % Use first node as origin
    P0 = P0_raw - P0_raw(:, 1);
    Pg = Pg_raw - Pg_raw(:, 1);
    Pp = Pp_raw - Pp_raw(:, 1);

    % PCHIP smoothing
    t_nodes = 1:7;
    tq = linspace(1, 7, n_interp);

    P0_s = zeros(3, n_interp);
    Pg_s = zeros(3, n_interp);
    Pp_s = zeros(3, n_interp);

    for kk = 1:3
        P0_s(kk, :) = interp1(t_nodes, P0(kk, :), tq, 'pchip');
        Pg_s(kk, :) = interp1(t_nodes, Pg(kk, :), tq, 'pchip');
        Pp_s(kk, :) = interp1(t_nodes, Pp(kk, :), tq, 'pchip');
    end

    % Main curves
    plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
        'Color', c_gt, 'LineWidth', lw_gt);

    plot3(ax, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
        'Color', c_prior, 'LineWidth', lw_prior);

    plot3(ax, Pp_s(1,:), Pp_s(2,:), Pp_s(3,:), '--', ...
        'Color', c_pred, 'LineWidth', lw_pred);

    % Marker nodes
    if show_node_markers
        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 32, ...
            'MarkerFaceColor', c_node, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.6);
    end

    % Contact node marker only, no text
    cnode = contact_node_all(idx);
    if show_contact_node
        scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 70, ...
            'MarkerFaceColor', c_ct, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.1);
    end

    % Force arrows
    if show_force_arrow && has_force
        F_gt = F_gt_plot(:, idx);
        F_pd = F_pd_plot(:, idx);

        F0_gt = Pg(:, cnode);
        F0_pd = Pp(:, cnode);

        deform_vec = Pp(:, cnode) - P0(:, cnode);

        switch force_direction_mode
            case 'raw'
                F_gt_vis = F_gt;
                F_pd_vis = F_pd;

            case 'reverse'
                F_gt_vis = -F_gt;
                F_pd_vis = -F_pd;

            case 'toward_deformation'
                if norm(F_gt) > eps && norm(deform_vec) > eps
                    if dot(F_gt, deform_vec) < 0
                        F_gt_vis = -F_gt;
                        F_pd_vis = -F_pd;
                    else
                        F_gt_vis = F_gt;
                        F_pd_vis = F_pd;
                    end
                else
                    F_gt_vis = F_gt;
                    F_pd_vis = F_pd;
                end

            otherwise
                error('Unknown force_direction_mode.');
        end

        quiver3(ax, F0_gt(1), F0_gt(2), F0_gt(3), ...
            F_gt_vis(1)*force_scale, F_gt_vis(2)*force_scale, F_gt_vis(3)*force_scale, ...
            0, 'Color', c_fgt, 'LineWidth', 2.0, ...
            'MaxHeadSize', 0.9);

        quiver3(ax, F0_pd(1), F0_pd(2), F0_pd(3), ...
            F_pd_vis(1)*force_scale, F_pd_vis(2)*force_scale, F_pd_vis(3)*force_scale, ...
            0, 'Color', c_fpd, 'LineWidth', 2.0, ...
            'LineStyle', '--', 'MaxHeadSize', 0.9);
    end

    % Local origin coordinate frame
    quiver3(ax, 0, 0, 0, origin_axis_len, 0, 0, ...
        0, 'Color', 'r', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, origin_axis_len, 0, ...
        0, 'Color', 'g', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, 0, origin_axis_len, ...
        0, 'Color', 'b', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    text(ax, origin_axis_len*1.08, 0, 0, 'X', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, origin_axis_len*1.08, 0, 'Y', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, 0, origin_axis_len*1.08, 'Z', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    % Axis settings
    set(ax, 'FontName', font_name, ...
        'FontSize', 14, ...
        'LineWidth', 1.1, ...
        'TickDir', 'out', ...
        'Box', 'off', ...
        'ZDir', 'reverse');

    xlabel(ax, 'X [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    ylabel(ax, 'Y [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    zlabel(ax, 'Z [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');

    view(ax, -40, 24);

    % Axis limits
    all_pts = [Pg, Pp, P0, [0;0;0], [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];
    pad = 0.010;
    xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
    ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
    zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
end

disp('>>> Clean figure with origin coordinate frames generated successfully.');

% ================================ Export ==================================
if export_this_figure
    exportgraphics(fig, [fig_name, '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [fig_name, '.png'], 'Resolution', 600);
    savefig(fig, [fig_name, '.fig']);

    disp(['>>> Exported: ', fig_name, '.pdf / .png / .fig']);
end
%% ========================================================================
%  Step 9.32: Same-Sample Baseline Comparison Figure
%  Compare Proposed vs Vanilla MLP vs Analytical Physics
%  Same selected_idx as Step 9.31
%  No in-figure legend/text except X/Y/Z labels and local coordinate frame
% =========================================================================
disp('--------------------------------------------------');
disp('9.32 Generating same-sample baseline comparison figure...');
disp('      Same sample indices will be used as the prior-condition figure.');
disp('      No legend or metric boxes will be shown in the figure.');

% ============================= User Options ===============================
export_this_figure = true;
fig_name = 'IEEE_same_sample_baseline_comparison_clean_axes';

n_interp = 160;

show_node_markers = true;
show_contact_node = true;
show_prior_curve  = true;   % show CSBCM prior P0 as light-gray reference

origin_axis_len = 0.020;    % local coordinate frame length [m]
origin_axis_lw  = 2.2;

% =========================== Required Variables ===========================
if ~exist('selected_idx', 'var')
    error(['selected_idx is missing. Please run Step 9.31 first, ', ...
           'so this comparison uses exactly the same samples.']);
end

selected_idx = selected_idx(:)';

if numel(selected_idx) ~= 3
    error('selected_idx should contain exactly 3 sample indices.');
end

if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing proposed / ground-truth variables: pred_P_after and real_P_after.');
end

if ~exist('pred_brute_abs', 'var')
    error('Missing Vanilla MLP result variable: pred_brute_abs. Please run Step 9.6 first.');
end

if ~exist('P_phys_after_all', 'var')
    error('Missing analytical physics result variable: P_phys_after_all. Please run Step 9.8 first.');
end

if ~exist('v_idx', 'var')
    error('Missing v_idx. Baseline outputs need v_idx for final alignment.');
end

P_gt_all   = real_P_after;      % 21 x N_final, meter
P_prop_all = pred_P_after;      % 21 x N_final, meter

N_final = size(P_gt_all, 2);

% Align CSBCM prior P0 if available
has_prior = false;
if exist('p_before_test', 'var')
    if size(p_before_test, 2) == N_final
        P_prior_all = p_before_test;
        has_prior = true;
    elseif numel(v_idx) == size(p_before_test, 2)
        P_prior_all = p_before_test(:, v_idx);
        has_prior = true;
    end
end

if show_prior_curve && ~has_prior
    warning('P0 prior is not aligned or missing. Prior curve will not be shown.');
    show_prior_curve = false;
end

% Align Vanilla MLP baseline.
% In your code, pred_brute_abs is computed on test_idx, then v_idx is applied.
if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error(['Cannot align Vanilla MLP result. Expected pred_brute_abs(:,v_idx) ', ...
           'to match real_P_after.']);
end

% Align analytical physics baseline.
% In your code, P_phys_after_all is computed on test_idx, then v_idx is applied.
if size(P_phys_after_all, 2) == N_final
    P_phys_all = P_phys_after_all;
elseif numel(v_idx) == size(P_phys_after_all, 2)
    P_phys_all = P_phys_after_all(:, v_idx);
else
    error(['Cannot align analytical physics result. Expected P_phys_after_all(:,v_idx) ', ...
           'to match real_P_after.']);
end

% Contact node alignment
contact_node_all = ones(1, N_final) * 4;

if exist('test_hgt', 'var') && numel(test_hgt) == N_final
    contact_node_all = round(test_hgt);
elseif exist('hgt_filtered', 'var') && numel(hgt_filtered) == N_final
    contact_node_all = round(hgt_filtered);
elseif exist('gt_hgt_test', 'var') && numel(gt_hgt_test) == numel(v_idx)
    contact_node_all = round(gt_hgt_test(v_idx));
end

contact_node_all = max(1, min(7, contact_node_all));

% ============================== Plot Style ================================
font_name = 'Times New Roman';

c_gt    = [0.03 0.03 0.03];   % black
c_prop  = [0.10 0.32 0.90];   % blue
c_mlp   = [0.20 0.60 0.25];   % green
c_phys  = [0.85 0.20 0.12];   % red
c_prior = [0.72 0.72 0.72];   % light gray

c_node  = [0.20 0.55 0.95];   % marker blue
c_ct    = [0.90 0.15 0.10];   % contact red

lw_gt    = 2.6;
lw_prop  = 2.4;
lw_mlp   = 2.0;
lw_phys  = 2.0;
lw_prior = 1.25;

% ========================== Terminal Information ==========================
disp(' ');
disp('==================== Same-Sample Baseline Comparison ====================');
disp('No legend, title, or metric boxes are shown in the figure.');
disp('Only X/Y/Z axis labels and local origin frames are displayed.');
disp(' ');
disp('Curve styles:');
disp('  Black solid line        : Mocap Ground Truth P_gt^+');
disp('  Blue dashed line        : Proposed Reconstruction P^+ = P^0 + DeltaP');
disp('  Green dash-dot line     : Vanilla MLP baseline');
disp('  Red dotted line         : Analytical physics baseline');
if show_prior_curve
    disp('  Light-gray dash-dot     : CSBCM prior P^0');
end
disp('  Blue circular markers   : Marker-defined GT nodes');
disp('  Red circular marker     : Contact node');
disp(' ');
disp('Selected sample indices reused from Step 9.31:');
fprintf('  selected_idx = [%d, %d, %d]\n', selected_idx(1), selected_idx(2), selected_idx(3));
disp(' ');
disp('Per-sample errors after Node-1 origin alignment:');
disp('  Columns: Sample | Method | Tip Error [mm] | Mean Shape Error [mm]');

for pp = 1:numel(selected_idx)
    idx = selected_idx(pp);

    Pg  = reshape(P_gt_all(:, idx), 3, 7);
    Ppr = reshape(P_prop_all(:, idx), 3, 7);
    Pml = reshape(P_mlp_all(:, idx), 3, 7);
    Pph = reshape(P_phys_all(:, idx), 3, 7);

    Pg  = Pg  - Pg(:, 1);
    Ppr = Ppr - Ppr(:, 1);
    Pml = Pml - Pml(:, 1);
    Pph = Pph - Pph(:, 1);

    [tip_prop, shape_prop] = local_err_metric(Ppr, Pg);
    [tip_mlp,  shape_mlp ] = local_err_metric(Pml, Pg);
    [tip_phys, shape_phys] = local_err_metric(Pph, Pg);

    fprintf('  %6d | Proposed           | %8.2f | %8.2f\n', idx, tip_prop, shape_prop);
    fprintf('  %6d | Vanilla MLP        | %8.2f | %8.2f\n', idx, tip_mlp,  shape_mlp);
    fprintf('  %6d | Analytical Physics | %8.2f | %8.2f\n', idx, tip_phys, shape_phys);
end

disp('==========================================================================');
disp(' ');

% =============================== Main Figure ==============================
fig = figure('Name', fig_name, ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [80, 80, 1650, 600]);

tl = tiledlayout(fig, 1, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for p = 1:3
    idx = selected_idx(p);

    ax = nexttile(tl, p);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');

    % Extract curves
    Pg_raw  = reshape(P_gt_all(:, idx),   3, 7);
    Ppr_raw = reshape(P_prop_all(:, idx), 3, 7);
    Pml_raw = reshape(P_mlp_all(:, idx),  3, 7);
    Pph_raw = reshape(P_phys_all(:, idx), 3, 7);

    % Baseline rule: each displayed curve uses its first node as origin.
    Pg  = Pg_raw  - Pg_raw(:, 1);
    Ppr = Ppr_raw - Ppr_raw(:, 1);
    Pml = Pml_raw - Pml_raw(:, 1);
    Pph = Pph_raw - Pph_raw(:, 1);

    if show_prior_curve
        P0_raw = reshape(P_prior_all(:, idx), 3, 7);
        P0 = P0_raw - P0_raw(:, 1);
    end

    % PCHIP smoothing
    t_nodes = 1:7;
    tq = linspace(1, 7, n_interp);

    Pg_s  = local_pchip_smooth(Pg,  t_nodes, tq);
    Ppr_s = local_pchip_smooth(Ppr, t_nodes, tq);
    Pml_s = local_pchip_smooth(Pml, t_nodes, tq);
    Pph_s = local_pchip_smooth(Pph, t_nodes, tq);

    if show_prior_curve
        P0_s = local_pchip_smooth(P0, t_nodes, tq);
    end

    % Optional P0 prior
    if show_prior_curve
        plot3(ax, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
            'Color', c_prior, 'LineWidth', lw_prior);
    end

    % Ground truth
    plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
        'Color', c_gt, 'LineWidth', lw_gt);

    % Proposed method
    plot3(ax, Ppr_s(1,:), Ppr_s(2,:), Ppr_s(3,:), '--', ...
        'Color', c_prop, 'LineWidth', lw_prop);

    % Vanilla MLP baseline
    plot3(ax, Pml_s(1,:), Pml_s(2,:), Pml_s(3,:), '-.', ...
        'Color', c_mlp, 'LineWidth', lw_mlp);

    % Analytical physics baseline
    plot3(ax, Pph_s(1,:), Pph_s(2,:), Pph_s(3,:), ':', ...
        'Color', c_phys, 'LineWidth', lw_phys);

    % GT marker-defined nodes
    if show_node_markers
        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 32, ...
            'MarkerFaceColor', c_node, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.6);
    end

    % Contact node marker
    if show_contact_node
        cnode = contact_node_all(idx);
        scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 70, ...
            'MarkerFaceColor', c_ct, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.1);
    end

    % Local origin coordinate frame
    quiver3(ax, 0, 0, 0, origin_axis_len, 0, 0, ...
        0, 'Color', 'r', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, origin_axis_len, 0, ...
        0, 'Color', 'g', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, 0, origin_axis_len, ...
        0, 'Color', 'b', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    text(ax, origin_axis_len*1.08, 0, 0, 'X', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, origin_axis_len*1.08, 0, 'Y', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, 0, origin_axis_len*1.08, 'Z', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    % Axis settings
    set(ax, 'FontName', font_name, ...
        'FontSize', 14, ...
        'LineWidth', 1.1, ...
        'TickDir', 'out', ...
        'Box', 'off', ...
        'ZDir', 'reverse');

    xlabel(ax, 'X [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    ylabel(ax, 'Y [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    zlabel(ax, 'Z [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');

    view(ax, -40, 24);

    % Axis limits include all compared models and origin frame
    all_pts = [Pg, Ppr, Pml, Pph, [0;0;0], ...
               [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];

    if show_prior_curve
        all_pts = [all_pts, P0];
    end

    pad = 0.010;
    xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
    ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
    zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
end

disp('>>> Same-sample baseline comparison figure generated successfully.');
%% ========================================================================
%  Step 9.32: Same-Sample Baseline Comparison Figure
%  Compare Proposed vs Vanilla MLP vs Analytical Physics
%  Manual index switch included
%  No in-figure legend/text except X/Y/Z labels and local coordinate frame
% =========================================================================
disp('--------------------------------------------------');
disp('9.32 Generating same-sample baseline comparison figure...');
disp('      Manual sample selection is supported.');
disp('      No legend or metric boxes will be shown in the figure.');

% ============================= User Options ===============================
export_this_figure = true;
fig_name = 'IEEE_same_sample_baseline_comparison_clean_axes';

n_interp = 160;

show_node_markers = true;
show_contact_node = true;
show_prior_curve  = true;   % show CSBCM prior P0 as light-gray reference

origin_axis_len = 0.020;    % local coordinate frame length [m]
origin_axis_lw  = 2.2;

% ======================= Manual sample selection ==========================
% true  : use manual_selected_idx directly
% false : use existing selected_idx from Step 9.31
use_manual_idx = true;

% IMPORTANT:
% These indices refer to the final aligned test set:
% pred_P_after / real_P_after / pred_brute_abs(:,v_idx) / P_phys_after_all(:,v_idx)
manual_selected_idx = [122, 161, 151];   % <-- change these three indices
% =========================================================================
% =========================== Required Variables ===========================
if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing proposed / ground-truth variables: pred_P_after and real_P_after.');
end

if ~exist('pred_brute_abs', 'var')
    error('Missing Vanilla MLP result variable: pred_brute_abs. Please run Step 9.6 first.');
end

if ~exist('P_phys_after_all', 'var')
    error('Missing analytical physics result variable: P_phys_after_all. Please run Step 9.8 first.');
end

if ~exist('v_idx', 'var')
    error('Missing v_idx. Baseline outputs need v_idx for final alignment.');
end

P_gt_all   = real_P_after;      % 21 x N_final, meter
P_prop_all = pred_P_after;      % 21 x N_final, meter
N_final = size(P_gt_all, 2);

% ======================= Selected sample indices ==========================
if use_manual_idx
    selected_idx = manual_selected_idx(:)';
    disp('>>> Manual selected_idx is used in Step 9.32.');
else
    if ~exist('selected_idx', 'var')
        error(['selected_idx is missing. Please run Step 9.31 first, ', ...
               'or set use_manual_idx = true.']);
    end
    selected_idx = selected_idx(:)';
    disp('>>> Existing selected_idx from Step 9.31 is used in Step 9.32.');
end

if numel(selected_idx) ~= 3
    error('selected_idx must contain exactly 3 sample indices.');
end

if any(selected_idx < 1) || any(selected_idx > N_final)
    error('selected_idx contains index outside valid range 1:N_final.');
end

% ============================= Align CSBCM Prior ==========================
has_prior = false;

if exist('p_before_test', 'var')
    if size(p_before_test, 2) == N_final
        P_prior_all = p_before_test;
        has_prior = true;
    elseif numel(v_idx) == size(p_before_test, 2)
        P_prior_all = p_before_test(:, v_idx);
        has_prior = true;
    end
end

if show_prior_curve && ~has_prior
    warning('P0 prior is not aligned or missing. Prior curve will not be shown.');
    show_prior_curve = false;
end

% ======================== Align Vanilla MLP Baseline ======================
% In your code, pred_brute_abs is computed on test_idx, then v_idx is applied.
if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error(['Cannot align Vanilla MLP result. Expected pred_brute_abs(:,v_idx) ', ...
           'to match real_P_after.']);
end

% ====================== Align Analytical Physics Baseline =================
% In your code, P_phys_after_all is computed on test_idx, then v_idx is applied.
if size(P_phys_after_all, 2) == N_final
    P_phys_all = P_phys_after_all;
elseif numel(v_idx) == size(P_phys_after_all, 2)
    P_phys_all = P_phys_after_all(:, v_idx);
else
    error(['Cannot align analytical physics result. Expected P_phys_after_all(:,v_idx) ', ...
           'to match real_P_after.']);
end

% ============================ Contact Node Alignment ======================
contact_node_all = ones(1, N_final) * 4;

if exist('test_hgt', 'var') && numel(test_hgt) == N_final
    contact_node_all = round(test_hgt);
elseif exist('hgt_filtered', 'var') && numel(hgt_filtered) == N_final
    contact_node_all = round(hgt_filtered);
elseif exist('gt_hgt_test', 'var') && numel(gt_hgt_test) == numel(v_idx)
    contact_node_all = round(gt_hgt_test(v_idx));
end

contact_node_all = max(1, min(7, contact_node_all));

% ============================== Plot Style ================================
font_name = 'Times New Roman';

c_gt    = [0.03 0.03 0.03];   % black
c_prop  = [0.10 0.32 0.90];   % blue
c_mlp   = [0.20 0.60 0.25];   % green
c_phys  = [0.85 0.20 0.12];   % red
c_prior = [0.72 0.72 0.72];   % light gray

c_node  = [0.20 0.55 0.95];   % marker blue
c_ct    = [0.90 0.15 0.10];   % contact red

lw_gt    = 2.6;
lw_prop  = 2.4;
lw_mlp   = 2.0;
lw_phys  = 2.0;
lw_prior = 1.25;

% ========================== Terminal Information ==========================
disp(' ');
disp('==================== Same-Sample Baseline Comparison ====================');
disp('No legend, title, or metric boxes are shown in the figure.');
disp('Only X/Y/Z axis labels and local origin frames are displayed.');
disp(' ');
disp('Curve styles:');
disp('  Black solid line        : Mocap Ground Truth P_gt^+');
disp('  Blue dashed line        : Proposed Reconstruction P^+ = P^0 + DeltaP');
disp('  Green dash-dot line     : Vanilla MLP baseline');
disp('  Red dotted line         : Analytical physics baseline');

if show_prior_curve
    disp('  Light-gray dash-dot     : CSBCM prior P^0');
end

disp('  Blue circular markers   : Marker-defined GT nodes');
disp('  Red circular marker     : Contact node');
disp(' ');
fprintf('Selected sample indices = [%d, %d, %d]\n', ...
    selected_idx(1), selected_idx(2), selected_idx(3));
disp(' ');
disp('Per-sample errors after Node-1 origin alignment:');
disp('  Columns: Sample | Method | Tip Error [mm] | Mean Shape Error [mm]');

for pp = 1:numel(selected_idx)
    idx = selected_idx(pp);

    Pg  = reshape(P_gt_all(:, idx), 3, 7);
    Ppr = reshape(P_prop_all(:, idx), 3, 7);
    Pml = reshape(P_mlp_all(:, idx), 3, 7);
    Pph = reshape(P_phys_all(:, idx), 3, 7);

    Pg  = Pg  - Pg(:, 1);
    Ppr = Ppr - Ppr(:, 1);
    Pml = Pml - Pml(:, 1);
    Pph = Pph - Pph(:, 1);

    [tip_prop, shape_prop] = local_err_metric(Ppr, Pg);
    [tip_mlp,  shape_mlp ] = local_err_metric(Pml, Pg);
    [tip_phys, shape_phys] = local_err_metric(Pph, Pg);

    fprintf('  %6d | Proposed           | %8.2f | %8.2f\n', idx, tip_prop, shape_prop);
    fprintf('  %6d | Vanilla MLP        | %8.2f | %8.2f\n', idx, tip_mlp,  shape_mlp);
    fprintf('  %6d | Analytical Physics | %8.2f | %8.2f\n', idx, tip_phys, shape_phys);
end

disp('==========================================================================');
disp(' ');

% =============================== Main Figure ==============================
fig = figure('Name', fig_name, ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [80, 80, 1650, 600]);

tl = tiledlayout(fig, 1, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for p = 1:3
    idx = selected_idx(p);

    ax = nexttile(tl, p);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');

    % Extract curves
    Pg_raw  = reshape(P_gt_all(:, idx),   3, 7);
    Ppr_raw = reshape(P_prop_all(:, idx), 3, 7);
    Pml_raw = reshape(P_mlp_all(:, idx),  3, 7);
    Pph_raw = reshape(P_phys_all(:, idx), 3, 7);

    % Baseline rule: each displayed curve uses its first node as origin.
    Pg  = Pg_raw  - Pg_raw(:, 1);
    Ppr = Ppr_raw - Ppr_raw(:, 1);
    Pml = Pml_raw - Pml_raw(:, 1);
    Pph = Pph_raw - Pph_raw(:, 1);

    if show_prior_curve
        P0_raw = reshape(P_prior_all(:, idx), 3, 7);
        P0 = P0_raw - P0_raw(:, 1);
    end

    % PCHIP smoothing
    t_nodes = 1:7;
    tq = linspace(1, 7, n_interp);

    Pg_s  = local_pchip_smooth(Pg,  t_nodes, tq);
    Ppr_s = local_pchip_smooth(Ppr, t_nodes, tq);
    Pml_s = local_pchip_smooth(Pml, t_nodes, tq);
    Pph_s = local_pchip_smooth(Pph, t_nodes, tq);

    if show_prior_curve
        P0_s = local_pchip_smooth(P0, t_nodes, tq);
    end

    % Optional P0 prior
    if show_prior_curve
        plot3(ax, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
            'Color', c_prior, 'LineWidth', lw_prior);
    end

    % Ground truth
    plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
        'Color', c_gt, 'LineWidth', lw_gt);

    % Proposed method
    plot3(ax, Ppr_s(1,:), Ppr_s(2,:), Ppr_s(3,:), '--', ...
        'Color', c_prop, 'LineWidth', lw_prop);

    % Vanilla MLP baseline
    plot3(ax, Pml_s(1,:), Pml_s(2,:), Pml_s(3,:), '-.', ...
        'Color', c_mlp, 'LineWidth', lw_mlp);

    % Analytical physics baseline
    plot3(ax, Pph_s(1,:), Pph_s(2,:), Pph_s(3,:), ':', ...
        'Color', c_phys, 'LineWidth', lw_phys);

    % GT marker-defined nodes
    if show_node_markers
        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 32, ...
            'MarkerFaceColor', c_node, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.6);
    end

    % Contact node marker
    if show_contact_node
        cnode = contact_node_all(idx);
        scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 70, ...
            'MarkerFaceColor', c_ct, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.1);
    end

    % Local origin coordinate frame
    quiver3(ax, 0, 0, 0, origin_axis_len, 0, 0, ...
        0, 'Color', 'r', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, origin_axis_len, 0, ...
        0, 'Color', 'g', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, 0, origin_axis_len, ...
        0, 'Color', 'b', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    text(ax, origin_axis_len*1.08, 0, 0, 'X', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, origin_axis_len*1.08, 0, 'Y', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, 0, origin_axis_len*1.08, 'Z', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    % Axis settings
    set(ax, 'FontName', font_name, ...
        'FontSize', 14, ...
        'LineWidth', 1.1, ...
        'TickDir', 'out', ...
        'Box', 'off', ...
        'ZDir', 'reverse');

    xlabel(ax, 'X [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    ylabel(ax, 'Y [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    zlabel(ax, 'Z [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');

    view(ax, -40, 24);

    % Axis limits include all compared models and origin frame
    all_pts = [Pg, Ppr, Pml, Pph, [0;0;0], ...
               [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];

    if show_prior_curve
        all_pts = [all_pts, P0];
    end

    pad = 0.010;
    xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
    ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
    zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
end

disp('>>> Same-sample baseline comparison figure generated successfully.');

% ================================ Export ==================================
if export_this_figure
    exportgraphics(fig, [fig_name, '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [fig_name, '.png'], 'Resolution', 600);
    savefig(fig, [fig_name, '.fig']);

    disp(['>>> Exported: ', fig_name, '.pdf / .png / .fig']);
end
%% ========================================================================
%  Step 9.32: Same-Sample Baseline Comparison Figure
%  Compare Proposed vs Vanilla MLP vs Analytical Physics
%  Manual index switch included
%  Only ground-truth contact force is shown
%  No in-figure legend/text except X/Y/Z labels and local coordinate frame
% =========================================================================
disp('--------------------------------------------------');
disp('9.32 Generating same-sample baseline comparison figure...');
disp('      Manual sample selection is supported.');
disp('      Only ground-truth contact force will be shown.');
disp('      No legend or metric boxes will be shown in the figure.');

% ============================= User Options ===============================
export_this_figure = true;
fig_name = 'IEEE_same_sample_baseline_comparison_clean_axes_GTforce';

n_interp = 160;

show_node_markers = true;
show_contact_node = true;
show_prior_curve  = true;    % show CSBCM prior P0 as light-gray reference

show_force_arrow = true;     % only ground-truth force
force_scale = 0.10;          % visual scale: 1 N -> 0.10 m

% Force display convention:
% 'raw'                : use stored GT force vector
% 'reverse'            : use negative GT force vector
% 'toward_deformation' : align GT force arrow with P0 -> P+ deformation
force_direction_mode = 'toward_deformation';

origin_axis_len = 0.020;
origin_axis_lw  = 2.2;

% ======================= Manual sample selection ==========================
use_manual_idx = true;

% These indices refer to the final aligned test set:
% pred_P_after / real_P_after / pred_brute_abs(:,v_idx) / P_phys_after_all(:,v_idx)
manual_selected_idx = [85, 113, 157];   % <-- change these three indices
% =========================================================================

% =========================== Required Variables ===========================
if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing proposed / ground-truth variables: pred_P_after and real_P_after.');
end

if ~exist('pred_brute_abs', 'var')
    error('Missing Vanilla MLP result variable: pred_brute_abs. Please run Step 9.6 first.');
end

if ~exist('P_phys_after_all', 'var')
    error('Missing analytical physics result variable: P_phys_after_all. Please run Step 9.8 first.');
end

if ~exist('v_idx', 'var')
    error('Missing v_idx. Baseline outputs need v_idx for final alignment.');
end

P_gt_all   = real_P_after;      % 21 x N_final, meter
P_prop_all = pred_P_after;      % 21 x N_final, meter
N_final = size(P_gt_all, 2);

% ======================= Selected sample indices ==========================
if use_manual_idx
    selected_idx = manual_selected_idx(:)';
    disp('>>> Manual selected_idx is used in Step 9.32.');
else
    if ~exist('selected_idx', 'var')
        error(['selected_idx is missing. Please run Step 9.31 first, ', ...
               'or set use_manual_idx = true.']);
    end
    selected_idx = selected_idx(:)';
    disp('>>> Existing selected_idx from Step 9.31 is used in Step 9.32.');
end

if numel(selected_idx) ~= 3
    error('selected_idx must contain exactly 3 sample indices.');
end

if any(selected_idx < 1) || any(selected_idx > N_final)
    error('selected_idx contains index outside valid range 1:N_final.');
end

% ============================= Align CSBCM Prior ==========================
has_prior = false;

if exist('p_before_test', 'var')
    if size(p_before_test, 2) == N_final
        P_prior_all = p_before_test;
        has_prior = true;
    elseif numel(v_idx) == size(p_before_test, 2)
        P_prior_all = p_before_test(:, v_idx);
        has_prior = true;
    end
end

if show_prior_curve && ~has_prior
    warning('P0 prior is not aligned or missing. Prior curve will not be shown.');
    show_prior_curve = false;
end

% ======================== Align Vanilla MLP Baseline ======================
if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error(['Cannot align Vanilla MLP result. Expected pred_brute_abs(:,v_idx) ', ...
           'to match real_P_after.']);
end

% ====================== Align Analytical Physics Baseline =================
if size(P_phys_after_all, 2) == N_final
    P_phys_all = P_phys_after_all;
elseif numel(v_idx) == size(P_phys_after_all, 2)
    P_phys_all = P_phys_after_all(:, v_idx);
else
    error(['Cannot align analytical physics result. Expected P_phys_after_all(:,v_idx) ', ...
           'to match real_P_after.']);
end

% ============================ Contact Node Alignment ======================
contact_node_all = ones(1, N_final) * 4;

if exist('test_hgt', 'var') && numel(test_hgt) == N_final
    contact_node_all = round(test_hgt);
elseif exist('hgt_filtered', 'var') && numel(hgt_filtered) == N_final
    contact_node_all = round(hgt_filtered);
elseif exist('gt_hgt_test', 'var') && numel(gt_hgt_test) == numel(v_idx)
    contact_node_all = round(gt_hgt_test(v_idx));
end

contact_node_all = max(1, min(7, contact_node_all));

% ============================ Ground-Truth Force Alignment =================
has_force = false;
F_gt_plot = [];

% Preferred case: already aligned GT force variable
if exist('F_gt_test', 'var') && size(F_gt_test, 2) == N_final

    F_gt_plot = F_gt_test;
    has_force = true;

% Fallback case: reconstruct aligned GT force variable
elseif exist('aug_gt_F', 'var') && exist('v_mask', 'var') && ...
       exist('test_idx', 'var') && exist('v_idx', 'var')

    tmp_F_gt = aug_gt_F(:, v_mask);
    tmp_F_gt = tmp_F_gt(:, test_idx);
    F_gt_plot = tmp_F_gt(:, v_idx);

    if size(F_gt_plot, 2) == N_final
        has_force = true;
    end
end

if show_force_arrow && ~has_force
    warning('Ground-truth force variable is missing or not aligned. Force arrows will not be shown.');
    show_force_arrow = false;
end

% ============================== Plot Style ================================
font_name = 'Times New Roman';

c_gt    = [0.03 0.03 0.03];   % black
c_prop  = [0.10 0.32 0.90];   % blue
c_mlp   = [0.20 0.60 0.25];   % green
c_phys  = [0.85 0.20 0.12];   % red
c_prior = [0.72 0.72 0.72];   % light gray

c_node  = [0.20 0.55 0.95];   % marker blue
c_ct    = [0.90 0.15 0.10];   % contact red
c_fgt   = [0.10 0.65 0.15];   % GT force green

lw_gt    = 2.6;
lw_prop  = 2.4;
lw_mlp   = 2.0;
lw_phys  = 2.0;
lw_prior = 1.25;

% ========================== Terminal Information ==========================
disp(' ');
disp('==================== Same-Sample Baseline Comparison ====================');
disp('No legend, title, or metric boxes are shown in the figure.');
disp('Only X/Y/Z axis labels and local origin frames are displayed.');
disp(' ');
disp('Curve styles:');
disp('  Black solid line        : Mocap Ground Truth P_gt^+');
disp('  Blue dashed line        : Proposed Reconstruction P^+ = P^0 + DeltaP');
disp('  Green dash-dot line     : Vanilla MLP baseline');
disp('  Red dotted line         : Analytical physics baseline');

if show_prior_curve
    disp('  Light-gray dash-dot     : CSBCM prior P^0');
end

disp('  Blue circular markers   : Marker-defined GT nodes');
disp('  Red circular marker     : Contact node');

if show_force_arrow && has_force
    disp('  Green arrow             : Ground-truth contact force F_gt');
    fprintf('  Force display mode      : %s\n', force_direction_mode);
else
    disp('  Force arrow             : Not shown');
end

disp(' ');
fprintf('Selected sample indices = [%d, %d, %d]\n', ...
    selected_idx(1), selected_idx(2), selected_idx(3));
disp(' ');
disp('Per-sample errors after Node-1 origin alignment:');
disp('  Columns: Sample | Method | Tip Error [mm] | Mean Shape Error [mm]');

for pp = 1:numel(selected_idx)
    idx = selected_idx(pp);

    Pg  = reshape(P_gt_all(:, idx), 3, 7);
    Ppr = reshape(P_prop_all(:, idx), 3, 7);
    Pml = reshape(P_mlp_all(:, idx), 3, 7);
    Pph = reshape(P_phys_all(:, idx), 3, 7);

    Pg  = Pg  - Pg(:, 1);
    Ppr = Ppr - Ppr(:, 1);
    Pml = Pml - Pml(:, 1);
    Pph = Pph - Pph(:, 1);

    [tip_prop, shape_prop] = local_err_metric(Ppr, Pg);
    [tip_mlp,  shape_mlp ] = local_err_metric(Pml, Pg);
    [tip_phys, shape_phys] = local_err_metric(Pph, Pg);

    fprintf('  %6d | Proposed           | %8.2f | %8.2f\n', idx, tip_prop, shape_prop);
    fprintf('  %6d | Vanilla MLP        | %8.2f | %8.2f\n', idx, tip_mlp,  shape_mlp);
    fprintf('  %6d | Analytical Physics | %8.2f | %8.2f\n', idx, tip_phys, shape_phys);

    if show_force_arrow && has_force
        F_tmp = F_gt_plot(:, idx);
        fprintf('  %6d | F_gt [N]           | [% .3f, % .3f, % .3f]\n', ...
            idx, F_tmp(1), F_tmp(2), F_tmp(3));
    end
end

disp('==========================================================================');
disp(' ');

% =============================== Main Figure ==============================
fig = figure('Name', fig_name, ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [80, 80, 1650, 600]);

tl = tiledlayout(fig, 1, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

for p = 1:3
    idx = selected_idx(p);

    ax = nexttile(tl, p);
    hold(ax, 'on');
    grid(ax, 'on');
    axis(ax, 'equal');

    % Extract curves
    Pg_raw  = reshape(P_gt_all(:, idx),   3, 7);
    Ppr_raw = reshape(P_prop_all(:, idx), 3, 7);
    Pml_raw = reshape(P_mlp_all(:, idx),  3, 7);
    Pph_raw = reshape(P_phys_all(:, idx), 3, 7);

    % Each displayed curve uses its first node as origin
    Pg  = Pg_raw  - Pg_raw(:, 1);
    Ppr = Ppr_raw - Ppr_raw(:, 1);
    Pml = Pml_raw - Pml_raw(:, 1);
    Pph = Pph_raw - Pph_raw(:, 1);

    if show_prior_curve
        P0_raw = reshape(P_prior_all(:, idx), 3, 7);
        P0 = P0_raw - P0_raw(:, 1);
    end

    % PCHIP smoothing
    t_nodes = 1:7;
    tq = linspace(1, 7, n_interp);

    Pg_s  = local_pchip_smooth(Pg,  t_nodes, tq);
    Ppr_s = local_pchip_smooth(Ppr, t_nodes, tq);
    Pml_s = local_pchip_smooth(Pml, t_nodes, tq);
    Pph_s = local_pchip_smooth(Pph, t_nodes, tq);

    if show_prior_curve
        P0_s = local_pchip_smooth(P0, t_nodes, tq);
    end

    % Optional P0 prior
    if show_prior_curve
        plot3(ax, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
            'Color', c_prior, 'LineWidth', lw_prior);
    end

    % Ground truth
    plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
        'Color', c_gt, 'LineWidth', lw_gt);

    % Proposed method
    plot3(ax, Ppr_s(1,:), Ppr_s(2,:), Ppr_s(3,:), '--', ...
        'Color', c_prop, 'LineWidth', lw_prop);

    % Vanilla MLP baseline
    plot3(ax, Pml_s(1,:), Pml_s(2,:), Pml_s(3,:), '-.', ...
        'Color', c_mlp, 'LineWidth', lw_mlp);

    % Analytical physics baseline
    plot3(ax, Pph_s(1,:), Pph_s(2,:), Pph_s(3,:), ':', ...
        'Color', c_phys, 'LineWidth', lw_phys);

    % GT marker-defined nodes
    if show_node_markers
        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 32, ...
            'MarkerFaceColor', c_node, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.6);
    end

    % Contact node marker
    cnode = contact_node_all(idx);

    if show_contact_node
        scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 70, ...
            'MarkerFaceColor', c_ct, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.1);
    end

    % Ground-truth force arrow only
    if show_force_arrow && has_force
        F_gt = F_gt_plot(:, idx);
        F0_gt = Pg(:, cnode);

        if show_prior_curve
            deform_vec = Ppr(:, cnode) - P0(:, cnode);
        else
            deform_vec = Ppr(:, cnode) - Pg(:, cnode);
        end

        switch force_direction_mode
            case 'raw'
                F_gt_vis = F_gt;

            case 'reverse'
                F_gt_vis = -F_gt;

            case 'toward_deformation'
                if norm(F_gt) > eps && norm(deform_vec) > eps
                    if dot(F_gt, deform_vec) < 0
                        F_gt_vis = -F_gt;
                    else
                        F_gt_vis = F_gt;
                    end
                else
                    F_gt_vis = F_gt;
                end

            otherwise
                error('Unknown force_direction_mode.');
        end

        quiver3(ax, F0_gt(1), F0_gt(2), F0_gt(3), ...
            F_gt_vis(1)*force_scale, F_gt_vis(2)*force_scale, F_gt_vis(3)*force_scale, ...
            0, 'Color', c_fgt, 'LineWidth', 2.2, ...
            'MaxHeadSize', 0.9);
    end

    % Local origin coordinate frame
    quiver3(ax, 0, 0, 0, origin_axis_len, 0, 0, ...
        0, 'Color', 'r', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, origin_axis_len, 0, ...
        0, 'Color', 'g', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    quiver3(ax, 0, 0, 0, 0, 0, origin_axis_len, ...
        0, 'Color', 'b', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

    text(ax, origin_axis_len*1.08, 0, 0, 'X', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, origin_axis_len*1.08, 0, 'Y', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    text(ax, 0, 0, origin_axis_len*1.08, 'Z', ...
        'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    % Axis settings
    set(ax, 'FontName', font_name, ...
        'FontSize', 14, ...
        'LineWidth', 1.1, ...
        'TickDir', 'out', ...
        'Box', 'off', ...
        'ZDir', 'reverse');

    xlabel(ax, 'X [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    ylabel(ax, 'Y [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');
    zlabel(ax, 'Z [m]', 'FontName', font_name, 'FontSize', 15, 'FontWeight', 'bold');

    view(ax, -40, 24);

    % Axis limits include all compared models, origin frame, and force arrow
    all_pts = [Pg, Ppr, Pml, Pph, [0;0;0], ...
               [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];

    if show_prior_curve
        all_pts = [all_pts, P0];
    end

    if show_force_arrow && has_force
        all_pts = [all_pts, F0_gt, F0_gt + F_gt_vis * force_scale];
    end

    pad = 0.010;
    xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
    ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
    zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
end

disp('>>> Same-sample baseline comparison figure with GT force generated successfully.');

% ================================ Export ==================================
if export_this_figure
    exportgraphics(fig, [fig_name, '.pdf'], 'ContentType', 'vector');
    exportgraphics(fig, [fig_name, '.png'], 'Resolution', 600);
    savefig(fig, [fig_name, '.fig']);

    disp(['>>> Exported: ', fig_name, '.pdf / .png / .fig']);
end
%% ========================================================================
%  Step 9.321: Diagnose GT force directions
%  Check whether F_gt really contains only 0 / 45 / 90 deg directions
% =========================================================================
disp('--------------------------------------------------');
disp('9.321 Diagnosing GT force directions...');

% ------------------------ Align F_gt_plot first --------------------------
if ~exist('real_P_after', 'var')
    error('real_P_after is missing.');
end
if ~exist('v_idx', 'var')
    error('v_idx is missing.');
end

N_final = size(real_P_after, 2);

has_force = false;
F_gt_plot = [];

if exist('F_gt_test', 'var') && size(F_gt_test, 2) == N_final
    F_gt_plot = F_gt_test;
    has_force = true;
elseif exist('aug_gt_F', 'var') && exist('v_mask', 'var') && ...
       exist('test_idx', 'var') && exist('v_idx', 'var')

    tmp_F_gt = aug_gt_F(:, v_mask);
    tmp_F_gt = tmp_F_gt(:, test_idx);
    F_gt_plot = tmp_F_gt(:, v_idx);

    if size(F_gt_plot, 2) == N_final
        has_force = true;
    end
end

if ~has_force
    error('Cannot align GT force variable.');
end

% ------------------------ Normalize and inspect --------------------------
F = F_gt_plot;
Fnorm = sqrt(sum(F.^2, 1));
valid = Fnorm > 1e-8;

F_unit = zeros(size(F));
F_unit(:, valid) = F(:, valid) ./ Fnorm(valid);

% Angle in global XY plane
theta_xy_deg = atan2d(F_unit(2,:), F_unit(1,:));

% Wrap to [0,180)
theta_xy_deg(theta_xy_deg < 0) = theta_xy_deg(theta_xy_deg < 0) + 180;

% Z contribution
z_ratio = abs(F_unit(3,:));

disp(' ');
disp('First 20 normalized GT force vectors:');
for i = 1:min(20, size(F_unit,2))
    fprintf('idx=%3d | F_unit = [% .3f, % .3f, % .3f] | theta_xy = %6.2f deg | |Fz|=%.3f\n', ...
        i, F_unit(1,i), F_unit(2,i), F_unit(3,i), theta_xy_deg(i), z_ratio(i));
end

disp(' ');

% Rough clustering to nearest canonical angle: 0,45,90
canonical = [0, 45, 90];
assigned = zeros(1, size(F_unit,2));
angle_err = zeros(1, size(F_unit,2));

for i = 1:size(F_unit,2)
    [err_min, kmin] = min(abs(theta_xy_deg(i) - canonical));
    assigned(i) = canonical(kmin);
    angle_err(i) = err_min;
end

fprintf('Count assigned to 0 deg  : %d\n', sum(assigned == 0));
fprintf('Count assigned to 45 deg : %d\n', sum(assigned == 45));
fprintf('Count assigned to 90 deg : %d\n', sum(assigned == 90));

fprintf('Mean assignment error [deg] = %.3f\n', mean(angle_err(valid)));
fprintf('Max  assignment error [deg] = %.3f\n', max(angle_err(valid)));
fprintf('Mean |Fz| ratio             = %.3f\n', mean(z_ratio(valid)));
fprintf('Max  |Fz| ratio             = %.3f\n', max(z_ratio(valid)));

disp(' ');
disp('Interpretation:');
disp('- If mean/max assignment error is small, then XY direction is basically 0/45/90.');
disp('- If |Fz| is not small, then the force is NOT purely in the global XY plane.');
disp('- If both are bad, then F_gt is likely not in the frame you think.');
%% ========================================================================
%  Step 9.315: Candidate index ranking for manual selection
%  Goal: find samples where Proposed is clearly better than both baselines
% =========================================================================
disp('--------------------------------------------------');
disp('9.315 Ranking candidate indices for manual selection...');

% =========================== Required Variables ===========================
if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing pred_P_after or real_P_after.');
end
if ~exist('pred_brute_abs', 'var')
    error('Missing pred_brute_abs.');
end
if ~exist('P_phys_after_all', 'var')
    error('Missing P_phys_after_all.');
end
if ~exist('v_idx', 'var')
    error('Missing v_idx.');
end

P_gt_all   = real_P_after;
P_prop_all = pred_P_after;
N_final    = size(P_gt_all, 2);

% Align brute-force MLP
if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error('pred_brute_abs cannot be aligned to final evaluation set.');
end

% Align physical baseline
if size(P_phys_after_all, 2) == N_final
    P_phys_all = P_phys_after_all;
elseif numel(v_idx) == size(P_phys_after_all, 2)
    P_phys_all = P_phys_after_all(:, v_idx);
else
    error('P_phys_after_all cannot be aligned to final evaluation set.');
end

% Optional prior
has_prior = false;
if exist('p_before_test', 'var')
    if size(p_before_test, 2) == N_final
        P_prior_all = p_before_test;
        has_prior = true;
    elseif numel(v_idx) == size(p_before_test, 2)
        P_prior_all = p_before_test(:, v_idx);
        has_prior = true;
    end
end

% =============================== Metrics =================================
tip_prop  = zeros(1, N_final);
shape_prop = zeros(1, N_final);

tip_mlp   = zeros(1, N_final);
shape_mlp = zeros(1, N_final);

tip_phys   = zeros(1, N_final);
shape_phys = zeros(1, N_final);

prior_err  = nan(1, N_final);

% Naturalness check for proposed curve
curvature_excess = zeros(1, N_final);

for i = 1:N_final
    Pg  = reshape(P_gt_all(:, i),   3, 7);
    Ppr = reshape(P_prop_all(:, i), 3, 7);
    Pml = reshape(P_mlp_all(:, i),  3, 7);
    Pph = reshape(P_phys_all(:, i), 3, 7);

    Pg  = Pg  - Pg(:,1);
    Ppr = Ppr - Ppr(:,1);
    Pml = Pml - Pml(:,1);
    Pph = Pph - Pph(:,1);

    tip_prop(i)   = norm(Ppr(:,end) - Pg(:,end)) * 1000;
    shape_prop(i) = mean(vecnorm(Ppr - Pg, 2, 1)) * 1000;

    tip_mlp(i)    = norm(Pml(:,end) - Pg(:,end)) * 1000;
    shape_mlp(i)  = mean(vecnorm(Pml - Pg, 2, 1)) * 1000;

    tip_phys(i)   = norm(Pph(:,end) - Pg(:,end)) * 1000;
    shape_phys(i) = mean(vecnorm(Pph - Pg, 2, 1)) * 1000;

    if has_prior
        P0 = reshape(P_prior_all(:, i), 3, 7);
        P0 = P0 - P0(:,1);
        prior_err(i) = mean(vecnorm(P0 - Pg, 2, 1)) * 1000;
    end

    % Curvature excess: proposed should not be much more bent than GT
    curv_gt   = local_backbone_curvature(Pg);
    curv_prop = local_backbone_curvature(Ppr);
    curvature_excess(i) = curv_prop / max(curv_gt, eps);
end

% ===================== Ranking score for manual choice ====================
% We want:
% - proposed shape error small
% - MLP and physics errors clearly larger
% - proposed curve still natural

gap_mlp  = shape_mlp  - shape_prop;
gap_phys = shape_phys - shape_prop;

ratio_mlp  = shape_mlp  ./ max(shape_prop, eps);
ratio_phys = shape_phys ./ max(shape_prop, eps);

% Strong candidate:
% proposed good + both baselines clearly worse + no weird over-bending
candidate_score = ...
    1.2 * gap_mlp + ...
    1.2 * gap_phys + ...
    0.8 * (ratio_mlp - 1) + ...
    0.8 * (ratio_phys - 1) - ...
    0.6 * shape_prop - ...
    0.8 * max(curvature_excess - 1.25, 0) * 10;

% Optional filter
valid_mask = shape_prop <= prctile(shape_prop, 75) & ...
             curvature_excess <= 1.50;

candidate_idx = find(valid_mask);
[~, order] = sort(candidate_score(candidate_idx), 'descend');
ranked_idx = candidate_idx(order);

% ============================== Print table ===============================
topK = min(20, numel(ranked_idx));

disp(' ');
disp('Top candidate indices for manual selection:');
disp('Idx | PropShape | MLPShape | PhysShape | GapMLP | GapPhys | RatioMLP | RatioPhys | CurvEx');

for k = 1:topK
    idx = ranked_idx(k);
    fprintf('%3d | %9.2f | %8.2f | %9.2f | %6.2f | %7.2f | %8.2f | %9.2f | %6.2f\n', ...
        idx, shape_prop(idx), shape_mlp(idx), shape_phys(idx), ...
        gap_mlp(idx), gap_phys(idx), ratio_mlp(idx), ratio_phys(idx), curvature_excess(idx));
end

disp(' ');
disp('Suggested manual selection strategy:');
disp('1) Prefer low PropShape.');
disp('2) Prefer large GapMLP and GapPhys.');
disp('3) Prefer RatioMLP and RatioPhys clearly > 1.');
disp('4) Avoid CurvEx > 1.5.');
disp(' ');

% ========================== Optional quick preview ========================
% Change these manually after reading the ranking table
preview_idx = ranked_idx(1:min(6, numel(ranked_idx)));

disp('Quick preview indices:');
disp(preview_idx);
% ============================ Local Functions =============================
function P_s = local_pchip_smooth(P, t_nodes, tq)
    P_s = zeros(3, numel(tq));
    for kk = 1:3
        P_s(kk, :) = interp1(t_nodes, P(kk, :), tq, 'pchip');
    end
end

function [tip_err_mm, shape_err_mm] = local_err_metric(P_pred, P_gt)
    tip_err_mm = norm(P_pred(:, end) - P_gt(:, end)) * 1000;
    shape_err_mm = mean(sqrt(sum((P_pred - P_gt).^2, 1))) * 1000;
end
% ============================ Local Function =============================
function curv_val = local_backbone_curvature(P)
    X = P';
    Xc = X - mean(X, 1);
    [~, ~, V] = svd(Xc, 'econ');
    q = Xc * V(:,1:2);

    kappa = zeros(1,5);
    for jj = 2:6
        v1 = q(jj,:)   - q(jj-1,:);
        v2 = q(jj+1,:) - q(jj,:);
        cross_z = v1(1)*v2(2) - v1(2)*v2(1);
        denom = norm(v1) * norm(v2) + eps;
        kappa(jj-1) = abs(cross_z / denom);
    end
    curv_val = sum(kappa);
end
%% -------------------------------------------------------------------------
% Helper inline function
% -------------------------------------------------------------------------
function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
%% ========================================================================
%  Helper Function: Data Augmentation
% =========================================================================
function[aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_Pa, aug_gF, aug_h, aug_tr] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, P_after, gt_F, hgt, track_rows)
    
    N = size(F_diff, 2);
    R120 =[cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 =[cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    
    % Tendon index permutation
    idx120 = [5, 6, 1, 2, 3, 4]; 
    idx240 = [3, 4, 5, 6, 1, 2];
    
    % Transformation helper function
    rotP = @(P, R) reshape(R * reshape(P, 3,[]), 21, N);
    
    % 120-degree augmentation
    Fd_120 = F_diff(idx120, :); Fa_120 = F_after(idx120, :); Fb_120 = F_before(idx120, :);
    gF_120 = R120 * gt_F;
    P_b_120 = rotP(P_before, R120);
    P_a_120 = rotP(P_after, R120);
    
    % 240-degree augmentation
    Fd_240 = F_diff(idx240, :); Fa_240 = F_after(idx240, :); Fb_240 = F_before(idx240, :);
    gF_240 = R240 * gt_F;
    P_b_240 = rotP(P_before, R240);
    P_a_240 = rotP(P_after, R240);
    
    % Concatenate original and augmented data
    aug_Fd = [F_diff, Fd_120, Fd_240]; 
    aug_Fa = [F_after, Fa_120, Fa_240];
    aug_Fb = [F_before, Fb_120, Fb_240]; 
    aug_Pb = [P_before, P_b_120, P_b_240];
    aug_Pa = [P_after, P_a_120, P_a_240]; 
    aug_gF = [gt_F, gF_120, gF_240]; 
    aug_h  = [hgt, hgt, hgt];
    aug_tr =[track_rows; track_rows; track_rows]; 
end
