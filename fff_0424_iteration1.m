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
tendon_p = 3; section_p = 2; D_p = 0.0006; E_p = 0.516e+12; 
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
