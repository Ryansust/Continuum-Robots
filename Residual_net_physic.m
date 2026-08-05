%% ========================================================================
%  Project: DL vs Physics - Robust Proprioceptive Pose Reconstruction
%  Module:  Net B + Physics Model (Replacing Net C)
% =========================================================================
clc; clear; close all;
rng('default'); % Ensure reproducibility 

%% ========================================================================
%  Step 1: Data Loading, ROI Filtering & Cleaning
% =========================================================================
disp('--------------------------------------------------');
disp('1. Loading and preprocessing data...');

FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
if ~isfile(FileName), error('File not found!'); end
dataTable = readtable(FileName);

F_after_raw  = double(table2array(dataTable(3:end, 23:28)))';  
F_before_raw = double(table2array(dataTable(3:end, 11:16)))';  
raw_mag_raw  = double(abs(table2array(dataTable(3:end, 2))))'; 
raw_dir_raw  = double(table2array(dataTable(3:end, 3)))';      
raw_hgt_raw  = double(table2array(dataTable(3:end, 4)))'; 
pos_text_raw = dataTable{3:end, 38}; 

disp('   > Executing ROI filtering (Nodes 3, 4, 5)...');
roi_mask = ismember(raw_hgt_raw, [3, 4, 5]);

F_after_sub  = F_after_raw(:, roi_mask);
F_before_sub = F_before_raw(:, roi_mask);
raw_mag_sub  = raw_mag_raw(roi_mask);
raw_dir_sub  = raw_dir_raw(roi_mask);
raw_hgt_sub  = raw_hgt_raw(roi_mask);
pos_text_sub = pos_text_raw(roi_mask);

if length(raw_mag_sub) < 50, error('Insufficient data after ROI filtering.'); end

disp('   > Removing invalid samples...');
bad_idx = any(isnan(F_after_sub), 1) | any(isnan(F_before_sub), 1) | ...
          isnan(raw_mag_sub) | isnan(raw_dir_sub) | isnan(raw_hgt_sub);
known_outliers = [686,16]; 
if ~isempty(known_outliers)
    bad_idx(known_outliers) = true; 
end

F_after  = F_after_sub(:, ~bad_idx);
F_before = F_before_sub(:, ~bad_idx);
raw_mag  = raw_mag_sub(~bad_idx);
raw_dir  = raw_dir_sub(~bad_idx);
raw_hgt  = raw_hgt_sub(~bad_idx);
pos_text = pos_text_sub(~bad_idx); 

F_diff = F_after - F_before;
N = length(raw_mag);
fprintf('   > Final effective samples: %d\n', N);

disp('   > Parsing pose data and generating ground truth...');
P_before = zeros(21, N); 
gt_F_vec = zeros(3, N);

for i = 1:N
    real_offset = get_RealOffset_1S3CT(pos_text{i});
    P_before(:, i) = reshape(real_offset(:, 3:end),[], 1); 
    
    u_vec = [0;0;0];
    switch raw_dir(i)
        case 2, u_vec = [-1; 0; 0];
        case 3, u_vec =[-sind(45); cosd(45); 0];
        case 4, u_vec = [0; 1; 0];
    end
    gt_F_vec(:, i) = raw_mag(i) * u_vec;
end

%% ========================================================================
%  Step 2 & 3: Augmentation & Dataset Construction
% =========================================================================
disp('--------------------------------------------------');
disp('2. Executing rotational augmentation & Building Sets...');
[aug_F_diff, aug_F_after, aug_F_before, aug_P_before, aug_gt_F, aug_hgt] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F_vec, raw_hgt);

inputs_f_final   = [aug_F_after; aug_F_diff; aug_F_before]; 
targets_f_final  = aug_gt_F;
inputs_loc_final =[aug_F_diff; aug_F_after; aug_P_before]; 
targets_loc_final = double(aug_hgt) / 9.0; 

bad_total = any(isnan(inputs_f_final), 1) | any(isinf(inputs_f_final), 1) | ...
            any(isnan(inputs_loc_final), 1) | any(isinf(inputs_loc_final), 1);
if sum(bad_total) > 0
    inputs_f_final(:, bad_total) = []; targets_f_final(:, bad_total) =[];
    inputs_loc_final(:, bad_total) = []; targets_loc_final(:, bad_total) =[];
    aug_gt_F(:, bad_total) =[];
end

epsilon = 1e-7;
inputs_f_final = inputs_f_final + epsilon * randn(size(inputs_f_final));
targets_f_final = targets_f_final + epsilon * randn(size(targets_f_final));
inputs_loc_final = inputs_loc_final + epsilon * randn(size(inputs_loc_final));
fprintf('   > Final input samples: %d\n', size(inputs_f_final, 2));

%% ========================================================================
%  Step 4: Net B - Force Estimation
% =========================================================================
disp('--------------------------------------------------');
disp('4. Training Net B Force...');

net_force = feedforwardnet([40, 20]);
net_force.trainFcn = 'trainlm';
net_force.trainParam.showWindow = false;

[net_force, tr_f] = train(net_force, inputs_f_final, targets_f_final);

% Evaluate
pred_f = net_force(inputs_f_final(:, tr_f.testInd));
targ_f = targets_f_final(:, tr_f.testInd);

if any(isnan(pred_f(:))), error('Net B Force produced NaN!'); end
mae_f = mean(abs(sqrt(sum(pred_f.^2)) - sqrt(sum(targ_f.^2))));
fprintf('   > Force MAE: %.4f N\n', mae_f);

%% ========================================================================
%  Step 5: Net B - Location Sensing (Weighted Loss)
% =========================================================================
disp('--------------------------------------------------');
disp('5. Training Net B Location (Weighted Loss)...');

% 5.1 Filter High-Force Samples
v_mask = sqrt(sum(aug_gt_F.^2)) > 0.08;
raw_in = inputs_loc_final(:, v_mask);
raw_tg = targets_loc_final(:, v_mask);
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

net_loc = fitnet([60, 40, 20]);
net_loc.trainFcn = 'trainlm'; 
net_loc.trainParam.showWindow = true; 
net_loc.trainParam.epochs = 1500;
net_loc.trainParam.max_fail = 20; 
net_loc.trainParam.goal = 1e-6;


net_loc.divideParam.trainRatio = 0.80;
net_loc.divideParam.valRatio   = 0.20;
net_loc.divideParam.testRatio  = 0.0;  % Manual eval later

[net_loc, tr_l] = train(net_loc, in_norm, tg_norm, [], [], weights_vec);

%% ========================================================================
%  Step 6: Evaluation & Visualization (Net B)
% =========================================================================
disp('--------------------------------------------------');
disp('6. Evaluating Net B performance...');

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
figure('Name', 'Net B: Location Results', 'Color', 'w', 'Position', [100, 100, 1000, 400]);
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
%  Step 7: 物理模型重构 (全样本支持版：原始 + 旋转增强数据)
% =========================================================================
disp('--------------------------------------------------');
disp('7. 正在运行物理学模型进行形态重构 (支持旋转增强数据)...');

% 1. 准备索引追溯
N_raw = length(raw_mag_raw);
track_excel_raw = 3 : (N_raw + 2); 
track_excel_sub = track_excel_raw(roi_mask);
track_excel_clean = track_excel_sub(~bad_idx);

% 对应 augment_data_by_rotation 里的逻辑
aug_track_excel = [track_excel_clean, track_excel_clean, track_excel_clean];
aug_rot_type = [zeros(1, length(track_excel_clean)), ...
                120 * ones(1, length(track_excel_clean)), ...
                240 * ones(1, length(track_excel_clean))];

% 最终索引库
final_track_excel = aug_track_excel(v_mask);
final_rot_type = aug_rot_type(v_mask);

% 2. 这里的 test_idx 包含了原始和增强样本
data_pool = inputs_loc_final(:, v_mask);
targets_net_c = data_pool(13:33, :); 

% 重新定义测试集：为了看旋转效果，我们随机取样（包含增强数据）
num_samples = size(data_pool, 2);
num_test = round(0.1 * num_samples); % 抽 10%
test_idx_in_pool = randperm(num_samples, num_test);

target_test = targets_net_c(:, test_idx_in_pool);
pred_test = zeros(21, num_test); 

% 3. 原始参数库
force_array_raw       = double(table2array(dataTable(3:end, 5:10))); 
outer_force_array_raw = double(table2array(dataTable(3:end, 2)));   
direction_array_raw   = double(table2array(dataTable(3:end, 3)));   
height_array_raw      = double(table2array(dataTable(3:end, 4)));   

% 仿真翻转矩阵 (根据你之前的结论)
Trans_sim = [-1, 0, 0; 0, 1, 0; 0, 0, 1]; 

fprintf('   > 正在执行物理仿真 (包含旋转补偿)...\n');

for i = 1:num_test
    idx_p = test_idx_in_pool(i);
    excel_row = final_track_excel(idx_p);
    rot_deg = final_rot_type(idx_p); % 获取该样本的旋转类型 (0, 120, 或 240)
    exp_id = excel_row - 2; 

    % --- A. 肌腱拉力变换 ---
    F_raw = force_array_raw(exp_id, :)'; % 1x6 原始数据
    
    % 根据旋转角度置换肌腱索引 (需匹配 augment_data_by_rotation 里的 idx120/240)
    if rot_deg == 120
        F_rot = F_raw([5, 6, 1, 2, 3, 4]); % 120度置换
    elseif rot_deg == 240
        F_rot = F_raw([3, 4, 5, 6, 1, 2]); % 240度置换
    else
        F_rot = F_raw; % 0度不置换
    end
    
    % 转换为仿真需要的 F_tendon2 (假设你的 solve 程序内部还需要一次 mapping)
    F_tendon_n = F_rot * 0.00981;
    F_tendon2 = [F_tendon_n(5); F_tendon_n(6); F_tendon_n(1); F_tendon_n(2); F_tendon_n(3); F_tendon_n(4)];

    % --- B. 外力矢量变换 ---
    f_mag = abs(outer_force_array_raw(exp_id));
    dir_code = direction_array_raw(exp_id);
    u_vec_orig = [0;0;0];
    switch dir_code
        case 2, u_vec_orig = [1; 0; 0];
        case 3, u_vec_orig = [sind(45); cosd(45); 0];
        case 4, u_vec_orig = [0; 1; 0];    
    end
    
    % 1. 先进行数据增强旋转
    Rz = [cosd(rot_deg), -sind(rot_deg), 0; sind(rot_deg), cosd(rot_deg), 0; 0, 0, 1];
    u_vec_rot = Rz * u_vec_orig;
    
    % 2. 再进行仿真坐标系翻转
    F_ex_sim = f_mag * (Trans_sim * u_vec_rot);

    % --- C. 求解 ---
    touch_id = height_array_raw(exp_id); 
    if touch_id == 0, touch_id = 14; end 

    [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon, section, D, E, L_a, L_b, N_d, H_list, mu, delta_alpha, G_load, F_ex_sim, F_tendon2, touch_id);
    
    % --- D. Marker 偏置与提取 (保持之前的 4mm 逻辑) ---
    offset_distance = 0.004; 
    V_local = [0; -offset_distance; 0]; 
    P_Theo_marker = zeros(size(P_Theo));
    for pt = 1:size(P_Theo, 2)
        P_Theo_marker(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local;
    end
    
    id_list = [2, 4, 6, 8, 10, 12, 14]; 
    kappa = (size(P_Theo, 2) - 1) / (section * N_d); 
    marker_indices = round(id_list * kappa) + 1;
    
    P_Theo_aligned = P_Theo_marker(:, marker_indices);
    pred_test(:, i) = reshape(P_Theo_aligned, 21, 1);
end
disp('   > 物理模型全样本求解完成！');

%% ========================================================================
%  Step 8: 整体重构误差评估
% =========================================================================
disp('--------------------------------------------------');
disp('8. 评估物理模型表现...');

dist_errs = zeros(1, num_test);
for i = 1:num_test
    p_p = reshape(pred_test(:, i), 3,[]);
    p_r = reshape(target_test(:, i), 3,[]);
    dist_errs(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
end
mean_dist = mean(dist_errs);
fprintf('   >[Physics Solver] 整体形态平均误差: %.4f m (%.2f mm)\n\n', mean_dist, mean_dist*1000);

num_plot_random = 3;
plot_ids = randperm(num_test, num_plot_random);

for k = 1:num_plot_random
    idx = plot_ids(k);
    idx_in_pool = test_idx_in_pool(idx); 
    real_excel_row = final_track_excel(idx_in_pool);

    P_p = reshape(pred_test(:, idx), 3,[]);
    P_r = reshape(target_test(:, idx), 3,[]);

    figure('Name', sprintf('Random Sample %d', k), 'Color', 'w', 'Position',[100+k*50, 100+k*50, 800, 600]);
    hold on; grid on; axis equal;
    
    h_base = plot3(0, 0, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
    plot3([0, P_r(1,1)],[0, P_r(2,1)],[0, P_r(3,1)], 'k-', 'LineWidth', 2);
    plot3([0, P_p(1,1)],[0, P_p(2,1)],[0, P_p(3,1)], 'm--', 'LineWidth', 1.5);
    
    h_r = plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-o', 'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor','k'); 
    h_p = plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'm--s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor','m');
    
    quiver3(0,0,0, 0.02,0,0, 'r', 'LineWidth', 3, 'MaxHeadSize', 2); 
    quiver3(0,0,0, 0,0.02,0, 'g', 'LineWidth', 3, 'MaxHeadSize', 2); 
    quiver3(0,0,0, 0,0,0.02, 'b', 'LineWidth', 3, 'MaxHeadSize', 2); 
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); 
    title(sprintf('Random Evaluation | Err: %.2f mm\n[Excel Row: %d]', dist_errs(idx)*1000, real_excel_row));
    legend([h_base, h_r, h_p], {'True Base (0,0,0)', 'Truth (Nokov)', 'Physics (Offset)'}, 'Location','best'); 
    view(30, 20);
end

%% ========================================================================
%  Step 9: 尖端 (Tip) 专项误差分析 & 【最差情况并精准定位 Excel】
% =========================================================================
disp('--------------------------------------------------');
disp('9. 正在进行尖端 (Tip) 专项误差分析与追溯...');

tip_idx =[19, 20, 21];
tip_pred = pred_test(tip_idx, :);
tip_real = target_test(tip_idx, :);

tip_vec = tip_pred - tip_real;
tip_dist = sqrt(sum(tip_vec.^2, 1)); 

tip_mae = mean(tip_dist);
tip_rmse = sqrt(mean(tip_dist.^2));
tip_max = max(tip_dist);

fprintf('   > [Tip] 平均误差 (MAE):  %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);
fprintf('   >[Tip] 均方根误差(RMSE): %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip] 最大误差 (Max):  %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

[sorted_tip_err, sort_idx] = sort(tip_dist, 'descend');
num_worst = 5;
worst_indices = sort_idx(1:num_worst);

disp('   =================================================');
disp('   🚨 警报！发现 5 个尖端误差最大的样本，追溯结果如下：');
for i = 1:num_worst
    idx = worst_indices(i);
    idx_in_pool = test_idx_in_pool(idx); 
    
    real_excel_row = final_track_excel(idx_in_pool);
    fprintf('   > Worst #%d: Tip Error = %.2f mm | Excel Row: %d\n', ...
            i, sorted_tip_err(i)*1000, real_excel_row);

    P_p = reshape(pred_test(:, idx), 3,[]);
    P_r = reshape(target_test(:, idx), 3,[]);
    
    figure('Name', sprintf('Worst Tip Error #%d', i), 'Color', 'w', 'Position',[600+i*50, 100+i*50, 800, 600]);
    hold on; grid on; axis equal;
    
    h_base = plot3(0, 0, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
    quiver3(0,0,0, 0.02,0,0, 'r', 'LineWidth', 3); quiver3(0,0,0, 0,0.02,0, 'g', 'LineWidth', 3); quiver3(0,0,0, 0,0,0.02, 'b', 'LineWidth', 3); 
    
    plot3([0, P_r(1,1)],[0, P_r(2,1)],[0, P_r(3,1)], 'k-', 'LineWidth', 2);
    plot3([0, P_p(1,1)],[0, P_p(2,1)],[0, P_p(3,1)], 'm--', 'LineWidth', 1.5);
    
    h_r = plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-o', 'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor','k'); 
    h_p = plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'm--s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor','m');
    
    plot3(P_r(1,end), P_r(2,end), P_r(3,end), 'bp', 'MarkerSize', 18, 'MarkerFaceColor', 'b'); 
    plot3(P_p(1,end), P_p(2,end), P_p(3,end), 'rp', 'MarkerSize', 18, 'MarkerFaceColor', 'r'); 
    plot3([P_r(1,end), P_p(1,end)],[P_r(2,end), P_p(2,end)],[P_r(3,end), P_p(3,end)], 'k-', 'LineWidth', 3); 
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); 
    title(sprintf('【Worst Case #%d】 Tip Error: %.2f mm\n👉 Check Excel Row: %d', i, sorted_tip_err(i)*1000, real_excel_row));
    legend([h_base, h_r, h_p], {'True Base', 'Truth', 'Physics'}, 'Location','best'); 
    view(30, 20);
end

% -------------------------------------------------------------------------
% 散点追踪与直方图
% -------------------------------------------------------------------------
figure('Name', 'Physics Tip Error - Tracking', 'Color', 'w', 'Position',[100, 200, 800, 600]);
hold on; grid on; axis equal;

plot3(0, 0, 0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
quiver3(0,0,0, 0.02,0,0, 'r', 'LineWidth', 2); quiver3(0,0,0, 0,0.02,0, 'g', 'LineWidth', 2); quiver3(0,0,0, 0,0,0.02, 'b', 'LineWidth', 2); 

h1 = plot3(NaN,NaN,NaN, 'bo'); h2 = plot3(NaN,NaN,NaN, 'm.');
num_show = min(50, num_test);
idx_show = randperm(num_test, num_show);

for k = idx_show
    p_r = tip_real(:, k); p_p = tip_pred(:, k);
    plot3([p_r(1), p_p(1)],[p_r(2), p_p(2)],[p_r(3), p_p(3)], 'Color',[0.7 0.7 0.7]);
    plot3(p_r(1), p_r(2), p_r(3), 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    plot3(p_p(1), p_p(2), p_p(3), 'm.', 'MarkerSize', 12);
end
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); title('Tip Tracking 3D Space (Physics Solver)'); 
legend([h1, h2], {'Ground Truth Tip', 'Physics Prediction Tip'}); view(45, 30);

figure('Name', 'Physics Tip Error - Histogram', 'Color', 'w', 'Position',[950, 200, 800, 600]);
histogram(tip_dist * 1000, 30, 'FaceColor',[0.6 0.2 0.6]); 
xline(tip_mae * 1000, 'r--', 'LineWidth', 3);
xlabel('Tip Error (mm)', 'FontSize', 12); ylabel('Sample Count', 'FontSize', 12); 
title('Tip Error Distribution', 'FontSize', 14); grid on;
%% 
% =========================================================================
% 🌟 进阶高阶分析：Tip 误差的柱坐标物理解耦 (Cylindrical Error Decomposition)
% =========================================================================
disp('--------------------------------------------------');
disp('10. 正在进行高阶物理误差解耦分析 (Error Decomposition)...');

% 计算水平面投影半径 (弯曲幅度)
r_pred = sqrt(tip_pred(1,:).^2 + tip_pred(2,:).^2);
r_real = sqrt(tip_real(1,:).^2 + tip_real(2,:).^2);

% 计算水平方位角 (弯曲方向，单位：度)
theta_pred = atan2(tip_pred(2,:), tip_pred(1,:)) * 180/pi;
theta_real = atan2(tip_real(2,:), tip_real(1,:)) * 180/pi;

% 1. 弯曲幅度误差 (Radial Error)
err_r = r_pred - r_real; 

% 2. 弯曲方向误差 (Angular Error)
err_theta = theta_pred - theta_real;
err_theta = mod(err_theta + 180, 360) - 180; 

% 3. 轴向高度误差 (Z-axis Error)
err_z = tip_pred(3,:) - tip_real(3,:);

% 打印深度分析结论
fprintf('\n🎯 【误差分解物理洞察】 (N = %d 样本)\n', num_test);
fprintf('  > 绝对空间距离误差 (3D Norm): %.2f mm\n', mean(tip_dist)*1000);
fprintf('  -------------------------------------------------\n');
fprintf('  > 1. 弯曲幅度误差 (Radial Δr) : 平均值 %+.2f mm  (绝对均值 %.2f mm)\n', mean(err_r)*1000, mean(abs(err_r))*1000);
fprintf('  > 2. 弯曲方向偏角 (Angle Δθ)  : 平均值 %+.2f 度  (绝对均值 %.2f 度)\n', mean(err_theta), mean(abs(err_theta)));
fprintf('  > 3. 轴向高度压缩 (Height Δz) : 平均值 %+.2f mm  (绝对均值 %.2f mm)\n', mean(err_z)*1000, mean(abs(err_z))*1000);

% =========================================================================
% 🎨 绘制带保姆级标注的精美分析图
% =========================================================================
figure('Name', 'Physics Error Decomposition', 'Color', 'w', 'Position',[150, 300, 1100, 450]);

% --- 图1：弯曲幅度偏差 ---
subplot(1,3,1); hold on; grid on;
% 画半透明散点 (让数据分布一目了然)
scatter(ones(size(err_r)).*(1+(rand(size(err_r))-0.5)*0.2), err_r * 1000, 20, 'm', 'filled', 'MarkerFaceAlpha', 0.5);
boxplot(err_r * 1000, 'Colors', 'k', 'Symbol', '');
yline(0, 'r-', 'LineWidth', 2); % 完美的 0 误差红线
title('1. 弯曲幅度偏差 (\Delta r)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('误差值 (mm)', 'FontSize', 11);
set(gca, 'XTick',[]); 
y_lim = ylim;
text(1.15, y_lim(2)*0.8, '↑ 模型弯得比真机多', 'Color', 'm', 'FontSize', 10, 'FontWeight', 'bold');
text(1.15, y_lim(1)*0.8, '↓ 模型弯得比真机少', 'Color', 'b', 'FontSize', 10, 'FontWeight', 'bold');

% --- 图2：弯曲朝向偏差 ---
subplot(1,3,2); hold on; grid on;
scatter(ones(size(err_theta)).*(1+(rand(size(err_theta))-0.5)*0.2), err_theta, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
boxplot(err_theta, 'Colors', 'k', 'Symbol', '');
yline(0, 'r-', 'LineWidth', 2);
title('2. 弯曲方向偏差 (\Delta \theta)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('角度误差 (度)', 'FontSize', 11);
set(gca, 'XTick',[]);
y_lim = ylim;
text(1.15, y_lim(2)*0.8, '↑ 模型偏逆时针', 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
text(1.15, y_lim(1)*0.8, '↓ 模型偏顺时针', 'Color', 'b', 'FontSize', 10, 'FontWeight', 'bold');

% --- 图3：高度压缩偏差 ---
subplot(1,3,3); hold on; grid on;
scatter(ones(size(err_z)).*(1+(rand(size(err_z))-0.5)*0.2), err_z * 1000, 20, 'g', 'filled', 'MarkerFaceAlpha', 0.5);
boxplot(err_z * 1000, 'Colors', 'k', 'Symbol', '');
yline(0, 'r-', 'LineWidth', 2);
title('3. 轴向高度偏差 (\Delta z)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('高度误差 (mm)', 'FontSize', 11);
set(gca, 'XTick',[]);
xlim([0.5, 2]);
y_l = ylim;
text(1.3, y_l(2) - (y_l(2)-y_l(1))*0.1, '↑ 模型算出来更高', 'Color', [0 0.5 0], 'FontSize', 11, 'FontWeight', 'bold');
text(1.3, y_l(1) + (y_l(2)-y_l(1))*0.1, '↓ 模型算出来更矮', 'Color', [0.5 0 0], 'FontSize', 11, 'FontWeight', 'bold');

disp('>>> 高阶误差解耦绘图完成！图表已最大化显示。');
%% === 旋转增强辅助函数 ===
function[aug_Fd, aug_Fa, aug_Fb, aug_Pb, aug_gF, aug_h] = ...
    augment_data_by_rotation(F_diff, F_after, F_before, P_before, gt_F, hgt)
    N = size(F_diff, 2);
    R120 =[cosd(120), -sind(120), 0; sind(120), cosd(120), 0; 0, 0, 1];
    R240 =[cosd(240), -sind(240), 0; sind(240), cosd(240), 0; 0, 0, 1];
    idx120 =[5, 6, 1, 2, 3, 4]; idx240 =[3, 4, 5, 6, 1, 2];
    
    Fd_120 = F_diff(idx120, :); Fa_120 = F_after(idx120, :); Fb_120 = F_before(idx120, :);
    gF_120 = R120 * gt_F;
    P_tmp = reshape(P_before, 3,[]); P_120 = reshape(R120 * P_tmp, 21, N);
    
    Fd_240 = F_diff(idx240, :); Fa_240 = F_after(idx240, :); Fb_240 = F_before(idx240, :);
    gF_240 = R240 * gt_F; P_240 = reshape(R240 * P_tmp, 21, N);
    
    aug_Fd =[F_diff, Fd_120, Fd_240]; aug_Fa =[F_after, Fa_120, Fa_240];
    aug_Fb =[F_before, Fb_120, Fb_240]; aug_Pb =[P_before, P_120, P_240];
    aug_gF =[gt_F, gF_120, gF_240]; aug_h  =[hgt, hgt, hgt];
end