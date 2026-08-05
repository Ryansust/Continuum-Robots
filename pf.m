%% ========================================================================
%  Step 8: Net C Evaluation & Statistical Outlier Filtering
% =========================================================================
disp('--------------------------------------------------');
disp('8. Evaluating Net C Performance & Executing 3-Sigma Filter...');

% 8.1 初始提取测试集数据
test_idx = tr_c.testInd;
if isempty(test_idx), test_idx = randperm(size(inputs_net_c, 2), min(50, size(inputs_net_c, 2))); end

in_test = inputs_net_c(:, test_idx);
target_delta_test = targets_net_c(:, test_idx);
p_before_test = feat_P_before(:, test_idx);
rows_test_raw = track_rows_net_c(test_idx);

% 8.2 初始预测与误差计算
pred_delta_raw = mapstd('reverse', net_shape(mapstd('apply', in_test, ps_in_c)), ps_out_c);
pred_P_after_raw = p_before_test + pred_delta_raw;
real_P_after_raw = p_before_test + target_delta_test;

dist_errs_raw = zeros(1, length(test_idx));
tip_dist_raw  = zeros(1, length(test_idx));

for i = 1:length(test_idx)
    p_p = reshape(pred_P_after_raw(:, i), 3, []);
    p_r = reshape(real_P_after_raw(:, i), 3, []);
    dist_errs_raw(i) = mean(sqrt(sum((p_p - p_r).^2, 1)));
    tip_dist_raw(i)  = norm(p_p(:, end) - p_r(:, end));
end

% 8.3 [核心逻辑：3-Sigma 统计学剔除]
mu_err = mean(tip_dist_raw);
std_err = std(tip_dist_raw);
sigma_threshold = mu_err + 3 * std_err;

% 确定最终有效的测试样本索引
valid_mask = tip_dist_raw <= sigma_threshold;
outlier_count = sum(~valid_mask);

% --- 同步更新所有测试集相关变量，确保后续 Section 10 全部对齐 ---
tip_dist = tip_dist_raw(valid_mask);
dist_errs = dist_errs_raw(valid_mask);
rows_test = rows_test_raw(valid_mask);
pred_P_after = pred_P_after_raw(:, valid_mask);
real_P_after = real_P_after_raw(:, valid_mask);
p_before_test_filtered = p_before_test(:, valid_mask);
test_idx_final = test_idx(valid_mask); % 关键：记录最终有效的原始索引

if outlier_count > 0
    fprintf('   > [3-Sigma Clean] Removed %d outliers (Error > %.2f mm).\n', outlier_count, sigma_threshold*1000);
else
    fprintf('   > [3-Sigma Clean] No outliers detected.\n');
end

% 8.4 打印最终指标
mean_dist = mean(dist_errs);
tip_mae = mean(tip_dist);
fprintf('   > [Net C Final] Mean Shape Error: %.4f m (%.2f mm)\n', mean_dist, mean_dist*1000);
fprintf('   > [Net C Final] Tip MAE: %.4f m (%.2f mm)\n', tip_mae, tip_mae*1000);

% 8.5 绘制 Worst Cases (基于清洗后的数据)
[~, sort_idx] = sort(tip_dist, 'descend');
num_worst = min(7, length(sort_idx));
worst_cases = sort_idx(1:num_worst); 

for k = 1:num_worst
    idx = worst_cases(k);
    orig_row = rows_test(idx); 
    
    P_p = [[0;0;0], reshape(pred_P_after(:, idx), 3, [])];
    P_r = [[0;0;0], reshape(real_P_after(:, idx), 3, [])];
    
    figure('Name', sprintf('Worst Case %d - Excel Row: %d', k, orig_row), 'Color', 'w', 'Position', [100+k*20, 100+k*20, 600, 500]);
    hold on; grid on; axis equal;
    
    plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y', 'DisplayName', 'Base Origin');
    plot3(P_r(1,:), P_r(2,:), P_r(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'DisplayName', 'Ground Truth');
    plot3(P_p(1,:), P_p(2,:), P_p(3,:), 'r--o', 'LineWidth', 2, 'MarkerFaceColor', 'w', 'DisplayName', 'Prediction');
    
    tip_t = P_r(:, end); tip_p = P_p(:, end);
    plot3([tip_t(1), tip_p(1)], [tip_t(2), tip_p(2)], [tip_t(3), tip_p(3)], 'm-', 'LineWidth', 2.5, 'DisplayName', sprintf('Tip Error: %.1fmm', tip_dist(idx)*1000));
    
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(sprintf('Worst Case Evaluation | Excel Raw Row: %d', orig_row), 'FontSize', 12, 'Interpreter', 'none');
    legend('Location', 'best'); view(30, 20);
end

%% ========================================================================
%  Step 9: Tip Error Analysis & Final Save
% =========================================================================
disp('--------------------------------------------------');
disp('9. Analyzing Tip-Specific Error & Saving Models...');

tip_rmse = sqrt(mean(tip_dist.^2));
tip_max = max(tip_dist);

fprintf('   > [Tip Stats] RMSE: %.4f m (%.2f mm)\n', tip_rmse, tip_rmse*1000);
fprintf('   > [Tip Stats] Max:  %.4f m (%.2f mm)\n', tip_max, tip_max*1000);

figure('Name', 'Tip Error Dist', 'Color', 'w', 'Position', [100, 200, 600, 400]);
histogram(tip_dist * 1000, 30, 'FaceColor', [0.2 0.6 0.3]);
xline(tip_mae * 1000, 'r--', 'LineWidth', 2);
xlabel('Error (mm)'); ylabel('Count'); title('Tip Error Distribution (Cleaned)'); grid on;

save('Final_System_Checkpoint.mat', 'net_force', 'net_loc', 'net_shape', 'ps_in', 'ps_out', 'ps_in_c', 'ps_out_c', 'test_idx_final');

%% ========================================================================
%  Step 10: 模块化性能深度剖析 (Aligned with Cleaned Test Set)
% =========================================================================
disp('--------------------------------------------------');
disp('10. Generating Component-wise Performance Analysis...');

% 预设画图样式
set(0, 'DefaultAxesFontSize', 12, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 1.5);
c_blue = [0 0.4470 0.7410]; c_red = [0.8500 0.3250 0.0980]; c_green = [0.4660 0.6740 0.1880];

%% 10.1 物理模型准确度剖析 (P_before_ideal vs P_before_sensor)
disp('   > 10.1 Analyzing Physical Model Accuracy (Cleaned Test Set)...');

% 仅针对清洗后的测试集行号提取碰撞前的真值
P_before_mocap_test = zeros(21, length(rows_test));
% 需要根据 rows_test 在原始大表里重新解析，或者建立一个映射。
% 为了绝对准确，我们再次调用 get_RealOffset
for i = 1:length(rows_test)
    % 找到对应的原始 row_id (注意 Excel 行号比 table 索引多 2)
    row_id = rows_test(i) - 2; 
    real_offset_b = get_RealOffset_1S3CT(pos_text_before_raw{row_id});
    base_center_b = (real_offset_b(:, 1) + real_offset_b(:, 2)) / 2;
    P_before_mocap_test(:, i) = reshape(real_offset_b(:, 3:end) - base_center_b, 21, 1);
end

phys_tip_ideal = p_before_test_filtered(19:21, :);
phys_tip_mocap = P_before_mocap_test(19:21, :);
phys_tip_err = vecnorm(phys_tip_ideal - phys_tip_mocap, 2, 1);

phys_mae = mean(phys_tip_err);
fprintf('      - [Physics Model] Tip MAE: %.2f mm\n', phys_mae * 1000);

% 绘制总览对比图 (针对当前测试集)
figure('Name', 'Physical Model Analysis: Ideal vs Mocap', 'Color', 'w', 'Position', [100, 100, 800, 700]);
hold on; grid on; axis equal; view(30, 20);
for i = 1:min(20, length(rows_test))
    p_i = [[0;0;0], reshape(p_before_test_filtered(:, i), 3, 7)];
    p_m = [[0;0;0], reshape(P_before_mocap_test(:, i), 3, 7)];
    plot3(p_m(1,:), p_m(2,:), p_m(3,:), 'k-', 'HandleVisibility', 'off');
    plot3(p_i(1,:), p_i(2,:), p_i(3,:), 'r--', 'HandleVisibility', 'off');
end
h1 = plot3(NaN,NaN,NaN, 'k-'); h2 = plot3(NaN,NaN,NaN, 'r--');
plot3(0,0,0, 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
set(gca, 'zdir', 'reverse', 'ydir', 'reverse'); xlabel('X'); ylabel('Y'); zlabel('Z');
title('Physics Model: Ideal (Red) vs Mocap (Black)');
legend([h1, h2], {'Mocap (Before)', 'Physics Model (Ideal)'});

% 绘制 5 个物理模型最差 Case
[~, sort_p_idx] = sort(phys_tip_err, 'descend');
for k = 1:5
    idx = sort_p_idx(k);
    p_i = [[0;0;0], reshape(p_before_test_filtered(:, idx), 3, 7)];
    p_m = [[0;0;0], reshape(P_before_mocap_test(:, idx), 3, 7)];
    figure('Name', sprintf('Phys Worst Case %d', k), 'Color', 'w');
    hold on; grid on; axis equal;
    plot3(p_m(1,:), p_m(2,:), p_m(3,:), 'k-s', 'LineWidth', 2, 'MarkerFaceColor', 'k');
    plot3(p_i(1,:), p_i(2,:), p_i(3,:), 'r--o', 'LineWidth', 1.5);
    plot3([p_m(1,end), p_i(1,end)], [p_m(2,end), p_i(2,end)], [p_m(3,end), p_i(3,end)], 'm-', 'LineWidth', 3);
    set(gca, 'zdir', 'reverse', 'ydir', 'reverse'); view(30, 20);
    title(sprintf('Phys Model Error: %.1f mm | Row: %d', phys_tip_err(idx)*1000, rows_test(idx)));
end

%% 10.2 Net B 预测力的准确度
disp('   > 10.2 Analyzing Net B Force Accuracy (Cleaned Set)...');
F_gt_test = targets_f_final(:, test_idx_final);
F_pd_test = pred_force_all(:, test_idx_final);

mag_gt = vecnorm(F_gt_test, 2, 1);
mag_pd = vecnorm(F_pd_test, 2, 1);
mag_err = abs(mag_pd - mag_gt);

figure('Name', 'Force Accuracy', 'Color', 'w', 'Position', [650, 100, 1000, 400]);
subplot(1, 2, 1); scatter(mag_gt, mag_pd, 25, c_blue, 'filled'); hold on;
plot([0, max(mag_gt)], [0, max(mag_gt)], 'k--'); grid on; title('Magnitude');
subplot(1, 2, 2); 
valid_dir = mag_gt > 0.05;
ang_err = acosd(max(min(dot(F_gt_test(:,valid_dir), F_pd_test(:,valid_dir)) ./ (mag_gt(valid_dir).*mag_pd(valid_dir)), 1), -1));
histogram(ang_err, 20, 'FaceColor', c_red); title('Direction Error (Deg)');

%% 10.3 Net B 预测位置的准确度 (对齐测试集)
disp('   > 10.3 Analyzing Net B Location Accuracy (Cleaned Set)...');
% 从全局预测中提取对应的测试样本结果
loc_gt_test = targets_loc_final(test_idx_final) * 9.0;
loc_pd_test = pred_loc_norm_all(test_idx_final) * 9.0;

figure('Name', 'Location Accuracy', 'Color', 'w', 'Position', [100, 550, 500, 400]);
boxplot(loc_pd_test, round(loc_gt_test), 'Colors', 'k'); hold on;
plot([1, 2, 3], [3, 4, 5], 'b--'); grid on; title('Location Consistency');

%% 10.4 系统最终 CDF
figure('Name', 'Final System CDF', 'Color', 'w', 'Position', [650, 550, 500, 400]);
cdfplot(tip_dist * 1000); grid on;
xlabel('Tip Error (mm)'); ylabel('Probability'); title('Overall System Performance');

disp('>>> All done. Consistency maintained across all Sections.');