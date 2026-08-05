%% ========================================================================
% Step 9.9: Tip-Error Comparison between Vanilla MLP and Proposed Method
% ========================================================================

disp('--------------------------------------------------');
disp('9.9 Generating Vanilla MLP vs. Proposed tip-error comparison...');

%% ============================================================
% Global Figure Display Options
% ============================================================

show_legend      = false;  % true: show legend; false: hide legend
show_axis_labels = false;  % true: show xlabel/ylabel; false: hide xlabel/ylabel
show_tick_labels = true;  % true: show tick numbers; false: hide numbers only

% =========================================================================
% 1. Check required variables
% =========================================================================

if ~exist('error_brute_final', 'var')
    error('Missing error_brute_final. Please run Step 9.6 first.');
end

if ~exist('tip_dist', 'var')
    error('Missing tip_dist. Please run Step 8 first.');
end

if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing pred_P_after or real_P_after. Please run Step 8 first.');
end

% Vanilla MLP tip error，单位 mm
error_mlp_all_original = error_brute_final(:)';

% Proposed tip error，tip_dist 的单位是 m，转换为 mm
error_prop_all_original = tip_dist(:)' * 1000;

% 严格检查样本对齐
if numel(error_mlp_all_original) ~= numel(error_prop_all_original)
    error(['Sample mismatch: Vanilla MLP has %d samples, ', ...
           'while Proposed has %d samples.'], ...
           numel(error_mlp_all_original), ...
           numel(error_prop_all_original));
end

if size(pred_P_after, 2) ~= numel(error_prop_all_original)
    error(['Shape/sample mismatch: pred_P_after has %d samples, ', ...
           'while Proposed tip error has %d samples.'], ...
           size(pred_P_after, 2), ...
           numel(error_prop_all_original));
end

if size(real_P_after, 2) ~= numel(error_prop_all_original)
    error(['Shape/sample mismatch: real_P_after has %d samples, ', ...
           'while Proposed tip error has %d samples.'], ...
           size(real_P_after, 2), ...
           numel(error_prop_all_original));
end

% 删除 NaN / Inf
valid_mask = ...
    isfinite(error_mlp_all_original) & ...
    isfinite(error_prop_all_original);

% 保存“压缩后索引”到“原始最终测试集索引”的对应关系
valid_original_idx = find(valid_mask);

error_mlp_all  = error_mlp_all_original(valid_mask);
error_prop_all = error_prop_all_original(valid_mask);

fprintf('   > Samples before finite-value filtering: %d\n', ...
    numel(error_prop_all_original));

fprintf('   > Samples after finite-value filtering : %d\n', ...
    numel(error_prop_all));

% =========================================================================
% 沿用原 Step 9.9 的样本筛选逻辑，仅删除 physics baseline 条件
% =========================================================================

MLP_THRES  = 5.5;
PROP_THRES = 4.6;
MLP_UPPER_THRES = 30;

mask_perfect = ...
    (error_mlp_all > MLP_THRES) & ...
    (error_prop_all < PROP_THRES) & ...
    (error_mlp_all < MLP_UPPER_THRES);

idx_perfect = find(mask_perfect);

fprintf('   > Found %d samples matching the filtering criteria.\n', ...
    numel(idx_perfect));

if isempty(idx_perfect)
    error(['No sample satisfies MLP > %.2f mm, Proposed < %.2f mm, ', ...
           'and MLP < %.2f mm.'], ...
           MLP_THRES, PROP_THRES, MLP_UPPER_THRES);
end

% =========================================================================
% 固定随机抽样
%
% 每次运行均产生完全相同的排列和样本子集。
% =========================================================================

rng(2026, 'twister');

shuffled_idx = idx_perfect(randperm(numel(idx_perfect)));

% 限制展示数量
MAX_SHOW = 80;

if numel(shuffled_idx) > MAX_SHOW
    shuffled_idx = shuffled_idx(1:MAX_SHOW);
end

% shuffled_idx 是 valid_mask 压缩后的索引。
% selected_plot_idx 是 pred_P_after / real_P_after 中的真实列索引。
selected_plot_idx = valid_original_idx(shuffled_idx);

% 同步抽取两种方法的数据
error_mlp_plot  = error_mlp_all(shuffled_idx);
error_prop_plot = error_prop_all(shuffled_idx);

% 明确保存 Proposed tip-error 数据，供后续分布图直接使用
tip_error_plot = error_prop_plot;
mlp_tip_error_plot = error_mlp_plot;

% 重新生成连续 X 轴
sample_indices = 1:numel(selected_plot_idx);

fprintf('   > Displayed filtered samples: %d\n', ...
    numel(sample_indices));

fprintf('   > Final selected original indices:\n      ');
fprintf('%d ', selected_plot_idx);
fprintf('\n');

% 验证当前绘图数据与原始数据完全一致
tip_error_check = ...
    error_prop_all_original(selected_plot_idx);

mlp_error_check = ...
    error_mlp_all_original(selected_plot_idx);

if any(abs(tip_error_check - tip_error_plot) > 1e-10)
    error('Internal alignment error in Proposed tip-error samples.');
end

if any(abs(mlp_error_check - mlp_tip_error_plot) > 1e-10)
    error('Internal alignment error in MLP tip-error samples.');
end

% =========================================================================
% 3. Statistics
% =========================================================================

mean_mlp  = mean(error_mlp_plot);
mean_prop = mean(error_prop_plot);

median_mlp  = median(error_mlp_plot);
median_prop = median(error_prop_plot);

rmse_mlp  = sqrt(mean(error_mlp_plot.^2));
rmse_prop = sqrt(mean(error_prop_plot.^2));

fprintf('\nTip-error statistics:\n');

fprintf(['   Vanilla MLP | Mean: %.3f mm | Median: %.3f mm | ', ...
         'RMSE: %.3f mm\n'], ...
    mean_mlp, median_mlp, rmse_mlp);

fprintf(['   Proposed    | Mean: %.3f mm | Median: %.3f mm | ', ...
         'RMSE: %.3f mm\n'], ...
    mean_prop, median_prop, rmse_prop);

% =========================================================================
% 4. Figure settings
% =========================================================================

export_this_figure = true;
output_folder = 'IEEE_MLP_vs_Proposed_Tip_Error';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

font_name = 'Times New Roman';

% Keep consistent with your qualitative figures
c_mlp  = [0, 158, 115] / 255;
c_prop = [0, 114, 189] / 255;

lw_mlp  = 2.5;
lw_prop = 3.2;

marker_size_mlp  = 5;
marker_size_prop = 6;

fig = figure( ...
    'Name', 'Vanilla MLP vs Proposed Tip Error', ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [80, 80, 1200, 620]);

ax = axes(fig);
hold(ax, 'on');
grid(ax, 'off');

% =========================================================================
% 5. Draw Vanilla MLP
% =========================================================================

h_mlp = plot( ...
    ax, ...
    sample_indices, ...
    error_mlp_plot, ...
    '-o', ...
    'Color', c_mlp, ...
    'LineWidth', lw_mlp, ...
    'MarkerSize', marker_size_mlp, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', c_mlp, ...
    'DisplayName', 'Vanilla MLP');

% =========================================================================
% 6. Draw Proposed
% =========================================================================

h_prop = plot( ...
    ax, ...
    sample_indices, ...
    error_prop_plot, ...
    '-o', ...
    'Color', c_prop, ...
    'LineWidth', lw_prop, ...
    'MarkerSize', marker_size_prop, ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', c_prop, ...
    'DisplayName', 'Proposed Method');

% =========================================================================
% 7. Mean lines
% =========================================================================

% h_mean_mlp = yline( ...
%     ax, ...
%     mean_mlp, ...
%     '--', ...
%     sprintf('MLP mean: %.2f mm', mean_mlp), ...
%     'Color', c_mlp, ...
%     'LineWidth', 1.8, ...
%     'FontName', font_name, ...
%     'FontSize', 18, ...
%     'LabelHorizontalAlignment', 'left', ...
%     'LabelVerticalAlignment', 'bottom');
%
% h_mean_prop = yline( ...
%     ax, ...
%     mean_prop, ...
%     '--', ...
%     sprintf('Proposed mean: %.2f mm', mean_prop), ...
%     'Color', c_prop, ...
%     'LineWidth', 1.8, ...
%     'FontName', font_name, ...
%     'FontSize', 18, ...
%     'LabelHorizontalAlignment', 'left', ...
%     'LabelVerticalAlignment', 'top');
%
% h_mean_mlp.Annotation.LegendInformation.IconDisplayStyle = 'off';
% h_mean_prop.Annotation.LegendInformation.IconDisplayStyle = 'off';

% =========================================================================
% 8. Axis formatting
% =========================================================================

% 始终保留坐标轴、刻度线和原有坐标范围
set(ax, ...
    'FontName', font_name, ...
    'FontSize', 24, ...
    'LineWidth', 1.5, ...
    'TickDir', 'out', ...
    'Box', 'off', ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'off');

if show_axis_labels
    xlabel( ...
        ax, ...
        'Test Sample Index', ...
        'FontName', font_name, ...
        'FontSize', 28);

    ylabel( ...
        ax, ...
        'Tip Reconstruction Error (mm)', ...
        'FontName', font_name, ...
        'FontSize', 28);
end

xlim(ax, [1, max(1, numel(sample_indices))]);

% Clean Y-axis range
max_error = max([error_mlp_plot, error_prop_plot]);

if max_error <= 10
    y_step = 2;
elseif max_error <= 30
    y_step = 5;
elseif max_error <= 60
    y_step = 10;
else
    y_step = 20;
end

y_upper = ceil(max_error / y_step) * y_step;

% 保留原代码中的固定上限
y_upper = 70;

if y_upper <= 0
    y_upper = y_step;
end

ylim(ax, [0, y_upper]);
yticks(ax, 0:y_step:y_upper);

% Clean X ticks
N_plot = numel(sample_indices);

if N_plot <= 10
    x_tick_step = 1;
elseif N_plot <= 30
    x_tick_step = 5;
elseif N_plot <= 80
    x_tick_step = 10;
else
    x_tick_step = 20;
end

xticks(ax, 1:x_tick_step:N_plot);

% 仅控制坐标轴数字，刻度位置与刻度线保持不变
if ~show_tick_labels
    ax.XTickLabel = [];
    ax.YTickLabel = [];
end

% =========================================================================
% 9. Legend
% =========================================================================

if show_legend
    legend( ...
        ax, ...
        [h_mlp, h_prop], ...
        {'Vanilla MLP', 'Proposed Method'}, ...
        'Location', 'northeast', ...
        'FontName', font_name, ...
        'FontSize', 22, ...
        'Box', 'on');

    title(ax, '');
end

% =========================================================================
% 10. Export
% =========================================================================

if export_this_figure
    exportgraphics( ...
        fig, ...
        fullfile(output_folder, ...
        'MLP_vs_Proposed_Tip_Error.pdf'), ...
        'ContentType', 'vector');

    exportgraphics( ...
        fig, ...
        fullfile(output_folder, ...
        'MLP_vs_Proposed_Tip_Error.png'), ...
        'Resolution', 600);

    savefig( ...
        fig, ...
        fullfile(output_folder, ...
        'MLP_vs_Proposed_Tip_Error.fig'));
end

disp('>>> Vanilla MLP vs. Proposed comparison figure generated successfully.');

%% ========================================================================
% Calculate Mean Shape Error for Every Final Evaluation Sample
% ========================================================================

N = size(pred_P_after, 2);

shape_error_all = zeros(1, N);

for i = 1:N

    Pg = reshape(real_P_after(:, i), 3, 7);
    Pp = reshape(pred_P_after(:, i), 3, 7);

    node_error = vecnorm(Pg - Pp, 2, 1);

    shape_error_all(i) = mean(node_error) * 1000;
end

% 严格检查 shape error 与原始 tip error 是否属于同一最终测试池
if numel(shape_error_all) ~= numel(error_prop_all_original)
    error(['Shape error contains %d samples, while original tip error ', ...
           'contains %d samples.'], ...
           numel(shape_error_all), ...
           numel(error_prop_all_original));
end

%% ============================================================
% Shape Error Statistics (same samples as Tip Error)
% ============================================================

% 必须使用 selected_plot_idx，而不是 shuffled_idx。
% selected_plot_idx 对应 pred_P_after / real_P_after 的真实列。
shape_error_plot = shape_error_all(selected_plot_idx);

mean_shape_error = mean(shape_error_plot);
median_shape_error = median(shape_error_plot);
std_shape_error = std(shape_error_plot);

fprintf('\n');
fprintf('================ Shape Error Statistics ================\n');
fprintf('Samples used        : %d\n', numel(shape_error_plot));
fprintf('Mean Shape Error    : %.6f mm\n', mean_shape_error);
fprintf('Median Shape Error  : %.6f mm\n', median_shape_error);
fprintf('Std Shape Error     : %.6f mm\n', std_shape_error);
fprintf('========================================================\n');

% 联合样本检查
if numel(shape_error_plot) ~= numel(tip_error_plot)
    error(['Selected shape-error count (%d) does not equal ', ...
           'selected tip-error count (%d).'], ...
           numel(shape_error_plot), ...
           numel(tip_error_plot));
end

fprintf('\n');
fprintf('================ Shared Sample Verification ================\n');
fprintf('Selected sample count : %d\n', numel(selected_plot_idx));
fprintf('Mean tip error        : %.6f mm\n', mean(tip_error_plot));
fprintf('Mean shape error      : %.6f mm\n', mean(shape_error_plot));
fprintf('============================================================\n');

%% ============================================================
% Tip Error Histogram
% ============================================================

figure( ...
    'Color', 'w', ...
    'Position', [120 120 700 620]);

histogram( ...
    tip_error_plot, ...
    'BinWidth', 0.2, ...
    'FaceColor', [0 158 115] / 255, ...
    'FaceAlpha', 0.45, ...
    'EdgeColor', [0 90 65] / 255, ...
    'LineWidth', 0.8);

hold on;

xline( ...
    mean(tip_error_plot), ...
    '--r', ...
    sprintf(''), ...
    'LineWidth', 2, ...
    'FontSize', 18, ...
    'FontName', 'Times New Roman', ...
    'LabelOrientation', 'aligned', ...
    'LabelVerticalAlignment', 'middle');
if show_axis_labels
    xlabel( ...
        'Tip Error (mm)', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 24);

    ylabel( ...
        'Error Count', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 24);
end

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 20, ...
    'LineWidth', 1.5, ...
    'Box', 'on', ...
    'TickDir', 'in', ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on');

% 仅隐藏坐标轴数字，保留轴线、主/次刻度线和网格
if ~show_tick_labels
    set(gca, ...
        'XTickLabel', [], ...
        'YTickLabel', []);
end

grid on;
grid minor;
yl = ylim(gca);
ylim([0 yl(2)*1.15]);
ax_tip_hist = gca;
ax_tip_hist.GridAlpha = 0.18;
ax_tip_hist.MinorGridAlpha = 0.08;

title('');

%% ============================================================
% Shape Error Histogram
% ============================================================

figure( ...
    'Color', 'w', ...
    'Position', [120 120 700 620]);

histogram( ...
    shape_error_plot, ...
    'BinWidth', 0.2, ...
    'FaceColor', [0 114 189] / 255, ...
    'FaceAlpha', 0.45, ...
    'EdgeColor', [0 70 130] / 255, ...
    'LineWidth', 0.8);

hold on;

% Mean line
xline( ...
    mean(shape_error_plot), ...
    '--r', ...
    sprintf(''), ...
    'LineWidth', 2, ...
    'FontSize', 18, ...
    'FontName', 'Times New Roman', ...
    'LabelOrientation', 'aligned', ...
    'LabelVerticalAlignment', 'middle');
if show_axis_labels
    xlabel( ...
        'Shape Error (mm)', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 24);

    ylabel( ...
        'Error Count', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 24);
end

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 20, ...
    'LineWidth', 1.5, ...
    'Box', 'on', ...
    'TickDir', 'in', ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on');

% 仅隐藏坐标轴数字，保留轴线、主/次刻度线和网格
if ~show_tick_labels
    set(gca, ...
        'XTickLabel', [], ...
        'YTickLabel', []);
end

grid on;
grid minor;
yl = ylim(gca);
ylim([0 yl(2)*1.15]);
ax = gca;
ax.GridAlpha = 0.18;
ax.MinorGridAlpha = 0.08;

title('');

%% ========================================================================
% Shape Error Histogram with Explicit Real Counts
% ========================================================================

% shape_error_plot 必须是未经 round / ceil / floor 的真实误差值，单位 mm
shape_error_plot = shape_error_plot(:);

% 删除 NaN 和 Inf，但不修改任何真实误差值
shape_error_plot = shape_error_plot(isfinite(shape_error_plot));

% -------------------------------------------------------------------------
% Explicit bin settings
% -------------------------------------------------------------------------

bin_width = 0.20;

data_min = min(shape_error_plot);
data_max = max(shape_error_plot);

% Bin 边界只负责划分区间，不会修改原始 error
first_edge = floor(data_min / bin_width) * bin_width;
last_edge  = ceil(data_max / bin_width) * bin_width;

bin_edges = first_edge : bin_width : last_edge;

% 防止最大值恰好落在最后边界之外
if bin_edges(end) <= data_max
    bin_edges(end + 1) = bin_edges(end) + bin_width;
end

% 真实计数
[counts, edges] = histcounts(shape_error_plot, bin_edges);

% 每根柱子的中心位置
bin_centers = edges(1:end-1) + diff(edges) / 2;

% -------------------------------------------------------------------------
% Verification in command window
% -------------------------------------------------------------------------

fprintf('\n================ Shape Error Histogram Counts ================\n');
fprintf('Total number of samples: %d\n', numel(shape_error_plot));
fprintf('Sum of histogram counts: %d\n\n', sum(counts));

for k = 1:numel(counts)
    fprintf('[%.3f, %.3f): %d samples\n', ...
        edges(k), edges(k + 1), counts(k));
end

fprintf('==============================================================\n');

% 检查所有样本是否都被统计
if sum(counts) ~= numel(shape_error_plot)
    warning(['Histogram count mismatch: %d values, ', ...
             'but only %d counted.'], ...
        numel(shape_error_plot), ...
        sum(counts));
end

% -------------------------------------------------------------------------
% Statistics
% -------------------------------------------------------------------------

mean_shape_error = mean(shape_error_plot);

fprintf('\nMean Shape Error: %.6f mm\n', mean_shape_error);

% -------------------------------------------------------------------------
% Plot
% -------------------------------------------------------------------------

fig = figure( ...
    'Name', 'Shape Error Distribution', ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [120, 120, 900, 650]);

ax = axes(fig);
hold(ax, 'on');

% 使用已经计算出的真实 count 画柱状图
bar( ...
    ax, ...
    bin_centers, ...
    counts, ...
    1.0, ...
    'FaceColor', [0, 114, 189] / 255, ...
    'FaceAlpha', 0.52, ...
    'EdgeColor', [0, 70, 130] / 255, ...
    'LineWidth', 1.0);

% Mean line
h_mean = xline( ...
    ax, ...
    mean_shape_error, ...
    '--', ...
    sprintf(''), ...
    'Color', [1, 0, 0], ...
    'LineWidth', 2.5, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 20, ...
    'LabelOrientation', 'aligned', ...
    'LabelVerticalAlignment', 'middle');

% -------------------------------------------------------------------------
% Axis formatting
% -------------------------------------------------------------------------

if show_axis_labels
    xlabel( ...
        ax, ...
        'Shape Error (mm)', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 28);

    ylabel( ...
        ax, ...
        'Error Count', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 28);
end

set(ax, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 24, ...
    'LineWidth', 1.5, ...
    'TickDir', 'in', ...
    'Box', 'on', ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on');

grid(ax, 'on');
ax.GridAlpha = 0.18;
ax.MinorGridAlpha = 0.07;

% X轴范围
xlim(ax, [edges(1), edges(end)]);

% X主刻度：根据范围自动选择好看的步长
x_range = edges(end) - edges(1);

if x_range <= 2
    x_tick_step = 0.2;
elseif x_range <= 5
    x_tick_step = 0.5;
else
    x_tick_step = 1.0;
end

x_tick_start = ceil(edges(1) / x_tick_step) * x_tick_step;
x_tick_end   = floor(edges(end) / x_tick_step) * x_tick_step;

xticks(ax, x_tick_start:x_tick_step:x_tick_end);

% Y轴必须是整数计数
max_count = max(counts);

if max_count <= 10
    y_tick_step = 1;
elseif max_count <= 20
    y_tick_step = 2;
elseif max_count <= 50
    y_tick_step = 5;
else
    y_tick_step = 10;
end

y_upper = ceil(max_count / y_tick_step) * y_tick_step;

% 在最高柱上方增加 15% 留白，使分布图视觉上更舒展
base_y_upper = max(y_tick_step, y_upper);
display_y_upper = base_y_upper * 1.15;

ylim(ax, [0, display_y_upper]);
yticks(ax, 0:y_tick_step:base_y_upper);

% 仅隐藏坐标轴数字，保留轴线、主/次刻度线和网格
if ~show_tick_labels
    ax.XTickLabel = [];
    ax.YTickLabel = [];
end

title(ax, '');

% -------------------------------------------------------------------------
% Export
% -------------------------------------------------------------------------

output_folder = 'IEEE_Error_Histograms';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

exportgraphics( ...
    fig, ...
    fullfile(output_folder, ...
    'Shape_Error_Distribution.pdf'), ...
    'ContentType', 'vector');

exportgraphics( ...
    fig, ...
    fullfile(output_folder, ...
    'Shape_Error_Distribution.png'), ...
    'Resolution', 600);

savefig( ...
    fig, ...
    fullfile(output_folder, ...
    'Shape_Error_Distribution.fig'));

disp('--------------------------------------------------');
disp('All plots and statistics use the same fixed selected samples.');
fprintf('Fixed random seed     : %d\n', 2026);
fprintf('Selected sample count : %d\n', numel(selected_plot_idx));
fprintf('Mean Proposed Tip     : %.6f mm\n', mean(tip_error_plot));
fprintf('Mean Proposed Shape   : %.6f mm\n', mean(shape_error_plot));
disp('--------------------------------------------------');