disp('--------------------------------------------------');
disp('9.33 Generating 6 separated right-half comparison figures...');
disp('      3 directions x 2 methods = 6 figures.');
disp('      Units are converted from m to mm for plotting.');
disp('      Axis limits and ticks are rounded to clean values.');
disp('      No subplot or tiledlayout is used.');

export_this_figure = true;
output_folder = 'IEEE_6_right_half_panels_mm_clean_ticks';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

n_interp = 160;

show_force_arrow = true;
show_contact_node = true;
show_grid = true;
show_origin_frame = true;
show_prior_context_for_nonprior = true;

% For final clean panels:
show_axis_labels = true;
show_tick_labels = true;

% For checking coordinate labels, temporarily use:
% show_axis_labels = true;
% show_tick_labels = true;

plot_unit_scale = 1000;     % m -> mm

force_scale = 20;           % mm
reverse_force_arrow = true;

origin_axis_len = 20;       % mm
origin_axis_lw  = 2.4;

use_manual_idx = true;
manual_selected_idx = [1, 6, 103];

font_name = 'Times New Roman';
fig_position = [120, 120, 560, 520];

if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing pred_P_after or real_P_after.');
end

if ~exist('pred_brute_abs', 'var')
    error('Missing pred_brute_abs.');
end

if ~exist('p_before_test', 'var')
    error('Missing p_before_test.');
end

if ~exist('v_idx', 'var')
    error('Missing v_idx.');
end

P_gt_all   = real_P_after;
P_prop_all = pred_P_after;
N_final = size(P_gt_all, 2);

if size(p_before_test, 2) == N_final
    P_prior_all = p_before_test;
elseif numel(v_idx) == size(p_before_test, 2)
    P_prior_all = p_before_test(:, v_idx);
else
    error('Cannot align p_before_test to final evaluation set.');
end

if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error('Cannot align pred_brute_abs to final evaluation set.');
end

contact_node_all = ones(1, N_final) * 4;

if exist('test_hgt', 'var') && numel(test_hgt) == N_final
    contact_node_all = round(test_hgt);
elseif exist('hgt_filtered', 'var') && numel(hgt_filtered) == N_final
    contact_node_all = round(hgt_filtered);
elseif exist('gt_hgt_test', 'var') && numel(gt_hgt_test) == numel(v_idx)
    contact_node_all = round(gt_hgt_test(v_idx));
end

contact_node_all = max(1, min(7, contact_node_all));

has_force = false;
F_gt_plot = [];

if exist('F_gt_test', 'var') && size(F_gt_test, 2) == N_final
    F_gt_plot = F_gt_test;
    has_force = true;
elseif exist('aug_gt_F', 'var') && exist('v_mask', 'var') && exist('test_idx', 'var') && exist('v_idx', 'var')
    tmp_F_gt = aug_gt_F(:, v_mask);
    tmp_F_gt = tmp_F_gt(:, test_idx);
    F_gt_plot = tmp_F_gt(:, v_idx);

    if size(F_gt_plot, 2) == N_final
        has_force = true;
    end
end

if ~has_force
    error('Cannot align ground-truth force.');
end

force_class = zeros(1, N_final);
force_nominal_vec = zeros(3, N_final);
R_restore_all = zeros(3, 3, N_final);

for i = 1:N_final
    [force_class(i), force_nominal_vec(:, i), R_restore_all(:, :, i)] = ...
        local_augmented_force_to_original_class_and_rotation(F_gt_plot(:, i));
end

if use_manual_idx
    selected_idx = manual_selected_idx(:)';

    if numel(selected_idx) ~= 3
        error('manual_selected_idx must contain exactly 3 indices.');
    end

    if any(selected_idx < 1) || any(selected_idx > N_final)
        error('manual_selected_idx contains index outside valid range.');
    end
else
    desired_classes = [0, 45, 90];
    selected_idx = zeros(1, 3);

    for kk = 1:3
        idx_pool = find(force_class == desired_classes(kk));
        if isempty(idx_pool)
            error('No sample found for %d deg.', desired_classes(kk));
        end
        selected_idx(kk) = idx_pool(1);
    end
end

selected_force_classes = force_class(selected_idx);

c_gt              = [0, 0, 0] / 255;
c_prior_context   = [145, 145, 145] / 255;

c_prop            = [0, 114, 189] / 255;
c_mlp             = [0, 158, 115] / 255;

c_gt_marker       = [52, 152, 219] / 255;
c_cmp_marker_fill = [255, 255, 255] / 255;

c_contact         = [213, 94, 0] / 255;
c_force           = [0, 140, 72] / 255;
c_tip_error       = [90, 90, 90] / 255;

lw_gt             = 3.0;
lw_cmp            = 2.8;
lw_prior_context  = 2.2;
lw_tip            = 2.2;

method_list = {'proposed', 'mlp'};
method_print_name = {'Proposed', 'Vanilla_MLP'};

use_global_axis_limits = true;

global_all_pts = [];

for col_tmp = 1:3
    idx_tmp = selected_idx(col_tmp);
    R_restore_tmp = R_restore_all(:, :, idx_tmp);

    Pg_tmp   = reshape(P_gt_all(:, idx_tmp),    3, 7);
    P0_tmp   = reshape(P_prior_all(:, idx_tmp), 3, 7);
    Ppr_tmp  = reshape(P_prop_all(:, idx_tmp),  3, 7);
    Pml_tmp  = reshape(P_mlp_all(:, idx_tmp),   3, 7);

    Pg_tmp   = Pg_tmp   - Pg_tmp(:, 1);
    P0_tmp   = P0_tmp   - P0_tmp(:, 1);
    Ppr_tmp  = Ppr_tmp  - Ppr_tmp(:, 1);
    Pml_tmp  = Pml_tmp  - Pml_tmp(:, 1);

    Pg_tmp   = R_restore_tmp * Pg_tmp;
    P0_tmp   = R_restore_tmp * P0_tmp;
    Ppr_tmp  = R_restore_tmp * Ppr_tmp;
    Pml_tmp  = R_restore_tmp * Pml_tmp;

    Pg_tmp   = plot_unit_scale * Pg_tmp;
    P0_tmp   = plot_unit_scale * P0_tmp;
    Ppr_tmp  = plot_unit_scale * Ppr_tmp;
    Pml_tmp  = plot_unit_scale * Pml_tmp;

    cnode_tmp = contact_node_all(idx_tmp);

    F_vis_tmp = force_nominal_vec(:, idx_tmp);
    if norm(F_vis_tmp) > eps
        F_vis_tmp = F_vis_tmp / norm(F_vis_tmp);
    end

    if reverse_force_arrow
        F_vis_tmp = -F_vis_tmp;
    end

    F0_tmp = Pg_tmp(:, cnode_tmp);

    global_all_pts = [global_all_pts, ...
        Pg_tmp, P0_tmp, Ppr_tmp, Pml_tmp, ...
        Pg_tmp(:, end), P0_tmp(:, end), Ppr_tmp(:, end), Pml_tmp(:, end), ...
        Pg_tmp(:, cnode_tmp), ...
        F0_tmp, F0_tmp + F_vis_tmp * force_scale, ...
        [0;0;0], ...
        [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];
end

% ======================= Clean rounded axis limits and ticks =======================
% ======================= Compact balanced axis limits and ticks =======================
% Goal:
%   1) compact view
%   2) each axis divided into exactly 3 segments
%   3) all tick labels are clean multiples of 10

global_pad_xy = 3;     % mm
global_pad_z  = 2;     % mm

x_raw = [min(global_all_pts(1,:))-global_pad_xy, max(global_all_pts(1,:))+global_pad_xy];
y_raw = [min(global_all_pts(2,:))-global_pad_xy, max(global_all_pts(2,:))+global_pad_xy];
z_raw = [min(global_all_pts(3,:))-global_pad_z,  max(global_all_pts(3,:))+global_pad_z];

% Three segments only. Each segment is an integer multiple of 10 mm.
segment_xy = 20;       % ticks like -40 -20 0 20
segment_z  = 40;       % ticks like 0 40 80 120

total_xy = 3 * segment_xy;
total_z  = 3 * segment_z;

% ---------- X axis ----------
x_center = mean(x_raw);
x_start = round((x_center - total_xy/2) / 10) * 10;
x_lim_global = [x_start, x_start + total_xy];

while x_lim_global(1) > x_raw(1) || x_lim_global(2) < x_raw(2)
    segment_xy = segment_xy + 10;
    total_xy = 3 * segment_xy;
    x_start = floor((x_center - total_xy/2) / 10) * 10;
    x_lim_global = [x_start, x_start + total_xy];
end

x_ticks_global = x_lim_global(1):segment_xy:x_lim_global(2);

% ---------- Y axis ----------
segment_y = 20;
total_y = 3 * segment_y;

y_center = mean(y_raw);
y_start = round((y_center - total_y/2) / 10) * 10;
y_lim_global = [y_start, y_start + total_y];

while y_lim_global(1) > y_raw(1) || y_lim_global(2) < y_raw(2)
    segment_y = segment_y + 10;
    total_y = 3 * segment_y;
    y_start = floor((y_center - total_y/2) / 10) * 10;
    y_lim_global = [y_start, y_start + total_y];
end

y_ticks_global = y_lim_global(1):segment_y:y_lim_global(2);

% ---------- Z axis ----------
% TDCR backbone is normalized from z = 0, so keep z lower limit at 0.
segment_z = 40;
z_lim_global = [0, 3 * segment_z];

while z_lim_global(2) < z_raw(2)
    segment_z = segment_z + 10;
    z_lim_global = [0, 3 * segment_z];
end

z_ticks_global = z_lim_global(1):segment_z:z_lim_global(2);

fprintf('\nCompact balanced axis limits in mm:\n');
fprintf('  X: [%g, %g], ticks = ', x_lim_global(1), x_lim_global(2));
fprintf('%g ', x_ticks_global);
fprintf('\n');

fprintf('  Y: [%g, %g], ticks = ', y_lim_global(1), y_lim_global(2));
fprintf('%g ', y_ticks_global);
fprintf('\n');

fprintf('  Z: [%g, %g], ticks = ', z_lim_global(1), z_lim_global(2));
fprintf('%g ', z_ticks_global);
fprintf('\n\n');
% =====================================================================================

fprintf('\nUnified rounded axis limits in mm:\n');
fprintf('  X: [%g, %g]\n', x_lim_global(1), x_lim_global(2));
fprintf('  Y: [%g, %g]\n', y_lim_global(1), y_lim_global(2));
fprintf('  Z: [%g, %g]\n\n', z_lim_global(1), z_lim_global(2));

disp(' ');
disp('==================== Separated 6-Figure Information ====================');
fprintf('Selected sample indices = [%d, %d, %d]\n', selected_idx(1), selected_idx(2), selected_idx(3));
fprintf('Nominal force classes   = [%d deg, %d deg, %d deg]\n', selected_force_classes(1), selected_force_classes(2), selected_force_classes(3));
disp('Units: mm.');
disp('Each figure contains: GT + current method + prior context + markers + contact node + force arrow + tip error line + origin frame.');
disp('No title / no legend. Axis labels and tick labels controlled by switches.');
disp('==========================================================================');
disp(' ');

for col = 1:3
    idx = selected_idx(col);
    cls = force_class(idx);

    R_restore = R_restore_all(:, :, idx);

    Pg_raw  = reshape(P_gt_all(:, idx),    3, 7);
    P0_raw  = reshape(P_prior_all(:, idx), 3, 7);
    Ppr_raw = reshape(P_prop_all(:, idx),  3, 7);
    Pml_raw = reshape(P_mlp_all(:, idx),   3, 7);

    Pg  = Pg_raw  - Pg_raw(:, 1);
    P0  = P0_raw  - P0_raw(:, 1);
    Ppr = Ppr_raw - Ppr_raw(:, 1);
    Pml = Pml_raw - Pml_raw(:, 1);

    Pg  = R_restore * Pg;
    P0  = R_restore * P0;
    Ppr = R_restore * Ppr;
    Pml = R_restore * Pml;

    Pg  = plot_unit_scale * Pg;
    P0  = plot_unit_scale * P0;
    Ppr = plot_unit_scale * Ppr;
    Pml = plot_unit_scale * Pml;

    cnode = contact_node_all(idx);

    F_vis = force_nominal_vec(:, idx);
    if norm(F_vis) > eps
        F_vis = F_vis / norm(F_vis);
    end

    if reverse_force_arrow
        F_vis = -F_vis;
    end

    t_nodes = 1:7;
    tq = linspace(1, 7, n_interp);

    Pg_s  = local_pchip_smooth(Pg,  t_nodes, tq);
    P0_s  = local_pchip_smooth(P0,  t_nodes, tq);
    Ppr_s = local_pchip_smooth(Ppr, t_nodes, tq);
    Pml_s = local_pchip_smooth(Pml, t_nodes, tq);

    for row = 1:2
        method_key = method_list{row};

        switch method_key
            case 'proposed'
                Pc = Ppr;
                Pc_s = Ppr_s;
                c_cmp = c_prop;
                line_style = '--';
                marker_style = 'o';

            case 'mlp'
                Pc = Pml;
                Pc_s = Pml_s;
                c_cmp = c_mlp;
                line_style = '-.';
                marker_style = 'o';
        end

        tip_err_mm = norm(Pc(:, end) - Pg(:, end));
        shape_err_mm = mean(vecnorm(Pc - Pg, 2, 1));

        fig_name = sprintf('dir_%03ddeg_%s_idx_%03d', cls, method_print_name{row}, idx);

        fig = figure('Name', fig_name, ...
            'Color', 'w', ...
            'Units', 'pixels', ...
            'Position', fig_position);

        ax = axes(fig);
        hold(ax, 'on');

        if show_grid
            grid(ax, 'on');
        else
            grid(ax, 'off');
        end

        axis(ax, 'equal');

        if show_prior_context_for_nonprior
            plot3(ax, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
                'Color', c_prior_context, ...
                'LineWidth', lw_prior_context);

            scatter3(ax, P0(1,:), P0(2,:), P0(3,:), 30, ...
                'Marker', 'o', ...
                'MarkerFaceColor', [1 1 1], ...
                'MarkerEdgeColor', c_prior_context, ...
                'LineWidth', 1.2);
        end

        plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
            'Color', c_gt, ...
            'LineWidth', lw_gt);

        plot3(ax, Pc_s(1,:), Pc_s(2,:), Pc_s(3,:), line_style, ...
            'Color', c_cmp, ...
            'LineWidth', lw_cmp);

        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 40, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c_gt_marker, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.8);

        scatter3(ax, Pc(1,:), Pc(2,:), Pc(3,:), 36, ...
            'Marker', marker_style, ...
            'MarkerFaceColor', c_cmp_marker_fill, ...
            'MarkerEdgeColor', c_cmp, ...
            'LineWidth', 1.0);

        scatter3(ax, Pg(1,end), Pg(2,end), Pg(3,end), 66, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c_gt, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.9);

        scatter3(ax, Pc(1,end), Pc(2,end), Pc(3,end), 66, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c_cmp, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.9);

        plot3(ax, [Pg(1,end), Pc(1,end)], ...
                  [Pg(2,end), Pc(2,end)], ...
                  [Pg(3,end), Pc(3,end)], ...
                  '--', ...
                  'Color', c_tip_error, ...
                  'LineWidth', lw_tip);

        if show_contact_node
            scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 86, ...
                'Marker', 'o', ...
                'MarkerFaceColor', c_contact, ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth', 1.1);
        end

        if show_force_arrow
            F0 = Pg(:, cnode);
            quiver3(ax, F0(1), F0(2), F0(3), ...
                F_vis(1)*force_scale, F_vis(2)*force_scale, F_vis(3)*force_scale, ...
                0, ...
                'Color', c_force, ...
                'LineWidth', 2.6, ...
                'MaxHeadSize', 0.9);
        end

        if show_origin_frame
            quiver3(ax, 0, 0, 0, origin_axis_len, 0, 0, ...
                0, 'Color', 'r', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

            quiver3(ax, 0, 0, 0, 0, origin_axis_len, 0, ...
                0, 'Color', 'g', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);

            quiver3(ax, 0, 0, 0, 0, 0, origin_axis_len, ...
                0, 'Color', 'b', 'LineWidth', origin_axis_lw, 'MaxHeadSize', 0.8);
        end

        set(ax, 'FontName', font_name, ...
            'FontSize', 13, ...
            'LineWidth', 1.2, ...
            'TickDir', 'out', ...
            'Box', 'off', ...
            'ZDir', 'reverse');

        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.ZGrid = 'on';
        ax.GridAlpha = 0.14;
        ax.MinorGridAlpha = 0.08;

        if show_axis_labels
            xlabel(ax, 'X (mm)', 'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold');
            ylabel(ax, 'Y (mm)', 'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold');
            zlabel(ax, 'Z (mm)', 'FontName', font_name, 'FontSize', 14, 'FontWeight', 'bold');
        else
            xlabel(ax, '');
            ylabel(ax, '');
            zlabel(ax, '');
        end

        view(ax, -40, 24);

        if use_global_axis_limits
            xlim(ax, x_lim_global);
            ylim(ax, y_lim_global);
            zlim(ax, z_lim_global);

            xticks(ax, x_ticks_global);
            yticks(ax, y_ticks_global);
            zticks(ax, z_ticks_global);
        else
            all_pts = [Pg, Pc, P0, Pg(:,end), Pc(:,end), Pg(:,cnode), [0;0;0]];

            if show_force_arrow
                all_pts = [all_pts, F0, F0 + F_vis * force_scale];
            end

            if show_origin_frame
                all_pts = [all_pts, ...
                    [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];
            end

            pad = 10;
            xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
            ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
            zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
        end

        if show_tick_labels
            ax.XTickLabelMode = 'auto';
            ax.YTickLabelMode = 'auto';
            ax.ZTickLabelMode = 'auto';
        else
            ax.XTickLabel = [];
            ax.YTickLabel = [];
            ax.ZTickLabel = [];
        end

        title(ax, '');
        legend(ax, 'off');

        fprintf('Direction %3d deg | idx %3d | %-12s | Tip %.2f mm | Shape %.2f mm\n', ...
            cls, idx, method_print_name{row}, tip_err_mm, shape_err_mm);

        if export_this_figure
            exportgraphics(fig, fullfile(output_folder, [fig_name, '.pdf']), 'ContentType', 'vector');
            exportgraphics(fig, fullfile(output_folder, [fig_name, '.png']), 'Resolution', 600);
            savefig(fig, fullfile(output_folder, [fig_name, '.fig']));
        end
    end
end

disp(' ');
disp(['>>> Exported 6 separated figures to folder: ', output_folder]);

function P_s = local_pchip_smooth(P, t_nodes, tq)
    P_s = zeros(3, numel(tq));
    for kk = 1:3
        P_s(kk, :) = interp1(t_nodes, P(kk, :), tq, 'pchip');
    end
end

function [force_class, F_nom, R_restore] = local_augmented_force_to_original_class_and_rotation(F_aug)
    F_aug = F_aug(:);

    if norm(F_aug) < 1e-10
        force_class = -1;
        F_nom = [0; 0; 0];
        R_restore = eye(3);
        return;
    end

    f = F_aug / norm(F_aug);

    base_dirs = [ ...
        -1,          -sqrt(2)/2,   0; ...
         0,           sqrt(2)/2,   1; ...
         0,           0,           0  ...
    ];

    base_labels = [0, 45, 90];

    R0 = eye(3);
    R120 = [cosd(120), -sind(120), 0; ...
            sind(120),  cosd(120), 0; ...
            0,          0,         1];

    R240 = [cosd(240), -sind(240), 0; ...
            sind(240),  cosd(240), 0; ...
            0,          0,         1];

    Rset = cat(3, R0, R120, R240);

    best_score = -inf;
    best_base_id = 1;
    best_rot_id = 1;

    for rr = 1:3
        for bb = 1:3
            cand = Rset(:, :, rr) * base_dirs(:, bb);
            cand = cand / norm(cand);

            score = dot(f, cand);

            if score > best_score
                best_score = score;
                best_base_id = bb;
                best_rot_id = rr;
            end
        end
    end

    force_class = base_labels(best_base_id);
    F_nom = base_dirs(:, best_base_id);

    R_aug = Rset(:, :, best_rot_id);
    R_restore = R_aug';
end