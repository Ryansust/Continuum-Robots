disp('--------------------------------------------------');
disp('9.31X Browsing candidate samples for 0/45/90 deg nominal force directions...');

topK_per_direction = 6;
n_interp = 120;
font_name = 'Times New Roman';

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
N_final = size(P_gt_all, 2);

if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error('Cannot align pred_brute_abs.');
end

if size(P_phys_after_all, 2) == N_final
    P_phys_all = P_phys_after_all;
elseif numel(v_idx) == size(P_phys_after_all, 2)
    P_phys_all = P_phys_after_all(:, v_idx);
else
    error('Cannot align P_phys_after_all.');
end

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

contact_node_all = ones(1, N_final) * 4;

if exist('test_hgt', 'var') && numel(test_hgt) == N_final
    contact_node_all = round(test_hgt);
elseif exist('hgt_filtered', 'var') && numel(hgt_filtered) == N_final
    contact_node_all = round(hgt_filtered);
elseif exist('gt_hgt_test', 'var') && numel(gt_hgt_test) == numel(v_idx)
    contact_node_all = round(gt_hgt_test(v_idx));
end

contact_node_all = max(1, min(7, contact_node_all));

shape_prop = zeros(1, N_final);
shape_mlp = zeros(1, N_final);
shape_phys = zeros(1, N_final);
tip_prop = zeros(1, N_final);
curvature_excess = zeros(1, N_final);
force_class = zeros(1, N_final);
force_nominal_vec = zeros(3, N_final);
R_restore_all = zeros(3, 3, N_final);

for i = 1:N_final
    [force_class(i), force_nominal_vec(:, i), R_restore_all(:,:,i)] = ...
        local_augmented_force_to_original_class_and_rotation(F_gt_plot(:, i));

    R_restore = R_restore_all(:,:,i);

    Pg  = reshape(P_gt_all(:, i),   3, 7);
    Ppr = reshape(P_prop_all(:, i), 3, 7);
    Pml = reshape(P_mlp_all(:, i),  3, 7);
    Pph = reshape(P_phys_all(:, i), 3, 7);

    Pg  = Pg  - Pg(:,1);
    Ppr = Ppr - Ppr(:,1);
    Pml = Pml - Pml(:,1);
    Pph = Pph - Pph(:,1);

    Pg  = R_restore * Pg;
    Ppr = R_restore * Ppr;
    Pml = R_restore * Pml;
    Pph = R_restore * Pph;

    tip_prop(i) = norm(Ppr(:,end) - Pg(:,end)) * 1000;
    shape_prop(i) = mean(vecnorm(Ppr - Pg, 2, 1)) * 1000;
    shape_mlp(i)  = mean(vecnorm(Pml - Pg, 2, 1)) * 1000;
    shape_phys(i) = mean(vecnorm(Pph - Pg, 2, 1)) * 1000;

    curv_gt = local_backbone_curvature(Pg);
    curv_prop = local_backbone_curvature(Ppr);
    curvature_excess(i) = curv_prop / max(curv_gt, eps);
end

desired_classes = [0, 45, 90];
candidate_indices = cell(1, 3);

for kk = 1:3
    cls = desired_classes(kk);
    idx_pool = find(force_class == cls);

    if isempty(idx_pool)
        error('No samples found for %d deg.', cls);
    end

    visually_safe = idx_pool( ...
        shape_prop(idx_pool) <= prctile(shape_prop, 80) & ...
        curvature_excess(idx_pool) <= 1.50);

    if isempty(visually_safe)
        visually_safe = idx_pool;
    end

    gap_mlp  = shape_mlp(visually_safe)  - shape_prop(visually_safe);
    gap_phys = shape_phys(visually_safe) - shape_prop(visually_safe);

    ratio_mlp  = shape_mlp(visually_safe)  ./ max(shape_prop(visually_safe), eps);
    ratio_phys = shape_phys(visually_safe) ./ max(shape_prop(visually_safe), eps);

    score = ...
        1.20 * gap_mlp + ...
        1.20 * gap_phys + ...
        0.80 * (ratio_mlp - 1) + ...
        0.80 * (ratio_phys - 1) - ...
        0.60 * shape_prop(visually_safe) - ...
        8.00 * max(curvature_excess(visually_safe) - 1.25, 0);

    [~, order] = sort(score, 'descend');
    candidate_indices{kk} = visually_safe(order(1:min(topK_per_direction, numel(order))));
end

disp(' ');
disp('Candidate indices by nominal force direction:');
for kk = 1:3
    fprintf('%2d deg candidates: ', desired_classes(kk));
    fprintf('%d ', candidate_indices{kk});
    fprintf('\n');
end
disp(' ');

fig = figure('Name', 'Candidate Browser: 0/45/90 deg', ...
    'Color', 'w', ...
    'Units', 'pixels', ...
    'Position', [60, 60, 1800, 900]);

tl = tiledlayout(fig, 3, topK_per_direction, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

c_gt    = [0.03 0.03 0.03];
c_prop  = [0.10 0.32 0.90];
c_mlp   = [0.20 0.60 0.25];
c_phys  = [0.85 0.20 0.12];
c_node  = [0.20 0.55 0.95];
c_ct    = [0.90 0.15 0.10];
c_fgt   = [0.10 0.65 0.15];

force_scale = 0.010;

for row = 1:3
    cls = desired_classes(row);
    idx_list = candidate_indices{row};

    for col = 1:topK_per_direction
        ax = nexttile(tl, (row-1)*topK_per_direction + col);
        hold(ax, 'on');
        grid(ax, 'on');
        axis(ax, 'equal');

        if col > numel(idx_list)
            axis(ax, 'off');
            continue;
        end

        idx = idx_list(col);
        R_restore = R_restore_all(:,:,idx);

        Pg  = reshape(P_gt_all(:, idx),   3, 7);
        Ppr = reshape(P_prop_all(:, idx), 3, 7);
        Pml = reshape(P_mlp_all(:, idx),  3, 7);
        Pph = reshape(P_phys_all(:, idx), 3, 7);

        Pg  = Pg  - Pg(:,1);
        Ppr = Ppr - Ppr(:,1);
        Pml = Pml - Pml(:,1);
        Pph = Pph - Pph(:,1);

        Pg  = R_restore * Pg;
        Ppr = R_restore * Ppr;
        Pml = R_restore * Pml;
        Pph = R_restore * Pph;

        tq = linspace(1, 7, n_interp);
        Pg_s  = local_pchip_smooth(Pg,  1:7, tq);
        Ppr_s = local_pchip_smooth(Ppr, 1:7, tq);
        Pml_s = local_pchip_smooth(Pml, 1:7, tq);
        Pph_s = local_pchip_smooth(Pph, 1:7, tq);

        plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-',  'Color', c_gt,   'LineWidth', 2.2);
        plot3(ax, Ppr_s(1,:), Ppr_s(2,:), Ppr_s(3,:), '--', 'Color', c_prop, 'LineWidth', 2.0);
        plot3(ax, Pml_s(1,:), Pml_s(2,:), Pml_s(3,:), '-.', 'Color', c_mlp,  'LineWidth', 1.6);
        plot3(ax, Pph_s(1,:), Pph_s(2,:), Pph_s(3,:), ':',  'Color', c_phys, 'LineWidth', 1.8);

        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 22, ...
            'MarkerFaceColor', c_node, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.5);

        cnode = contact_node_all(idx);
        scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 48, ...
            'MarkerFaceColor', c_ct, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.8);

        F_vis = force_nominal_vec(:, idx);
        F_vis = F_vis / max(norm(F_vis), eps);
        F_vis = -F_vis;

        F0 = Pg(:, cnode);
        quiver3(ax, F0(1), F0(2), F0(3), ...
            F_vis(1)*force_scale, F_vis(2)*force_scale, F_vis(3)*force_scale, ...
            0, 'Color', c_fgt, 'LineWidth', 1.8, 'MaxHeadSize', 0.8);

        set(ax, 'FontName', font_name, ...
            'FontSize', 9, ...
            'LineWidth', 0.8, ...
            'TickDir', 'out', ...
            'Box', 'off', ...
            'ZDir', 'reverse');

        view(ax, -40, 24);

        title(ax, sprintf('%d° | idx %d\nP %.1f / MLP %.1f / Phys %.1f', ...
            cls, idx, shape_prop(idx), shape_mlp(idx), shape_phys(idx)), ...
            'FontName', font_name, ...
            'FontSize', 9, ...
            'FontWeight', 'bold');

        all_pts = [Pg, Ppr, Pml, Pph, F0, F0 + F_vis * force_scale];

        pad = 0.010;
        xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
        ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
        zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
    end
end

disp('>>> Candidate browser generated.');
disp('Look at the figure and choose one good idx from each row.');
disp('Then set:');
fprintf('use_manual_idx = true;\nmanual_selected_idx = [idx_0deg, idx_45deg, idx_90deg];\n');


%% 

disp('--------------------------------------------------');
disp('9.32 Generating baseline comparison figure with original-frame restoration...');
disp('      Each panel corresponds to one original nominal force direction: 0, 45, 90 deg.');
disp('      Force and robot shapes are both restored from augmented frame to original frame.');
disp('      Only reversed nominal GT force direction is shown. No predicted force is drawn.');

export_this_figure = true;
fig_name = 'IEEE_baseline_comparison_original_frame_0_45_90_force_reverse_small';

n_interp = 160;

show_node_markers = true;
show_contact_node = true;
show_prior_curve  = true;
show_force_arrow  = true;

force_scale = 0.010;

origin_axis_len = 0.020;
origin_axis_lw  = 2.2;

use_manual_idx = true;
manual_selected_idx = [76, 119, 103];

font_name = 'Times New Roman';

if ~exist('pred_P_after', 'var') || ~exist('real_P_after', 'var')
    error('Missing proposed / ground-truth variables: pred_P_after and real_P_after.');
end

if ~exist('pred_brute_abs', 'var')
    error('Missing Vanilla MLP result variable: pred_brute_abs. Please run the brute-force MLP baseline first.');
end

if ~exist('P_phys_after_all', 'var')
    error('Missing analytical physics result variable: P_phys_after_all. Please run the analytical physics baseline first.');
end

if ~exist('v_idx', 'var')
    error('Missing v_idx. Baseline outputs need v_idx for final alignment.');
end

P_gt_all   = real_P_after;
P_prop_all = pred_P_after;
N_final = size(P_gt_all, 2);

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

if size(pred_brute_abs, 2) == N_final
    P_mlp_all = pred_brute_abs;
elseif numel(v_idx) == size(pred_brute_abs, 2)
    P_mlp_all = pred_brute_abs(:, v_idx);
else
    error('Cannot align Vanilla MLP result to real_P_after.');
end

if size(P_phys_after_all, 2) == N_final
    P_phys_all = P_phys_after_all;
elseif numel(v_idx) == size(P_phys_after_all, 2)
    P_phys_all = P_phys_after_all(:, v_idx);
else
    error('Cannot align analytical physics result to real_P_after.');
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
    error('Ground-truth force variable is missing or cannot be aligned. Cannot classify force direction.');
end

tip_prop = zeros(1, N_final);
shape_prop = zeros(1, N_final);
tip_mlp = zeros(1, N_final);
shape_mlp = zeros(1, N_final);
tip_phys = zeros(1, N_final);
shape_phys = zeros(1, N_final);
curvature_excess = zeros(1, N_final);
force_class = zeros(1, N_final);
force_nominal_vec = zeros(3, N_final);
R_restore_all = zeros(3, 3, N_final);

for i = 1:N_final
    [force_class(i), force_nominal_vec(:, i), R_restore_all(:, :, i)] = ...
        local_augmented_force_to_original_class_and_rotation(F_gt_plot(:, i));

    R_restore = R_restore_all(:, :, i);

    Pg  = reshape(P_gt_all(:, i),   3, 7);
    Ppr = reshape(P_prop_all(:, i), 3, 7);
    Pml = reshape(P_mlp_all(:, i),  3, 7);
    Pph = reshape(P_phys_all(:, i), 3, 7);

    Pg  = Pg  - Pg(:, 1);
    Ppr = Ppr - Ppr(:, 1);
    Pml = Pml - Pml(:, 1);
    Pph = Pph - Pph(:, 1);

    Pg  = R_restore * Pg;
    Ppr = R_restore * Ppr;
    Pml = R_restore * Pml;
    Pph = R_restore * Pph;

    [tip_prop(i), shape_prop(i)] = local_err_metric(Ppr, Pg);
    [tip_mlp(i),  shape_mlp(i)]  = local_err_metric(Pml, Pg);
    [tip_phys(i), shape_phys(i)] = local_err_metric(Pph, Pg);

    curv_gt = local_backbone_curvature(Pg);
    curv_prop = local_backbone_curvature(Ppr);
    curvature_excess(i) = curv_prop / max(curv_gt, eps);
end

if use_manual_idx
    selected_idx = manual_selected_idx(:)';

    if numel(selected_idx) ~= 3
        error('manual_selected_idx must contain exactly 3 indices.');
    end

    if any(selected_idx < 1) || any(selected_idx > N_final)
        error('manual_selected_idx contains index outside valid range 1:N_final.');
    end

    selected_force_classes = force_class(selected_idx);

    disp('>>> Manual selected_idx is used.');
else
    selected_idx = zeros(1, 3);
    desired_classes = [0, 45, 90];

    for kk = 1:3
        cls = desired_classes(kk);

        idx_pool = find(force_class == cls);

        if isempty(idx_pool)
            error('No sample found for nominal force direction %d deg.', cls);
        end

        valid_pool = idx_pool( ...
            shape_prop(idx_pool) <= prctile(shape_prop, 75) & ...
            curvature_excess(idx_pool) <= 1.50);

        if isempty(valid_pool)
            warning('No visually safe sample for %d deg. Relaxing visual filter.', cls);
            valid_pool = idx_pool(shape_prop(idx_pool) <= prctile(shape_prop, 85));
        end

        if isempty(valid_pool)
            warning('Still no filtered sample for %d deg. Using all samples in this class.', cls);
            valid_pool = idx_pool;
        end

        gap_mlp  = shape_mlp(valid_pool)  - shape_prop(valid_pool);
        gap_phys = shape_phys(valid_pool) - shape_prop(valid_pool);

        ratio_mlp  = shape_mlp(valid_pool)  ./ max(shape_prop(valid_pool), eps);
        ratio_phys = shape_phys(valid_pool) ./ max(shape_prop(valid_pool), eps);

        score = ...
            1.20 * gap_mlp + ...
            1.20 * gap_phys + ...
            0.80 * (ratio_mlp - 1) + ...
            0.80 * (ratio_phys - 1) - ...
            0.65 * shape_prop(valid_pool) - ...
            8.00 * max(curvature_excess(valid_pool) - 1.25, 0);

        [~, best_local] = max(score);
        selected_idx(kk) = valid_pool(best_local);
    end

    selected_force_classes = desired_classes;

    disp('>>> Automatic one-sample-per-force-direction selection is used.');
end

c_gt    = [0.03 0.03 0.03];
c_prop  = [0.10 0.32 0.90];
c_mlp   = [0.20 0.60 0.25];
c_phys  = [0.85 0.20 0.12];
c_prior = [0.72 0.72 0.72];

c_node  = [0.20 0.55 0.95];
c_ct    = [0.90 0.15 0.10];
c_fgt   = [0.10 0.65 0.15];

lw_gt    = 2.6;
lw_prop  = 2.4;
lw_mlp   = 2.0;
lw_phys  = 2.0;
lw_prior = 1.25;

disp(' ');
disp('==================== Baseline Comparison: Restored Original Frame ====================');
disp('No legend, title, or metric boxes are shown in the figure.');
disp('Force and all robot shapes are restored from augmented frame to original frame.');
disp('Force arrows are reversed and scaled down by 10x.');
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
disp('  Green arrow             : Reversed original nominal GT contact force direction');
disp(' ');
fprintf('Selected sample indices = [%d, %d, %d]\n', selected_idx(1), selected_idx(2), selected_idx(3));
fprintf('Nominal force classes   = [%d deg, %d deg, %d deg]\n', selected_force_classes(1), selected_force_classes(2), selected_force_classes(3));
disp(' ');
disp('Per-sample errors after Node-1 origin alignment and original-frame restoration:');
disp('  Direction | Sample | Method              | Tip Error [mm] | Mean Shape Error [mm] | CurvEx');

for pp = 1:3
    idx = selected_idx(pp);
    fprintf('  %8d | %6d | Proposed           | %14.2f | %21.2f | %.2f\n', ...
        force_class(idx), idx, tip_prop(idx), shape_prop(idx), curvature_excess(idx));
    fprintf('  %8d | %6d | Vanilla MLP        | %14.2f | %21.2f | %.2f\n', ...
        force_class(idx), idx, tip_mlp(idx), shape_mlp(idx), curvature_excess(idx));
    fprintf('  %8d | %6d | Analytical Physics | %14.2f | %21.2f | %.2f\n', ...
        force_class(idx), idx, tip_phys(idx), shape_phys(idx), curvature_excess(idx));
    fprintf('           |        | Raw augmented F_gt  | [% .3f, % .3f, % .3f]\n', ...
        F_gt_plot(1,idx), F_gt_plot(2,idx), F_gt_plot(3,idx));
    fprintf('           |        | Nominal F before reverse | [% .3f, % .3f, % .3f]\n', ...
        force_nominal_vec(1,idx), force_nominal_vec(2,idx), force_nominal_vec(3,idx));
    fprintf('           |        | Display F after reverse  | [% .3f, % .3f, % .3f]\n', ...
        -force_nominal_vec(1,idx), -force_nominal_vec(2,idx), -force_nominal_vec(3,idx));
end

disp('=====================================================================================');
disp(' ');

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

    R_restore = R_restore_all(:, :, idx);

    Pg_raw  = reshape(P_gt_all(:, idx),   3, 7);
    Ppr_raw = reshape(P_prop_all(:, idx), 3, 7);
    Pml_raw = reshape(P_mlp_all(:, idx),  3, 7);
    Pph_raw = reshape(P_phys_all(:, idx), 3, 7);

    Pg  = Pg_raw  - Pg_raw(:, 1);
    Ppr = Ppr_raw - Ppr_raw(:, 1);
    Pml = Pml_raw - Pml_raw(:, 1);
    Pph = Pph_raw - Pph_raw(:, 1);

    Pg  = R_restore * Pg;
    Ppr = R_restore * Ppr;
    Pml = R_restore * Pml;
    Pph = R_restore * Pph;

    if show_prior_curve
        P0_raw = reshape(P_prior_all(:, idx), 3, 7);
        P0 = P0_raw - P0_raw(:, 1);
        P0 = R_restore * P0;
    end

    t_nodes = 1:7;
    tq = linspace(1, 7, n_interp);

    Pg_s  = local_pchip_smooth(Pg,  t_nodes, tq);
    Ppr_s = local_pchip_smooth(Ppr, t_nodes, tq);
    Pml_s = local_pchip_smooth(Pml, t_nodes, tq);
    Pph_s = local_pchip_smooth(Pph, t_nodes, tq);

    if show_prior_curve
        P0_s = local_pchip_smooth(P0, t_nodes, tq);
    end

    if show_prior_curve
        plot3(ax, P0_s(1,:), P0_s(2,:), P0_s(3,:), '-.', ...
            'Color', c_prior, 'LineWidth', lw_prior);
    end

    plot3(ax, Pg_s(1,:), Pg_s(2,:), Pg_s(3,:), '-', ...
        'Color', c_gt, 'LineWidth', lw_gt);

    plot3(ax, Ppr_s(1,:), Ppr_s(2,:), Ppr_s(3,:), '--', ...
        'Color', c_prop, 'LineWidth', lw_prop);

    plot3(ax, Pml_s(1,:), Pml_s(2,:), Pml_s(3,:), '-.', ...
        'Color', c_mlp, 'LineWidth', lw_mlp);

    plot3(ax, Pph_s(1,:), Pph_s(2,:), Pph_s(3,:), ':', ...
        'Color', c_phys, 'LineWidth', lw_phys);

    if show_node_markers
        scatter3(ax, Pg(1,:), Pg(2,:), Pg(3,:), 32, ...
            'MarkerFaceColor', c_node, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.6);
    end

    cnode = contact_node_all(idx);

    if show_contact_node
        scatter3(ax, Pg(1,cnode), Pg(2,cnode), Pg(3,cnode), 70, ...
            'MarkerFaceColor', c_ct, ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.1);
    end

    if show_force_arrow
        F_vis = force_nominal_vec(:, idx);

        if norm(F_vis) > eps
            F_vis = F_vis / norm(F_vis);
        end

        F_vis = -F_vis;

        F0_gt = Pg(:, cnode);

        quiver3(ax, F0_gt(1), F0_gt(2), F0_gt(3), ...
            F_vis(1)*force_scale, F_vis(2)*force_scale, F_vis(3)*force_scale, ...
            0, 'Color', c_fgt, 'LineWidth', 2.2, ...
            'MaxHeadSize', 0.9);
    end

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

    all_pts = [Pg, Ppr, Pml, Pph, [0;0;0], ...
               [origin_axis_len;0;0], [0;origin_axis_len;0], [0;0;origin_axis_len]];

    if show_prior_curve
        all_pts = [all_pts, P0];
    end

    if show_force_arrow
        all_pts = [all_pts, F0_gt, F0_gt + F_vis * force_scale];
    end

    pad = 0.010;
    xlim(ax, [min(all_pts(1,:))-pad, max(all_pts(1,:))+pad]);
    ylim(ax, [min(all_pts(2,:))-pad, max(all_pts(2,:))+pad]);
    zlim(ax, [min(all_pts(3,:))-pad, max(all_pts(3,:))+pad]);
end

disp('>>> Baseline comparison figure with reversed small force arrows generated successfully.');

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

function curv_val = local_backbone_curvature(P)
    X = P';
    Xc = X - mean(X, 1);
    [~, ~, V] = svd(Xc, 'econ');
    q = Xc * V(:, 1:2);

    kappa = zeros(1, 5);
    for jj = 2:6
        v1 = q(jj, :)   - q(jj-1, :);
        v2 = q(jj+1, :) - q(jj, :);

        cross_z = v1(1)*v2(2) - v1(2)*v2(1);
        denom = norm(v1) * norm(v2) + eps;

        kappa(jj-1) = abs(cross_z / denom);
    end

    curv_val = sum(kappa);
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