%% ========================================================================
% Step 9.34: Joint Visualization of Shape, Contact Location, and Force
% ========================================================================

disp('--------------------------------------------------');
disp('9.34 Generating joint interaction-and-shape comparison figure...');
disp('      Compare GT vs prediction for:');
disp('      1) post-contact backbone shape');
disp('      2) contact disk location');
disp('      3) contact force vector');

export_this_figure = true;
output_folder = 'IEEE_joint_shape_location_force';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% ========================================================================
% User settings
% ========================================================================

% Index in the FINAL filtered evaluation set:
% Change this to 1, 6, or 103 according to the case you want to show.
selected_case_idx = 6;

plot_unit_scale = 1000;      % m -> mm
n_interp = 160;

font_name = 'Times New Roman';
fig_position = [120, 120, 620, 570];

show_grid = true;
show_axis_labels = false;
show_tick_labels = false;
show_origin_frame = true;
showlegend = false;
view_azimuth = 130;
view_elevation = 12;
plot_box_aspect = [1 1 2.1];

% Force-vector visualization scale
% Arrow length in the figure = force magnitude [N] × this value [mm/N]
force_display_scale = 200;    % mm/N

% Keep consistent with the previous qualitative plots
reverse_force_arrow = true;

origin_axis_len = 27;        % mm
origin_axis_lw = 5;

% ========================================================================
% Colors and styles
% ========================================================================

c_shape_gt       = [0, 0, 0] / 255;
c_shape_pred     = [0, 114, 189] / 255;

% c_marker_gt      = [52, 152, 219] / 255;
c_marker_gt      = [255, 255, 255] / 255;
c_marker_pred    = [255, 255, 255] / 255;

c_contact_gt     = [213, 94, 0] / 255;
c_contact_pred   = [0, 0, 255] / 255;

% c_force_gt       = [0, 120, 70] / 255;
% c_force_pred     = [120, 40, 180] / 255;
c_force_gt       = [0, 0, 0] / 255;
c_force_pred     = [0, 114, 189] / 255;

c_location_error = [120, 120, 120] / 255;

lw_shape_gt      = 5;
lw_shape_pred    = 5;
lw_force_gt      = 5;
lw_force_pred    = 5;
lw_location_err  = 3;

ms_shape_node    = 110;
ms_shape_tip     = 226;
ms_contact_gt    = 260;
ms_contact_pred  = 260;

% ========================================================================
% Required-variable checks
% ========================================================================

required_vars = { ...
    'real_P_after', ...
    'pred_P_after', ...
    'v_mask', ...
    'test_idx', ...
    'v_idx', ...
    'aug_gt_F', ...
    'pred_force_all', ...
    'raw_tg', ...
    'pred_loc_norm_all'};

for kk = 1:numel(required_vars)
    if ~exist(required_vars{kk}, 'var')
        error('Missing required variable: %s', required_vars{kk});
    end
end

N_final = size(real_P_after, 2);

if selected_case_idx < 1 || selected_case_idx > N_final
    error('selected_case_idx must be between 1 and %d.', N_final);
end

% ========================================================================
% Align Net-B outputs and ground truth with the FINAL evaluation set
%
% Data alignment:
% all augmented samples
%       -> v_mask
%       -> test_idx
%       -> v_idx
% ========================================================================

% ---------- Ground-truth force ----------
F_gt_all = aug_gt_F(:, v_mask);
F_gt_test = F_gt_all(:, test_idx);
F_gt_final = F_gt_test(:, v_idx);

% ---------- Predicted force ----------
F_pred_all = pred_force_all(:, v_mask);
F_pred_test = F_pred_all(:, test_idx);
F_pred_final = F_pred_test(:, v_idx);

% ---------- Ground-truth contact disk ----------
location_gt_all = raw_tg * 9.0;
location_gt_test = location_gt_all(:, test_idx);
location_gt_final = location_gt_test(:, v_idx);

% ---------- Predicted contact disk ----------
location_pred_all = pred_loc_norm_all * 9.0;
location_pred_test = location_pred_all(:, test_idx);
location_pred_final = location_pred_test(:, v_idx);

% Match the location evaluation convention used in Step 6
location_pred_final(location_pred_final < 3) = 3;
location_pred_final(location_pred_final > 5) = 5;

if size(F_gt_final, 2) ~= N_final
    error('GT force alignment failed: expected %d samples, obtained %d.', ...
        N_final, size(F_gt_final, 2));
end

if size(F_pred_final, 2) ~= N_final
    error('Predicted-force alignment failed: expected %d samples, obtained %d.', ...
        N_final, size(F_pred_final, 2));
end

if numel(location_gt_final) ~= N_final
    error('GT-location alignment failed.');
end

if numel(location_pred_final) ~= N_final
    error('Predicted-location alignment failed.');
end

% ========================================================================
% Extract selected case
% ========================================================================

idx = selected_case_idx;

Pg_raw = reshape(real_P_after(:, idx), 3, 7);
Pp_raw = reshape(pred_P_after(:, idx), 3, 7);

% Use the first stored node as the common plotting origin
Pg = Pg_raw - Pg_raw(:, 1);
Pp = Pp_raw - Pp_raw(:, 1);

F_gt_aug = F_gt_final(:, idx);
F_pred_aug = F_pred_final(:, idx);

disk_gt = location_gt_final(idx);
disk_pred = location_pred_final(idx);

% ========================================================================
% Restore augmented sample to its original nominal direction
% ========================================================================

[force_class, ~, R_restore] = ...
    local_augmented_force_to_original_class_and_rotation(F_gt_aug);

Pg = R_restore * Pg;
Pp = R_restore * Pp;

F_gt = R_restore * F_gt_aug;
F_pred = R_restore * F_pred_aug;

% Convert shape coordinates from m to mm
Pg = plot_unit_scale * Pg;
Pp = plot_unit_scale * Pp;

% Keep the same displayed-force convention as your previous figures
if reverse_force_arrow
    F_gt = -F_gt;
    F_pred = -F_pred;
end

% ========================================================================
% PCHIP backbone smoothing
% ========================================================================

t_nodes = 1:7;
tq = linspace(1, 7, n_interp);

Pg_s = zeros(3, numel(tq));
Pp_s = zeros(3, numel(tq));

for dim_id = 1:3
    Pg_s(dim_id, :) = interp1(t_nodes, Pg(dim_id, :), tq, 'pchip');
    Pp_s(dim_id, :) = interp1(t_nodes, Pp(dim_id, :), tq, 'pchip');
end

% ========================================================================
% Map continuous disk positions to the plotted backbone
%
% Physical marker arrangement from bottom to top:
% Disk 1, 3, 5, 7, 9, 11, 13
%
% Stored plotting order is reversed:
% P(:,1) -> Disk13
% P(:,2) -> Disk11
% ...
% P(:,7) -> Disk1
% ========================================================================

contact_gt = local_get_continuous_disk_position(Pg, disk_gt);
contact_pred = local_get_continuous_disk_position(Pp, disk_pred);

% Force arrows begin from their corresponding contact locations
force_tip_gt = contact_gt + F_gt * force_display_scale;
force_tip_pred = contact_pred + F_pred * force_display_scale;

% ========================================================================
% Metrics for command-window reporting
% ========================================================================

tip_error_mm = norm(Pp(:, end) - Pg(:, end));
shape_error_mm = mean(vecnorm(Pp - Pg, 2, 1));

location_error_disk = abs(disk_pred - disk_gt);

force_vector_error_N = norm(F_pred_aug - F_gt_aug);
force_magnitude_gt_N = norm(F_gt_aug);
force_magnitude_pred_N = norm(F_pred_aug);

if force_magnitude_gt_N > eps && force_magnitude_pred_N > eps
    force_angle_error_deg = acosd(max(-1, min(1, ...
        dot(F_gt_aug, F_pred_aug) / ...
        (force_magnitude_gt_N * force_magnitude_pred_N))));
else
    force_angle_error_deg = NaN;
end

% ========================================================================
% Compact axis limits
% ========================================================================

all_pts = [ ...
    Pg, ...
    Pp, ...
    contact_gt, ...
    contact_pred, ...
    force_tip_gt, ...
    force_tip_pred, ...
    [0; 0; 0], ...
    [origin_axis_len; 0; 0], ...
    [0; origin_axis_len; 0], ...
    [0; 0; origin_axis_len]];

global_pad_xy = 3;
global_pad_z = 2;

x_raw = [min(all_pts(1, :)) - global_pad_xy, ...
         max(all_pts(1, :)) + global_pad_xy];

y_raw = [min(all_pts(2, :)) - global_pad_xy, ...
         max(all_pts(2, :)) + global_pad_xy];

z_raw = [min(all_pts(3, :)) - global_pad_z, ...
         max(all_pts(3, :)) + global_pad_z];

% ---------- X axis: exactly three clean segments ----------
segment_x = 10;
x_center = mean(x_raw);

while true
    total_x = 3 * segment_x;
    x_start = floor((x_center - total_x / 2) / 10) * 10;
    x_lim = [x_start, x_start + total_x];

    if x_lim(1) <= x_raw(1) && x_lim(2) >= x_raw(2)
        break;
    end

    segment_x = segment_x + 10;
end

x_ticks = x_lim(1):segment_x:x_lim(2);

% ---------- Y axis: exactly three clean segments ----------
segment_y = 10;
y_center = mean(y_raw);

while true
    total_y = 3 * segment_y;
    y_start = floor((y_center - total_y / 2) / 10) * 10;
    y_lim = [y_start, y_start + total_y];

    if y_lim(1) <= y_raw(1) && y_lim(2) >= y_raw(2)
        break;
    end

    segment_y = segment_y + 10;
end

y_ticks = y_lim(1):segment_y:y_lim(2);

% ---------- Z axis: exactly three clean segments ----------
segment_z = 30;
z_lim = [0, 3 * segment_z];

while z_lim(2) < z_raw(2)
    segment_z = segment_z + 10;
    z_lim = [0, 3 * segment_z];
end

z_ticks = z_lim(1):segment_z:z_lim(2);

% ========================================================================
% Draw figure
% ========================================================================

fig_name = sprintf('joint_case_%03d_dir_%03ddeg', idx, force_class);

fig = figure( ...
    'Name', fig_name, ...
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
pbaspect(ax, plot_box_aspect);

% ---------- Ground-truth shape ----------
h_shape_gt = plot3( ...
    ax, ...
    Pg_s(1, :), ...
    Pg_s(2, :), ...
    Pg_s(3, :), ...
    '-', ...
    'Color', c_shape_gt, ...
    'LineWidth', lw_shape_gt);

scatter3( ...
    ax, ...
    Pg(1, :), ...
    Pg(2, :), ...
    Pg(3, :), ...
    ms_shape_node, ...
    'Marker', 'o', ...
    'MarkerFaceColor', c_marker_gt, ...
    'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);

% ---------- Predicted shape ----------
h_shape_pred = plot3( ...
    ax, ...
    Pp_s(1, :), ...
    Pp_s(2, :), ...
    Pp_s(3, :), ...
    '-', ...
    'Color', c_shape_pred, ...
    'LineWidth', lw_shape_pred);

scatter3( ...
    ax, ...
    Pp(1, :), ...
    Pp(2, :), ...
    Pp(3, :), ...
    ms_shape_node, ...
    'Marker', 'o', ...
    'MarkerFaceColor', c_marker_pred, ...
    'MarkerEdgeColor', c_shape_pred, ...
    'LineWidth', 2);

% ---------- Tip markers ----------
scatter3( ...
    ax, ...
    Pg(1, end), ...
    Pg(2, end), ...
    Pg(3, end), ...
    ms_shape_tip, ...
    'Marker', 'o', ...
    'MarkerFaceColor', c_shape_gt, ...
    'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);

scatter3( ...
    ax, ...
    Pp(1, end), ...
    Pp(2, end), ...
    Pp(3, end), ...
    ms_shape_tip, ...
    'Marker', 'o', ...
    'MarkerFaceColor', c_shape_pred, ...
    'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);

% ---------- Tip-error connector ----------
plot3( ...
    ax, ...
    [Pg(1, end), Pp(1, end)], ...
    [Pg(2, end), Pp(2, end)], ...
    [Pg(3, end), Pp(3, end)], ...
    '-', ...
    'Color', c_location_error, ...
    'LineWidth', 3);

% ---------- Ground-truth contact ----------
h_contact_gt = scatter3( ...
    ax, ...
    contact_gt(1), ...
    contact_gt(2), ...
    contact_gt(3), ...
    ms_contact_gt, ...
    'Marker', 'o', ...
    'MarkerFaceColor', c_contact_gt, ...
    'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);

% ---------- Predicted contact ----------
h_contact_pred = scatter3( ...
    ax, ...
    contact_pred(1), ...
    contact_pred(2), ...
    contact_pred(3), ...
    ms_contact_pred, ...
    'Marker', 'o', ...
    'MarkerFaceColor', c_contact_pred, ...
    'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);

% ---------- Location-error connector ----------
plot3( ...
    ax, ...
    [contact_gt(1), contact_pred(1)], ...
    [contact_gt(2), contact_pred(2)], ...
    [contact_gt(3), contact_pred(3)], ...
    ':', ...
    'Color', c_location_error, ...
    'LineWidth', lw_location_err);

% ---------- Ground-truth force ----------
h_force_gt = quiver3( ...
    ax, ...
    contact_gt(1), ...
    contact_gt(2), ...
    contact_gt(3), ...
    F_gt(1) * force_display_scale, ...
    F_gt(2) * force_display_scale, ...
    F_gt(3) * force_display_scale, ...
    0, ...
    'Color', c_force_gt, ...
    'LineWidth', lw_force_gt, ...
    'MaxHeadSize', 2);

% ---------- Predicted force ----------
h_force_pred = quiver3( ...
    ax, ...
    contact_pred(1), ...
    contact_pred(2), ...
    contact_pred(3), ...
    F_pred(1) * force_display_scale, ...
    F_pred(2) * force_display_scale, ...
    F_pred(3) * force_display_scale, ...
    0, ...
    'Color', c_force_pred, ...
    'LineWidth', lw_force_pred, ...
    'MaxHeadSize', 2);

% ---------- Robot-base coordinate frame ----------
if show_origin_frame
    quiver3( ...
        ax, 0, 0, 0, ...
        origin_axis_len, 0, 0, ...
        0, ...
        'Color', 'r', ...
        'LineWidth', origin_axis_lw, ...
        'MaxHeadSize', 2);

    quiver3( ...
        ax, 0, 0, 0, ...
        0, origin_axis_len, 0, ...
        0, ...
        'Color', 'g', ...
        'LineWidth', origin_axis_lw, ...
        'MaxHeadSize', 2);

    quiver3( ...
        ax, 0, 0, 0, ...
        0, 0, origin_axis_len, ...
        0, ...
        'Color', 'b', ...
        'LineWidth', origin_axis_lw, ...
        'MaxHeadSize', 2);
end

% ========================================================================
% Axis formatting
% ========================================================================

set( ...
    ax, ...
    'FontName', font_name, ...
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

xlim(ax, x_lim);
ylim(ax, y_lim);
zlim(ax, z_lim);

xticks(ax, x_ticks);
yticks(ax, y_ticks);
zticks(ax, z_ticks);

if show_axis_labels
    xlabel( ...
        ax, ...
        'X (mm)', ...
        'FontName', font_name, ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

    ylabel( ...
        ax, ...
        'Y (mm)', ...
        'FontName', font_name, ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

    zlabel( ...
        ax, ...
        'Z (mm)', ...
        'FontName', font_name, ...
        'FontSize', 14, ...
        'FontWeight', 'bold');
else
    xlabel(ax, '');
    ylabel(ax, '');
    zlabel(ax, '');
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

view(ax, view_azimuth, view_elevation);
pbaspect(ax, plot_box_aspect);

title(ax, '');

% ========================================================================
% Legend
% ========================================================================
if showlegend
    lgd = legend( ...
        ax, ...
        [ ...
            h_shape_gt, ...
            h_shape_pred, ...
            h_contact_gt, ...
            h_contact_pred, ...
            h_force_gt, ...
            h_force_pred], ...
        { ...
            'GT Shape', ...
            'Predicted Shape', ...
            'GT Contact', ...
            'Predicted Contact', ...
            'GT Force', ...
            'Predicted Force'}, ...
        'Location', 'northeast', ...
        'FontName', font_name, ...
        'FontSize', 10, ...
        'Box', 'on');
    
    lgd.ItemTokenSize = [22, 10];
end


% ========================================================================
% Command-window report
% ========================================================================

disp(' ');
disp('================ Joint Prediction Comparison ================');
fprintf('Final evaluation case index : %d\n', idx);
fprintf('Nominal force class         : %d deg\n', force_class);

fprintf('\nShape:\n');
fprintf('  Mean shape error          : %.3f mm\n', shape_error_mm);
fprintf('  Tip error                 : %.3f mm\n', tip_error_mm);

fprintf('\nContact location:\n');
fprintf('  GT disk                   : %.3f\n', disk_gt);
fprintf('  Predicted disk            : %.3f\n', disk_pred);
fprintf('  Absolute location error   : %.3f disk\n', location_error_disk);

fprintf('\nContact force:\n');
fprintf('  GT force                  : [%.4f, %.4f, %.4f] N\n', ...
    F_gt_aug(1), F_gt_aug(2), F_gt_aug(3));
fprintf('  Predicted force           : [%.4f, %.4f, %.4f] N\n', ...
    F_pred_aug(1), F_pred_aug(2), F_pred_aug(3));
fprintf('  GT magnitude              : %.4f N\n', force_magnitude_gt_N);
fprintf('  Predicted magnitude       : %.4f N\n', force_magnitude_pred_N);
fprintf('  Force-vector error        : %.4f N\n', force_vector_error_N);
fprintf('  Direction error           : %.3f deg\n', force_angle_error_deg);
disp('=============================================================');
disp(' ');

% ========================================================================
% Export
% ========================================================================

if export_this_figure
    exportgraphics( ...
        fig, ...
        fullfile(output_folder, [fig_name, '.pdf']), ...
        'ContentType', 'vector');

    exportgraphics( ...
        fig, ...
        fullfile(output_folder, [fig_name, '.png']), ...
        'Resolution', 600);

    savefig( ...
        fig, ...
        fullfile(output_folder, [fig_name, '.fig']));
end

disp(['>>> Exported joint comparison figure to folder: ', output_folder]);


%% ========================================================================
% Local functions
% ========================================================================

function contact_pt = local_get_continuous_disk_position(P, disk_position)
% Map a continuous physical disk position to the plotted backbone.
%
% Physical markers from bottom to top:
%   Marker1 -> Disk1
%   Marker2 -> Disk3
%   Marker3 -> Disk5
%   Marker4 -> Disk7
%   Marker5 -> Disk9
%   Marker6 -> Disk11
%   Marker7 -> Disk13
%
% Stored shape order:
%   P(:,1) -> Disk13
%   P(:,2) -> Disk11
%   P(:,3) -> Disk9
%   P(:,4) -> Disk7
%   P(:,5) -> Disk5
%   P(:,6) -> Disk3
%   P(:,7) -> Disk1

    disk_position = max(1, min(13, disk_position));

    physical_disk_ids = [1, 3, 5, 7, 9, 11, 13];

    marker_points_bottom_to_top = [ ...
        P(:,7), ...
        P(:,6), ...
        P(:,5), ...
        P(:,4), ...
        P(:,3), ...
        P(:,2), ...
        P(:,1)];

    contact_pt = zeros(3, 1);

    for dim_id = 1:3
        contact_pt(dim_id) = interp1( ...
            physical_disk_ids, ...
            marker_points_bottom_to_top(dim_id, :), ...
            disk_position, ...
            'linear');
    end
end


function [force_class, F_nom, R_restore] = ...
    local_augmented_force_to_original_class_and_rotation(F_aug)

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
         0,           0,           0];

    base_labels = [0, 45, 90];

    R0 = eye(3);

    R120 = [ ...
        cosd(120), -sind(120), 0; ...
        sind(120),  cosd(120), 0; ...
        0,          0,         1];

    R240 = [ ...
        cosd(240), -sind(240), 0; ...
        sind(240),  cosd(240), 0; ...
        0,          0,         1];

    Rset = cat(3, R0, R120, R240);

    best_score = -inf;
    best_base_id = 1;
    best_rot_id = 1;

    for rr = 1:3
        for bb = 1:3
            candidate = Rset(:, :, rr) * base_dirs(:, bb);
            candidate = candidate / norm(candidate);

            score = dot(f, candidate);

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