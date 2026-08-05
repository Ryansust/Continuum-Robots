%% ========================================================================
%  Step 9.8: Physical Model Baseline
%  Corrected alignment:
%  test_idx indexes the v_mask-filtered Net-C dataset, not the full aug_* arrays.
% =========================================================================
disp('--------------------------------------------------');
disp('9.8 Calculating Physical Model Baseline Error with corrected indexing...');

% -------------------------- User Options ----------------------------------
external_force_sign = -1;    % try 1 first; if physics baseline is globally reversed, try -1

% -------------------------- Safety Check ----------------------------------
if ~exist('v_mask', 'var')
    error('Missing v_mask. Step 9.8 must be run after Net-C dataset construction.');
end

if ~exist('test_idx', 'var')
    error('Missing test_idx. Step 9.8 must be run after Net-C test split.');
end

if ~exist('aug_F_after', 'var') || ~exist('aug_gt_F', 'var') || ...
   ~exist('aug_hgt', 'var') || ~exist('aug_Pa', 'var')
    error('Missing required augmented variables: aug_F_after, aug_gt_F, aug_hgt, aug_Pa.');
end

% ---------------------- Correct index alignment ---------------------------
% test_idx is an index inside the v_mask-filtered Net-C dataset.
% Therefore, map it back to the original augmented dataset.
valid_aug_idx = find(v_mask);
global_test_idx = valid_aug_idx(test_idx);

fprintf('   > Net-C test samples: %d\n', numel(test_idx));
fprintf('   > Correctly mapped augmented test samples: %d\n', numel(global_test_idx));

% Extract aligned testing samples from the original augmented dataset.
F_after_test_all = aug_F_after(:, global_test_idx);
gt_F_vec_test    = aug_gt_F(:, global_test_idx);
gt_hgt_test      = aug_hgt(global_test_idx);
Pa_real_test_all = aug_Pa(:, global_test_idx);

N_test_samples = size(F_after_test_all, 2);
P_phys_after_all = zeros(21, N_test_samples);

% -------------------------- Physical Parameters ---------------------------
tendon_p = 3;
section_p = 2;
D_p = 0.0006;
E_p = 0.516e+12;
L_ap = 0.0665;
L_bp = 0.00;
N_dp = 7;

H_listp = linspace(0.0025, 0.0025, section_p*N_dp+1);

mu_p = 0.25;
delta_alphap = 0;
G_loadp = 4.000 * 0.00981;

tic;

for i = 1:N_test_samples
    % A. Tendon tension input.
    % Keep the same tendon index convention used in the CSBCM prior.
    Fa_raw = F_after_test_all(:, i);
    Fa_sim = [Fa_raw(5); Fa_raw(6); Fa_raw(1); Fa_raw(2); Fa_raw(3); Fa_raw(4)];

    % B. External force and contact location.
    F_ext_vec = external_force_sign * gt_F_vec_test(:, i);
    F_hgt_node = gt_hgt_test(i);

    % Marker node k corresponds approximately to solver loading point 2*k.
    % Node 3 -> 6, Node 4 -> 8, Node 5 -> 10.
    n_load = round(F_hgt_node * 2);
    n_load = max(1, min(14, n_load));

    % C. Solve physical model.
    [P_Theo, ~, R_mat, ~, ~, ~] = solve_continuum_shape_nofig(...
        tendon_p, section_p, D_p, E_p, L_ap, L_bp, N_dp, ...
        H_listp, mu_p, delta_alphap, ...
        G_loadp, F_ext_vec, Fa_sim, n_load);

    % D. 4-mm marker radial offset compensation.
    V_local = [0; -0.004; 0];

    P_m = zeros(3, size(P_Theo, 2));
    for pt = 1:size(P_Theo, 2)
        P_m(:, pt) = P_Theo(:, pt) + R_mat(:, :, pt) * V_local;
    end

    % E. Extract 7 marker-defined backbone nodes.
    marker_idx = round([2,4,6,8,10,12,14] * ((size(P_Theo,2)-1)/14)) + 1;
    P_phys_after_all(:, i) = reshape(P_m(:, marker_idx), 21, 1);
end

toc;

% -------------------------- Error Evaluation ------------------------------
err_phys_raw = zeros(1, N_test_samples);
shape_err_phys_raw = zeros(1, N_test_samples);

for i = 1:N_test_samples
    P_phys = reshape(P_phys_after_all(:, i), 3, 7);
    P_real = reshape(Pa_real_test_all(:, i), 3, 7);

    % Use first node as origin for fair geometric comparison.
    P_phys = P_phys - P_phys(:, 1);
    P_real = P_real - P_real(:, 1);

    err_phys_raw(i) = norm(P_phys(:, end) - P_real(:, end)) * 1000;
    shape_err_phys_raw(i) = mean(sqrt(sum((P_phys - P_real).^2, 1))) * 1000;
end

% Apply final filtering if v_idx exists.
if exist('v_idx', 'var')
    error_phys_final = err_phys_raw(v_idx);
    shape_error_phys_final = shape_err_phys_raw(v_idx);
else
    warning('v_idx does not exist. Using all physical baseline test samples.');
    error_phys_final = err_phys_raw;
    shape_error_phys_final = shape_err_phys_raw;
end

fprintf('   > Physical baseline external_force_sign = %+d\n', external_force_sign);
fprintf('   > Physical baseline Tip MAE:   %.2f mm\n', mean(error_phys_final));
fprintf('   > Physical baseline Shape MAE: %.2f mm\n', mean(shape_error_phys_final));

disp('>>> Step 9.8 physical baseline finished with corrected indexing.');