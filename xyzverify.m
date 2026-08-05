%% ========================================================================
%  Script: Sim vs Real Coordinate System Alignment Checker (Auto Filter Version)
%  Goal:   Overlay Theoretical Shape, Real Shape, and Display 6-Axis Forces
% =========================================================================
clc; clear; close all;

%% 1. 读取并解析 Excel 数据 (提前读取以便检索)
disp('1. 正在读取并解析 Excel 数据...');
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);

force_array = double(table2array(dataTable(3:end, 5:10)));
outer_force_array = double(table2array(dataTable(3:end, 2))); 
direction_array = double(table2array(dataTable(3:end, 3)));
height_array = double(table2array(dataTable(3:end, 4)));
position_text_array = dataTable{3:end, 38};

%% 2. 自动检索满足条件的行号 (核心修改区)
disp('2. 正在检索满足条件 (T1 ≠ 0, 且 T2~T6 = 0) 的数据...');

% 逻辑条件：第1列不为0，且第2至6列全为0
% (考虑到可能存在浮点数误差，如果你的"0"含有微小噪声，可以将 == 0 改为 < 1e-4)
condition = (force_array(:, 1) ~= 0) & all(force_array(:, 2:6) == 0, 2);

% 找到满足条件的索引
selected_indices = find(condition);

% 结果检查与保护机制
if isempty(selected_indices)
    error('❌ 没有找到任何满足条件 (T1不为0且其他力全为0) 的数据！请检查 Excel。');
end

num_found = length(selected_indices);
fprintf('✅ 共找到 %d 组满足条件的数据。\n', num_found);

% 防卡死保护：如果找到太多，最多只画前 10 组
max_plots = 10; 
if num_found > max_plots
    fprintf('⚠️ 为防止弹出窗口过多卡死电脑，仅展示前 %d 组。\n', max_plots);
    selected_indices = selected_indices(1:max_plots);
end

%% 3. 机器人基础参数设置
disp('3. 初始化机器人参数...');
tendon = 3;         
section = 2;        
D = 0.0006;         
E = 1.016e+12;      
L_a = 0.0665;         
L_b = 0.00;         
N_d = 7;            
H_list = linspace(0.0025, 0.0025, section*N_d+1); 
mu = 0.25;          
delta_alpha = 0; 
G_load = 4.000 * 0.00981; 

%selected_indices=[167,74,89,245,345];
%% 4. 开始对比计算与绘图
disp('4. 开始计算并绘图...');
num_plots = length(selected_indices);

for k = 1:num_plots
    exp_id = selected_indices(k);
    fprintf('   -> 正在处理第 %d 组数据 (对应有效数据区第 %d 行)...\n', k, exp_id);
    
    % --- A. 提取该组的输入参数 ---
    F_tendon = force_array(exp_id, :) * 0.00981; 
    F_tendon2 =[F_tendon(5), F_tendon(6), F_tendon(1), F_tendon(2), F_tendon(3), F_tendon(4)];
    
    f_mag = abs(outer_force_array(exp_id));
    dir_code = direction_array(exp_id);
    u_vec = [0;0;0];
    f_mag=0; % (注意：你原代码这里把 f_mag 强制置为 0 了)
    switch dir_code
        case 2, u_vec =[-1; 0; 0];
        case 3, u_vec =[-sind(45); cosd(45); 0];
        case 4, u_vec =[0; 1; 0];
    end
    F_ex = f_mag * u_vec;
    
    touch_id = height_array(exp_id); 
    if touch_id == 0, touch_id = section * N_d; end 
    
    % --- B. 获取真值与理论值 ---
    % 真实形状 (Nokov)
    P_Real = get_RealOffset_1S3CT(position_text_array{exp_id});
    
    % 理论形状 (Cosserat Solver)
    [P_Theo, ~, ~, ~, ~, ~] = solve_continuum_shape(tendon, section, D, E, L_a, L_b, N_d, H_list, mu, delta_alpha, G_load, F_ex, F_tendon2, touch_id);
    
    % --- C. 绘图重叠对比 (每个Case独立生成一张图) ---
    figure('Name', sprintf('Alignment Checker - Data Index %d', exp_id), 'Color', 'w', 'Position',[100 + k*30, 100 + k*30, 800, 600]);
    hold on; grid on; axis equal;
    
    % 1. 画真实数据
    h_real = plot3(P_Real(1,:), P_Real(2,:), P_Real(3,:), '--bs', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    
    % 2. 画理论模型
    h_theo = plot3(P_Theo(1,:), P_Theo(2,:), P_Theo(3,:), '-ro', 'LineWidth', 1.5, 'MarkerSize', 3, 'MarkerFaceColor', 'r');
    
    % 3. 画基坐标系
    quiver3(0,0,0, 0.05,0,0, 'r', 'LineWidth', 3, 'MaxHeadSize', 0.5); text(0.06,0,0, 'X', 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
    quiver3(0,0,0, 0,0.05,0, 'g', 'LineWidth', 3, 'MaxHeadSize', 0.5); text(0,0.06,0, 'Y', 'Color', 'g', 'FontSize', 12, 'FontWeight', 'bold');
    quiver3(0,0,0, 0,0,0.05, 'b', 'LineWidth', 3, 'MaxHeadSize', 0.5); text(0,0,0.06, 'Z', 'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 4. 打印 6 轴肌腱力与外力信息
    force_text = sprintf('【Input Forces】\nT_1: %.3f N   T_2: %.3f N   T_3: %.3f N\nT_4: %.3f N   T_5: %.3f N   T_6: %.3f N\n-----------------------\nExternal Load: %.3f N', ...
                         F_tendon(1), F_tendon(2), F_tendon(3), ...
                         F_tendon(4), F_tendon(5), F_tendon(6), f_mag);
                     
    text(0.05, 0.95, force_text, 'Units', 'normalized', ...
        'FontSize', 10, 'FontName', 'Courier New', 'FontWeight', 'bold', ...
        'BackgroundColor',[1 1 1 0.8], 'EdgeColor', 'k', 'Margin', 5, ...
        'VerticalAlignment', 'top');

    % 5. 视角与装饰
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('Data Index: %d | Dir: %d', exp_id, dir_code));
    view(30, 20);
    
    % 独立图例
    legend([h_real, h_theo], {'Ground Truth (Nokov)', 'Theoretical (Cosserat)'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
end

disp('✅ 所有检索到的数据已绘图完成！');