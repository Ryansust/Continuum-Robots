clc;
clear;
close all;

%% 1. 读取数据
FileName = '/Users/ryan/Desktop/continuum robot/force_data/after_processing_data_0816.xlsx';
dataTable = readtable(FileName);
% disp(dataTable(1:5,:)); % 显示前5行预览

% 获取总的实验数量（减去表头2行）
num_total_exp = height(dataTable) - 2;

% ================= 数据提取与类型转换 (关键修正) =================
% 建议全部在循环外转为 double 数组，提高效率并防止索引错误

% 1. 驱动力 (F1-F6) -> Excel Col 5-10
force_array = table2array(dataTable(3:end, 5:10));

% 2. 外力大小 -> Excel Col 2 (假设这是外力传感器读数)
outer_force_array = table2array(dataTable(3:end, 2)); 

% 3. 外力方向编码 -> Excel Col 3
direction_array = table2array(dataTable(3:end, 3));

% 4. 位姿真值 -> Excel Col 29
position_text_array = table2array(dataTable(3:end, 29));

% 5. 高度编码 (如果后面要用) -> Excel Col 4
height_array = table2array(dataTable(3:end, 4));

%% 2. 批量处理力和位姿数据
% 初始化变量
F_tendon = zeros(num_total_exp, 6);
F_ex = zeros(3, num_total_exp);
P_RealOffset = zeros(3, 9, num_total_exp);

for i = 1:num_total_exp
   % 计算驱动腱的力 (Tendon Forces) - 假设用全部6个
   F_tendon(i,:) = [0.00981*force_array(i,1), 0.00981*force_array(i,2), 0.00981*force_array(i,3), 0.00981*force_array(i,4), 0.00981*force_array(i,5), 0.00981*force_array(i,6)];

   % ---- 【核心修改部分：根据方向编码计算外部力 F_ex】 ----
   
   % [Fix 1] 使用正确的外力大小数据源
   % 导入外力传感器数据
   force_magnitude = abs(outer_force_array(i)); 
   
   % [Fix 2] 获取当前行的方向编码 (标量)
   direction_code = direction_array(i);
   
   unit_vec = [0; 0; 0]; % 初始化单位向量

   switch direction_code
       case 2
           unit_vec = [-1; 0; 0];         
       case 3
           unit_vec = [-sind(45); cosd(45); 0];
       case 4
           unit_vec = [0; 1; 0];        
       otherwise
           % 如果编码不是2,3,4，且力不为0，则警告。如果力本身是0（无外力），则不需要警告
           if force_magnitude > 0.1 
               warning('第 %d 行实验的方向编码 %d 无效，但存在外力 %.2f。F_ex设为0。', i, direction_code, force_magnitude);
           end
   end

   F_ex(:, i) = force_magnitude * unit_vec;

   % 解析位姿真值
   P_RealOffset(:,:,i) = get_RealOffset_1S3CT(position_text_array{i});
end

%% 3. 设置模型常量和参数
% (保持不变)
tendon = 3;
section = 1;
G = 8*0.00981;
D = 0.0006;
I = pi*D^4/64;
L_b = 0;
L_e = 0;
L_a= 0.070;
N_d = 9;
H_list  = linspace(0.0025, 0.0025, section*N_d+1);
kappa = 4;
id = [0, 2, 4, 7, 9];
T_offset_0 = repmat(eye(4), 1, 1, 9);
E = 1.816e11;
mu = 0;
EI = E * I;

%% 4. 选择多组实验进行批量计算和绘图 (双色系配色方案)

% =================== 筛选并绘图逻辑修正 ===================

% ---- Step 1: Define Filtering Criteria ----
target_column_index = 3;  % 对应 direction_array
target_value = 4;         % 筛选方向为 4 的数据
num_plots = 50;           % 绘图数量

% ---- Step 2: Find Matching Experiments ----
% 直接使用我们在第一步转换好的 array，速度更快且不出错
column_data = direction_array; 

group_indices = find(column_data == target_value);

% ---- Step 3: Randomly Select ----
if isempty(group_indices)
    error('Filtering Error: No experiments were found where column %d has a value of %d.', target_column_index, target_value);
end

if length(group_indices) < num_plots
    warning('Found only %d experiments matching criteria. Plotting all.', length(group_indices));
    experiments_to_plot = group_indices;
    num_plots = length(experiments_to_plot); % 更新实际绘图数量
else
    shuffled_indices = group_indices(randperm(length(group_indices)));
    experiments_to_plot = shuffled_indices(1:num_plots);
end

fprintf('Found %d experiments with val %d. Plotting %d.\n', length(group_indices), target_value, num_plots);

% =================== 开始绘图 ===================

figure('Color', 'white'); 
hold on;

% 颜色定义
cmap = parula(256); 

% --- 预计算 Y 轴范围用于颜色映射 (仅基于选定的实验) ---
all_y_values = [];
for k = 1:num_plots
    num_exp = experiments_to_plot(k);
    all_y_values = [all_y_values, P_RealOffset(2, :, num_exp)];
end
min_y = min(all_y_values);
max_y = max(all_y_values);

% 绘图循环
for k = 1:num_plots
    num_exp = experiments_to_plot(k);
    
    x_data = P_RealOffset(1, :, num_exp);
    y_data = P_RealOffset(2, :, num_exp);
    z_data = P_RealOffset(3, :, num_exp);
    
    % 计算颜色 (基于平均深度)
    average_y = mean(y_data);
    if max_y == min_y % 防止除以0
        normalized_depth = 0.5;
    else
        normalized_depth = (average_y - min_y) / (max_y - min_y);
    end
    
    color_index = round(1 + (normalized_depth * (size(cmap, 1) - 1)));
    color_index = max(1, min(color_index, size(cmap, 1)));
    current_color = cmap(color_index, :);

    plot3(x_data, y_data, z_data, '--s', ...
        'Color', current_color, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 6, ...
        'MarkerFaceColor', current_color);
end

% Colorbar 设置
colormap(cmap); 
c = colorbar;   
c.Label.String = 'Depth along Y-axis (m)'; 
caxis([min_y max_y]); 

% 坐标轴设置
quiver3(0, 0, 0, 0.03, 0, 0, 'r', 'LineWidth', 2); 
quiver3(0, 0, 0, 0, 0.03, 0, 'g', 'LineWidth', 2); 
quiver3(0, 0, 0, 0, 0, 0.03, 'b', 'LineWidth', 2); 
xlabel('x (m)'), ylabel('y (m)'), zlabel('z (m)');
title(sprintf('Visualization of Direction %d (N=%d)', target_value, num_plots));
axis equal, grid on; 
view(30,10);
set(gca, 'FontSize', 11, 'LineWidth', 1);
set(gca,'zdir','reverse'); 
set(gca,'ydir','reverse');

hold off;