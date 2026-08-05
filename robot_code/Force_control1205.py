import pandas as pd
import time
import csv
import msvcrt  # Windows专用按键检测
import os
from pymodbus.client import ModbusSerialClient as ModbusClient
# 导入原有模块
from ForceControl0813 import ZMCWrapper, PDController

# === 配置 ===
INPUT_EXCEL = 'trajectory.xlsx'  # 你的目标力值表
OUTPUT_LOG = 'pc1_force_log.csv' # 记录实际力值的表

def main():
    # --- 1. 初始化硬件 ---
    zaux = ZMCWrapper()
    if zaux.connect("192.168.0.11") != 0:
        print("控制器连接失败")
        return
    
    client = ModbusClient(method='rtu', port='COM3', baudrate=19200, timeout=20)
    if not client.connect():
        print("力传感器连接失败")
        return

    # 初始化轴 (复用你的逻辑)
    for i in range(6):
        zaux.set_atype(i, 1)
        zaux.set_units(i, 800) 
        zaux.SetInvertIn(i, 0)

    # --- 2. 读取 Excel 目标 ---
    try:
        # 读取 Excel，假设第一行是表头，数据从第二行开始
        # usecols="E:J" 读取 E 到 J 列
        df = pd.read_excel(INPUT_EXCEL, usecols="E:J") 
        targets = df.values # 转为numpy数组
        print(f"成功加载 {len(targets)} 组数据")
    except Exception as e:
        print(f"Excel 读取失败: {e}")
        return

    # --- 3. 准备记录文件 ---
    if not os.path.exists(OUTPUT_LOG):
        with open(OUTPUT_LOG, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Real_F0', 'Real_F1', 'Real_F2', 'Real_F3', 'Real_F4', 'Real_F5'])

    # --- 4. 循环执行 ---
    # 创建 PID 控制器
    pids = [PDController(Kp=0.3, Kd=0.01, goal_value=0, tolerance=5) for _ in range(6)]
    
    for idx, target_forces in enumerate(targets):
        print(f"\n>>> 正在执行第 {idx+1}/{len(targets)} 组姿态")
        print(f"目标力值: {target_forces}")
        print("PID 调节中... (稳定后按 'Enter' 记录数据并下一步，按 'Esc' 退出)")
        
        # 更新 PID 目标
        for i in range(6):
            pids[i].goal_value = target_forces[i]

        # 保持姿态循环
        while True:
            # 读取传感器
            rq = client.read_holding_registers(0x300, 12, 1)
            if rq.isError(): continue
            data = rq.encode()
            current_forces = [int.from_bytes(data[i*4+1:i*4+5], 'big', signed=True) for i in range(6)]

            # PID 计算与运动
            for i in range(6):
                output = 25 * pids[i].update(current_forces[i], 0.1) # 假定dt=0.1简化
                zaux.move(i, output)

            # 按键检测
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\r':  # Enter键
                    print(f"记录第 {idx+1} 组数据...")
                    # 写入 CSV
                    with open(OUTPUT_LOG, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([idx+1] + current_forces)
                    break # 跳出 while，进入下一个 for 循环
                elif key == b'\x1b': # Esc键
                    print("程序终止")
                    zaux.disconnect()
                    return
            
            time.sleep(0.05) # 稍微延时，防止CPU占用过高

    print("所有轨迹执行完毕！")
    zaux.disconnect()

if __name__ == "__main__":
    main()