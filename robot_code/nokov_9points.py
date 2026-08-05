import csv
import os
import time
import msvcrt # Windows专用
from 0607_9Points import MotionCap

# === 配置 ===
OUTPUT_FILE = 'pc2_mocap_log.csv'

def main():
    # 初始化动捕
    mocap = MotionCap('10.1.1.198') 
    print("动捕系统就绪。按 'Enter' 采集当前帧，按 'Esc' 退出。")

    # 准备文件
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # 表头
            header = ['Index']
            for i in range(1, 10): # 9个点
                header.extend([f'P{i}_X', f'P{i}_Y', f'P{i}_Z'])
            writer.writerow(header)

    count = 1
    
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\r': # Enter
                print(f"正在采集第 {count} 组...", end="")
                
                # 获取数据
                positions = mocap.opt_get_all_marker_positions()
                
                # 检查有效性
                flat_pos = []
                is_valid = True
                for pos in positions:
                    if mocap.is_invalid_position(pos):
                        is_valid = False
                    flat_pos.extend(pos[:3]) # 只取xyz
                
                if is_valid:
                    with open(OUTPUT_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([count] + flat_pos)
                    print(f" [成功] 已保存")
                    count += 1
                else:
                    print(f" [失败] 检测到丢点，请调整后再次按 Enter 重试！")
                    # count 不增加，让你重试
            
            elif key == b'\x1b': # Esc
                break
        
        time.sleep(0.01)

if __name__ == "__main__":
    main()