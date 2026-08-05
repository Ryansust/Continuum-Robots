import ctypes
import platform
import time
import os
import plotly.io as pio
import sys
import logging
import random

logging.getLogger('pymodbus').setLevel(logging.WARNING)
sys.path.append(r'C:\Users\a\OneDrive - City University of Hong Kong - Student\Code\src\exp')
dll_path = r'C:\Users\a\OneDrive - City University of Hong Kong - Student\Code\src\exp\zauxdll.dll'
zauxdll = ctypes.WinDLL(dll_path)

from pymodbus.client import ModbusSerialClient as ModbusClient
from exp.MotorControl import MotorControl
pio.renderers.default = "iframe_connected"
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.dirname(__file__))

class ZMCWrapper:

    # 初始化参数
    def __init__(self):
        self.handle = ctypes.c_void_p()
        self.sys_ip = ""
        self.sys_info = ""
        self.is_connected = False

    ###############################控制器连接################################################
    def connect(self, ip, console=[]):
        if self.handle.value is not None:
            self.disconnect()
        ip_bytes = ip.encode('utf-8')
        p_ip = ctypes.c_char_p(ip_bytes)
        print("Connecting to", ip, "...")
        ret = zauxdll.ZAux_OpenEth(p_ip, ctypes.pointer(self.handle))
        msg = "Connected"
        if ret == 0:
            msg = ip + " Connected"
            self.sys_ip = ip
            self.is_connected = True
        else:
            msg = "Connection Failed, Error " + str(ret)
            self.is_connected = False
        console.append(msg)
        console.append(self.sys_info)
        return ret

    # 断开连接
    def disconnect(self):
        ret = zauxdll.ZAux_Close(self.handle)
        self.is_connected = False
        return ret

###############################轴参数设置################################################
    def set_atype(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetAtype(self.handle, iaxis, iValue)
        if ret == 0:
            print("Set Axis (", iaxis, ") Atype:", iValue)
        else:
            print("Set Axis (", iaxis, ") Atype fail!")
        return ret

    def set_units(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetUnits(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Units:", iValue)
        else:
            print("Set Axis (", iaxis, ") Units fail!")
        return ret

    def set_accel(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetAccel(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Accel:", iValue)
        else:
            print("Set Accel (", iaxis, ") Accel fail!")
        return ret

    def set_decel(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetDecel(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Decel:", iValue)
        else:
            print("Set Axis (", iaxis, ") Decel fail!")
        return ret

    def set_creep(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetCreep(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Creep:", iValue)
        else:
            print("Set Axis (", iaxis, ") Creep fail!")
        return ret

    def set_speed(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetSpeed(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Speed:", iValue)
        else:
            print("Set Axis (", iaxis, ") Speed fail!")
        return ret

    def set_dpos(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetDpos(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Pose:", iValue)
        else:
            print("Set Axis (", iaxis, ") Atype fail!")
        return ret

    def set_merge(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetMerge(self.handle, iaxis, ctypes.c_float(iValue))
        return ret
###############################轴参数读取################################################
    def get_atype(self, iaxis):
        iValue = (ctypes.c_int)()
        ret = zauxdll.ZAux_Direct_GetAtype(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Atype:", iValue.value)
        else:
            print("Get Axis (", iaxis, ") Atype fail!")
        return ret

    def get_untis(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetUnits(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Units:", iValue.value)
        else:
            print("Get Axis (", iaxis, ") Units fail!")
        return ret

    def get_accel(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetAccel(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Accel:",  iValue.value)
        else:
            print("Get Axis (", iaxis, ") Accel fail!")
        return ret

    def get_decel(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetDecel(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Decel:",  iValue.value)
        else:
            print("Get Axis (", iaxis, ") Decel fail!")
        return ret

    def get_speed(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetSpeed(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") Speed:",  iValue.value)
        else:
            print("Get Axis (", iaxis, ") Speed fail!")
        return ret

    def get_creep(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetCreep(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Set Axis (", iaxis, ") Creep:", iValue)
        else:
            print("Set Axis (", iaxis, ") Creep fail!")
        return ret

    def get_dpos(self, iaxis):
        iValue = (ctypes.c_float)()
        ret = zauxdll.ZAux_Direct_GetMpos(self.handle, iaxis, ctypes.byref(iValue))
        if ret == 0:
            print("Get Axis (", iaxis, ") DPose:", iValue)
        else:
            print("Get Axis (", iaxis, ") Atype fail!")
        return iValue
###############################运动调用####################################################
    def move(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_Single_Move(self.handle, iaxis, ctypes.c_float(iValue))
        if ret == 0:
            print("Axis (", iaxis, ") Move:", iValue)
        else:
            print("Axis (", iaxis, ") Move Fail")
        return ret

    def single_absmove(self, iaxis, distance):
        ret = zauxdll.ZAux_Direct_Single_MoveAbs(self.handle, iaxis, distance)
        return ret

    def all_absmove(self, num, axislist, distancelist):
        # distancelist = (ctypes.c_float)()
        a = (ctypes.c_float * len(distancelist))(*distancelist)
        ret = zauxdll.ZAux_Direct_MoveAbs(self.handle,  num, (ctypes.c_int * len(distancelist))(*axislist), a)
        return ret

    def vmove(self, iaxis, idir):
        ret = zauxdll.ZAux_Direct_Single_Vmove(self.handle, iaxis, idir)
        if ret == 0:
            print("axis (", iaxis, ")Vmoving!")
        else:
            print("Vmoving fail!")
        return ret

    def directmove(self, iaxis, distance):
        ret = zauxdll.ZAux_Direct_Single_MoveAbs(self.handle, iaxis, distance)
        return ret

    def MoveDelay(self, iaxis, itime):
        ret = zauxdll.ZAux_Direct_MoveDelay(self.handle, iaxis, itime)
        return ret

    def SetRevIn(self, iaxis, iValue):
        ret = zauxdll.ZAux_Direct_SetRevIn(self.handle, iaxis, iValue)
        return ret

    def SetInvertIn(self, ionum, iValue):
        ret = zauxdll.ZAux_Direct_SetInvertIn(self.handle, ionum, iValue)
        return ret

    def SetDatumIn(self, iaxis, ionum):
        ret = zauxdll.ZAux_Direct_SetDatumIn(self.handle, iaxis, ionum)
        return ret

    def movehome(self, iaxis):
        ret = zauxdll.ZAux_Direct_Single_Datum(self.handle, iaxis, 3)
        return ret

    def SingleStop(self,iaxis,mode):
        ret = zauxdll.ZAux_Direct_Single_Cancel(self.handle,iaxis,mode)
        return ret

    def get_zero_status(self,iaxis):
        iValue = (ctypes.c_int)()
        iValue1 = (ctypes.c_int)()
        ret =zauxdll.ZAux_Direct_GetDatumIn(self.handle,iaxis,ctypes.byref(iValue))
        # ret1 = zauxdll.ZAux_Direct_GetIn(self.handle,1,ctypes.byref(iValue1))
        print(iValue1.value)
        # print(ret)
        return iValue1.value

    def get_home_status(self,iaxis):
        iValue = (ctypes.c_uint)()
        iValue1 = (ctypes.c_uint)()
        ret =zauxdll.ZAux_Direct_GetHomeStatus(self.handle,iaxis,ctypes.byref(iValue))
        return iValue.value

class PDController:
    def __init__(self, Kp, Kd, goal_value, tolerance):
        self.Kp = Kp
        self.Kd = Kd
        self.goal_value = goal_value
        self.tolerance = tolerance
        self.last_error = 0

    def update(self, current_value, delta_time):
        error_F = self.goal_value - current_value
        error_P=error_F/600
        error_derivative = (error_P - self.last_error) / delta_time if delta_time > 0 else 0
        output = self.Kp * error_P + self.Kd * error_derivative
        self.last_error = error_P

        if abs(error_F) < self.tolerance:
            output = 0

        return output

systype = platform.system()
if systype == 'Windows':
    if platform.architecture()[0] == '64bit':
        zauxdll = ctypes.WinDLL('./zauxdll.dll')
        print('Windows x64')
    else:
        zauxdll = ctypes.WinDLL('./zauxdll.dll')
        print('Windows x86')
elif systype == 'Darwin':
    zmcdll = ctypes.CDLL('./zmotion.dylib')
    print("macOS")
elif systype == 'Linux':
    zmcdll = ctypes.CDLL('./libbzmotion.so')
    print("Linux")
else:
    print("OS Not Supported!!")

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
UNIT = 0x1

if __name__ == "__main__":
    zaux = ZMCWrapper()
    # 连接控制器ip
    zaux.connect("192.168.0.11")
    # 连接力传感器ip
    client = ModbusClient(method='rtu', port='COM3',baudrate=19200, timeout=20)
    client.connect()
    log.debug("写保持寄存器并读回")
    rq = client.read_holding_registers(0x300, 2, 1)
    data = rq.encode()
    force1_read = data[1:5]
    force1 = int.from_bytes(force1_read, 'big', signed=True)
    print("chanel1:{}".format(force1))

    axislist = [0, 1, 2, 3, 4, 5]
    m = MotorControl()

    # 原点复位
    for i in axislist:
        zaux.set_atype(i, 1)
        zaux.set_units(i, 800)
        zaux.set_accel(i, 200)
        zaux.set_decel(i, 200)
        zaux.set_speed(i, 40)
        zaux.SetInvertIn(i, 0)
        zaux.set_creep(i, 5000)

    zaux.SetDatumIn(0, 0)
    zaux.SetDatumIn(1, 1)
    zaux.SetDatumIn(2, 2)
    zaux.SetDatumIn(3, 3)
    zaux.SetDatumIn(4, 4)
    zaux.SetDatumIn(5, 5)

    k = 1
    temp = 1
    zaux.movehome(0)
    zaux.movehome(1)
    zaux.movehome(2)
    zaux.movehome(3)
    zaux.movehome(4)
    zaux.movehome(5)

    while k == 1:
        a = zaux.get_home_status(0)
        b = zaux.get_home_status(1)
        c = zaux.get_home_status(2)
        d = zaux.get_home_status(3)
        e = zaux.get_home_status(4)
        f = zaux.get_home_status(5)
        if a == 1 and b == 1 and c == 1 and d == 1 and f == 1:
            time.sleep(1)
            start = 1
            print("axis get_home finished!")
            k = 0
            zaux.set_dpos(0, 0)
            zaux.set_dpos(1, 0)
            zaux.set_dpos(2, 0)
            zaux.set_dpos(3, 0)
            zaux.set_dpos(4, 0)
            zaux.set_dpos(5, 0)
        else:
            print("axis get_home moving!")
    #电机编码器相对零点的位置，需预先手动调整其至大致位置 25单位=1mm

    pos_zero0 = -613
    pos_zero1 = -589
    pos_zero2 = -863
    pos_zero3 = -679
    pos_zero4 = -858
    pos_zero5 = -446

    zaux.move(0, pos_zero0)
    zaux.move(1, pos_zero1)
    zaux.move(2, pos_zero2)
    zaux.move(3, pos_zero3)
    zaux.move(4, pos_zero4)
    zaux.move(5, pos_zero5)

    start = 0
    k = 1

    # #在这里定义各个轴的力，用来校准 符号+为旧设备 -为新设备 下面的进行相应更改
    # force0_set = -0
    # force1_set = -100
    # force2_set = -100
    # force3_set = -100
    # force4_set = -200
    # force5_set = -100
    #
    # PDController0 = PDController(Kp=0.3, Kd=0.01, goal_value=force0_set, tolerance=5)
    # PDController1 = PDController(Kp=0.3, Kd=0.01, goal_value=force1_set, tolerance=5)
    # PDController2 = PDController(Kp=0.3, Kd=0.01, goal_value=force2_set, tolerance=5)
    # PDController3 = PDController(Kp=0.3, Kd=0.01, goal_value=force3_set, tolerance=5)
    # PDController4 = PDController(Kp=0.3, Kd=0.01, goal_value=force4_set, tolerance=5)
    # PDController5 = PDController(Kp=0.3, Kd=0.01, goal_value=force5_set, tolerance=5)

    ################ 随机生成相邻线的力值 ################
    # 设置安全范围
    MIN_TENSION = 200  # 最小安全张力（绝对值）
    MAX_TENSION = 300  # 最大安全张力（绝对值）

    # 定义线的影响系数 (根据您的描述)
    LOW_IMPACT_AXES = [0, 2, 4]  # 位姿变化小的线
    HIGH_IMPACT_AXES = [1, 3, 5]  # 位姿变化大的线


    def generate_adjacent_tensions():
        """生成相邻线的力值"""
        # 随机选择起始位置 (0-5)
        # start_index = random.randint(0, 5)
        # start_index = random.choice([0, 1, 2, 5])
        start_index = 5
        # 随机选择施加力的线段数量 (1-3)
        num_active = 1

        # 确定施加力的线段索引 (考虑圆周连续性)
        active_indices = []
        for i in range(num_active):
            # 计算当前索引 (考虑圆周)
            current_index = (start_index + i) % 6
            active_indices.append(current_index)

        # 生成力值
        tensions = [-0.0] * 6
        for idx in active_indices:
            # 根据线的影响系数调整力值范围
            if idx in LOW_IMPACT_AXES:
                # 小变化线 - 施加较大力值
                tension = random.uniform(MAX_TENSION * 0.6, MAX_TENSION)
            else:
                # 大变化线 - 施加较小力值
                tension = random.uniform(MIN_TENSION, MAX_TENSION * 0.7)
            tensions[idx] = -tension

        return tensions, active_indices

    # 生成随机力值
    target_forces, active_indices = generate_adjacent_tensions()
    t2 = 150 + random.uniform(-50, 50)
    t1 = 500 + random.uniform(-50, 20)
    # 指定力值
    target_forces[0] = -103.06
    target_forces[1] = -485.39
    target_forces[2] = -0
    target_forces[3] = -0
    target_forces[4] = -0
    target_forces[5] = -0

    # 打印生成的力值和激活的线
    print("\n生成的力值:")
    print(f"force0_set = {target_forces[0]:.1f}")
    print(f"force1_set = {target_forces[1]:.1f}")
    print(f"force2_set = {target_forces[2]:.1f}")
    print(f"force3_set = {target_forces[3]:.1f}")
    print(f"force4_set = {target_forces[4]:.1f}")
    print(f"force5_set = {target_forces[5]:.1f}")
    print(f"{target_forces[0]:.2f},{target_forces[1]:.2f},{target_forces[2]:.2f},{target_forces[3]:.2f},{target_forces[4]:.2f},{target_forces[5]:.2f}")
    print(f"激活的线: {active_indices}")

    # 创建PD控制器
    PDController0 = PDController(Kp=0.3, Kd=0.01, goal_value=target_forces[0], tolerance=5)
    PDController1 = PDController(Kp=0.3, Kd=0.01, goal_value=target_forces[1], tolerance=5)
    PDController2 = PDController(Kp=0.3, Kd=0.01, goal_value=target_forces[2], tolerance=5)
    PDController3 = PDController(Kp=0.3, Kd=0.01, goal_value=target_forces[3], tolerance=5)
    PDController4 = PDController(Kp=0.3, Kd=0.01, goal_value=target_forces[4], tolerance=5)
    PDController5 = PDController(Kp=0.3, Kd=0.01, goal_value=target_forces[5], tolerance=5)


    while k==1:
        pos3 =zaux.get_dpos(3)
        print(pos3)
        if pos3.value==pos_zero3:
            start=1
            print('Start now:')
            k=0

    if start==1:
        # 创建数据记录结构
        baseline_forces = None  # 基准力值 (6个传感器的值)
        stable_encoders = None  # 稳定状态编码器值

        # 主控制循环
        for i in range(150):  # 增加总循环次数
            # 读取力传感器数据
            rq = client.read_holding_registers(0x300, 12, 1)
            data = rq.encode()
            force0_read = data[1:5]
            force1_read = data[5:9]
            force2_read = data[9:13]
            force3_read = data[13:17]
            force4_read = data[17:21]
            force5_read = data[21:25]

            current_forces = [
                int.from_bytes(force0_read, 'big', signed=True),
                int.from_bytes(force1_read, 'big', signed=True),
                int.from_bytes(force2_read, 'big', signed=True),
                int.from_bytes(force3_read, 'big', signed=True),
                int.from_bytes(force4_read, 'big', signed=True),
                int.from_bytes(force5_read, 'big', signed=True)
            ]

            # 获取当前编码器值
            current_encoders = [
                zaux.get_dpos(0).value,
                zaux.get_dpos(1).value,
                zaux.get_dpos(2).value,
                zaux.get_dpos(3).value,
                zaux.get_dpos(4).value,
                zaux.get_dpos(5).value
            ]

            # 基准力值记录 (第70次循环)
            if i == 70:
                baseline_forces = current_forces.copy()
                stable_encoders = current_encoders.copy()

                print("\n系统稳定状态已记录 (第70次循环)")
                print("基准力值: ", end="")
                print(" | ".join(f"{f:>6.1f}N" for f in current_forces))
                print("编码器值: ", end="")
                print(" | ".join(f"{e:>8.2f}" for e in current_encoders))
            # 正常控制逻辑
            delta_time = 0.1
            output0 = 25 * PDController0.update(current_forces[0], delta_time)
            output1 = 25 * PDController1.update(current_forces[1], delta_time)
            output2 = 25 * PDController2.update(current_forces[2], delta_time)
            output3 = 25 * PDController3.update(current_forces[3], delta_time)
            output4 = 25 * PDController4.update(current_forces[4], delta_time)
            output5 = 25 * PDController5.update(current_forces[5], delta_time)

            zaux.move(1, output1)
            zaux.move(3, output3)
            zaux.move(5, output5)
            zaux.move(0, output0)
            zaux.move(2, output2)
            zaux.move(4, output4)

            # 状态标识
            status_text = "正常"
            if i == 80:
                status_text = "基准记录"


            print(f"{status_text} 迭代 {i + 1:3d}: 力值=", end="")
            print(" | ".join(f"{f:>6.1f}N" for f in current_forces), end="")
            print(" | 编码器=", end="")
            print(" | ".join(f"{e:>8.2f}" for e in current_encoders))

            time.sleep(1)

            # 如果已记录数据，可以选择提前结束
            if stable_encoders is not None and i > 70:
                print("\n已收集实验数据，提前结束实验,统一输出实验结果")
                print("实验生成的随机力值:")
                print(
                    f"{target_forces[0]:.2f},{target_forces[1]:.2f},{target_forces[2]:.2f},{target_forces[3]:.2f},{target_forces[4]:.2f},{target_forces[5]:.2f}")
                print("基准力值：", end="")
                print(", ".join(f"{e:>8.2f}" for e in baseline_forces))
                print("基准编码器值：", end="")
                print(", ".join(f"{e:>8.2f}" for e in stable_encoders))
                # 创建单行CSV格式的输出
                csv_line = [
                    # 目标力值
                    f"{target_forces[0]:.2f}", f"{target_forces[1]:.2f}", f"{target_forces[2]:.2f}",
                    f"{target_forces[3]:.2f}", f"{target_forces[4]:.2f}", f"{target_forces[5]:.2f}",
                    # 基准力值
                    f"{baseline_forces[0]:.2f}", f"{baseline_forces[1]:.2f}", f"{baseline_forces[2]:.2f}",
                    f"{baseline_forces[3]:.2f}", f"{baseline_forces[4]:.2f}", f"{baseline_forces[5]:.2f}",
                    # 基准编码器
                    f"{stable_encoders[0]:.2f}", f"{stable_encoders[1]:.2f}", f"{stable_encoders[2]:.2f}",
                    f"{stable_encoders[3]:.2f}", f"{stable_encoders[4]:.2f}", f"{stable_encoders[5]:.2f}",
                ]

                # 打印单行CSV格式的数据
                print(",".join(csv_line))
                break
