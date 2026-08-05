import time
import numpy as np
from optical_tracker import opt
from pymodbus.client import ModbusSerialClient as ModbusClient
import ctypes
import platform
import time
import os
import plotly.io as pio
import sys
import logging

logging.getLogger('pymodbus').setLevel(logging.WARNING)
sys.path.append(r'C:\Users\a\OneDrive - City University of Hong Kong - Student\Code\src\exp')
dll_path = r'C:\Users\a\OneDrive - City University of Hong Kong - Student\Code\src\exp\zauxdll.dll'
zauxdll = ctypes.WinDLL(dll_path)

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

class MotionCap:
    def __init__(self, ip_address='10.1.1.198'):
        self.mocap = opt(ip_address)
        # define incalid_value
        self.INVALID_VALUE = 9.999999e+06
        self.TOLERANCE = 1e-4  # define torlerance

    def opt_get_marker_position(self, markerset, marker_index):
        marker = markerset.Markers[marker_index]
        return np.array([marker[0], marker[1], marker[2], 1])

    def is_invalid_position(self, position):
        # Check whether each coordinate is invalid
        return any(abs(coord - self.INVALID_VALUE) < self.TOLERANCE
                    for coord in position[:3])  # only check xyz coordinate

    def opt_get_all_marker_positions(self):
        '''
        :return: positions of all markers in the following order:
                 base1, ase2, mid1, mid2, mid3, mid4, mid5, mid6, tip1
        '''
        self.mocap.mainprocess()
        while True:
            markerset = self.mocap.getmarkerset()
            if markerset is not None:
                positions = [self.opt_get_marker_position(markerset, i) for i in range(9)]
                # 检查所有位置是否有无效值
                invalid_indices = [i for i, pos in enumerate(positions)
                                   if self.is_invalid_position(pos)]

                if invalid_indices:
                    # 如果发现无效值，生成警告信息
                    marker_names = ["base1", "base2", "mid1", "mid2", "mid3",
                                "mid4", "mid5", "mid6", "tip1"]
                    invalid_names = [marker_names[i] for i in invalid_indices]

                    print(f"\nWARNING: Motion capture data contains invalid values ({self.INVALID_VALUE})")
                    print(f"Problematic markers: {', '.join(invalid_names)}")
                    print("Please check marker placement and system configuration.\n")

                break
            time.sleep(0.01)
        return positions

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

# Usage
if __name__ == '__main__':
    # 连接力传感器ip
    client = ModbusClient(method='rtu', port='COM3',baudrate=19200, timeout=20)
    client.connect()
    log.debug("写保持寄存器并读回")
    rq = client.read_holding_registers(0x300, 12, 1)  # 06H写保持寄存器(起始寄存器号，值，从机号)->返回写的数值
    data = rq.encode()
    force0_read = data[1:5]
    force1_read = data[5:9]
    force2_read = data[9:13]
    force3_read = data[13:17]
    force4_read = data[17:21]
    force5_read = data[21:25]

    force1 = int.from_bytes(force1_read, 'big', signed=True)
    force3 = int.from_bytes(force3_read, 'big', signed=True)
    force5 = int.from_bytes(force5_read, 'big', signed=True)
    force0 = int.from_bytes(force0_read, 'big', signed=True)
    force2 = int.from_bytes(force2_read, 'big', signed=True)
    force4 = int.from_bytes(force4_read, 'big', signed=True)

    # print("force 0 current_value: {:.2f}".format(force0))
    # print("force 1 current_value: {:.2f}".format(force1))
    # print("force 2 current_value: {:.2f}".format(force2))
    # print("force 3 current_value: {:.2f}".format(force3))
    # print("force 4 current_value: {:.2f}".format(force4))
    # print("force 5 current_value: {:.2f}".format(force5))
    print(f"{force0:.2f},{force1:.2f},{force2:.2f},{force3:.2f},{force4:.2f},{force5:.2f}")


    mocap_handle = MotionCap()
    marker_positions = mocap_handle.opt_get_all_marker_positions()
    print('MotionCapture marker positions:', marker_positions)
    print(f"{force0:.2f},{force1:.2f},{force2:.2f},{force3:.2f},{force4:.2f},{force5:.2f}")
    print(marker_positions)


