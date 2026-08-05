import time
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import socket
import time
import numpy as np
from exp.zmotion import ZMCWrapper


class MotorControl:
    def __init__(self):
        self.zaux = ZMCWrapper()
        self.zaux.connect("192.168.0.11")  # 连接控制器ip   默认192.168.0.11

    def go_home(self):
        zaux = self.zaux
        axislist = [0, 1, 2, 3, 4, 5]

        settings = {
            'atype': 1,
            'units': 800,
            'accel': 200,
            'decel': 200,
            'speed': 50,
            'creep': 5000
        }

        for axis in axislist:
            for sets, value in settings.items():
                getattr(zaux, f'set_{sets}')(axis, value)
            zaux.SetInvertIn(axis, 0)
            zaux.set_merge(axis, 1)

        for axis in axislist:
            zaux.SetDatumIn(axis, axis)
            zaux.movehome(axis)

        while True:
            status = []
            for axis in axislist:
                status.append(zaux.get_home_status(axis))
            if all(zaux.get_home_status(axis) == 1 for axis in axislist):
                print("axis get_home finished!", status)
                time.sleep(5)
                break
            else:
                print('axis go home moving', status)
                time.sleep(1)

        for axis in axislist:
            zaux.set_dpos(axis, 0)

    def Q_control(self, Q_value, Q_i, L):
        '''
        use 6 Dofs q to control continuum robot
        :param Q_value: q value list for control
        :param Q_i: Initial position for each axis
        :param L: length of each segment
        :return:
        '''
        zaux = self.zaux
        axislist = [0, 1, 2, 3, 4, 5]
        len1 = L[0]
        len2 = L[0]+L[1]
        # parameters setting
        settings = {
            'atype': 1,
            'units': 800,
            'accel': 200,
            'decel': 200,
            'speed': 5,
            'creep': 5000
        }
        for axis in axislist:
            for sets, value in settings.items():
                getattr(zaux, f'set_{sets}')(axis, value)
            zaux.SetInvertIn(axis, 0)
            zaux.set_merge(axis, 1)

        coeff = [1, 1]
        l11, l12, l13 = Q_value[0], Q_value[2], Q_value[4]
        l21, l22, l23 = Q_value[1], Q_value[3], Q_value[5]

        q1 = Q_i[1] + (l11 - len1) * 25 * coeff[0]
        q3 = Q_i[3] + (l12 - len1) * 25 * coeff[0]
        q5 = Q_i[5] + (l13 - len1) * 25 * coeff[0]

        q0 = Q_i[0] + (l23 - len2) * 25 * coeff[1]
        q2 = Q_i[2] + (l21 - len2) * 25 * coeff[1]
        q4 = Q_i[4] + (l22 - len2) * 25 * coeff[1]

        print('q0, q1, q2, q3, q4, q5:', q0, q1, q2, q3, q4, q5)
        ########################
        # abs position control
        zaux.single_absmove(0, q0)
        zaux.single_absmove(2, q2)
        zaux.single_absmove(4, q4)
        zaux.single_absmove(1, q1)
        zaux.single_absmove(3, q3)
        zaux.single_absmove(5, q5)
        # time.sleep(5)

        while True:
            status = []
            for axis in axislist:
                status.append(zaux.get_dpos(axis))
            # print('status:', status)
            if all(zaux.checkmotion(axis) != 0 for axis in axislist):
                return True
            time.sleep(1)

        return False

    def go_balance(self, Q_i):
        zaux = self.zaux
        self.go_home()

        axislist = [0, 1, 2, 3, 4, 5]

        for axis in axislist:
            zaux.move(axis, Q_i[axis])

        while True:
            status = []
            for axis in axislist:
                status.append(zaux.get_dpos(axis))
            if all(zaux.get_dpos(axis) == Q_i[axis] for axis in axislist):
                print('Finished and we can send message:')
                time.sleep(2)
                break
            else:
                print('Not arrive balance position yet', status)
                time.sleep(1)

        return None

    def Z_Control_ini(self, z_ini):
        # gohomepos
        zaux = self.zaux
        # parameters setting
        settings = {
            'atype': 1,
            'units': 1600,
            'accel': 500,
            'decel': 500,
            'speed': 5,
            'creep': 500
        }
        for sets, value in settings.items():
            getattr(zaux, f'set_{sets}')(6, value)
            zaux.SetInvertIn(6, 0)
            zaux.set_merge(6, 1)
        zaux.SetDatumIn(6, 6)

        zaux.movehome(6)

        while True:
            if zaux.get_home_status(6) == 1:
                time.sleep(1)
                print("Z go_home finished!")
                break
            else:
                print("Z go_home moving!")
                time.sleep(5)
        zaux.set_dpos(6, 0)
        #
        zaux.move(6, z_ini)
        while True:
            print('Z initializing')
            if zaux.get_dpos(6).value == z_ini:
                print('Finish Z initializing and able to send message:')
                break
            time.sleep(1)
        return None

    def Z_Control_dof(self, Z_update, Z_i):
        # z轴电机上正下负
        zaux = self.zaux
        axislist = 6
        Z_value = Z_i + Z_update
        # 原点复位
        zaux.set_atype(axislist, 1)
        zaux.set_units(axislist, 1600)
        zaux.set_accel(axislist, 500)
        zaux.set_decel(axislist, 500)
        zaux.set_speed(axislist, 3)
        # 绝对位置移动
        zaux.single_absmove(6, Z_value)
        while True:
            time.sleep(1)
            if zaux.checkmotion(6) != 0:
                break
        print('Z电机绝对位置：', Z_value, '\n')
        return None

