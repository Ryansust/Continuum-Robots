from nokov.nokovsdk import *
import time
import sys, getopt
import signal

preFrmNo = 0
curFrmNo = 0

class opt:
    def __init__(self, serverIp):
        self.serverIp = serverIp
        self.framedata = None
        self.markerset = None
        self.mainprocess_init()


    def py_data_func(self, pFrameOfMocapData, pUserData):

        if pFrameOfMocapData == None:
            print("Not get the data frame.\n")
        else:
            frameData = pFrameOfMocapData.contents

            self.framedata = frameData

            for iMarkerSet in range(frameData.nMarkerSets):
                markerset = frameData.MocapData[iMarkerSet]
                self.markerset = markerset


    def py_msg_func(self, iLogLevel, szLogMessage):
        szLevel = "None"
        if iLogLevel == 4:
            szLevel = "Debug"
        elif iLogLevel == 3:
            szLevel = "Info"
        elif iLogLevel == 2:
            szLevel = "Warning"
        elif iLogLevel == 1:
            szLevel = "Error"

        print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))


    def py_forcePlate_func(self, pFocePlates, pUserData):
        if pFocePlates == None:
            print("Not get the forcePlate frame.\n")
            pass
        else:
            ForcePlatesData = pFocePlates.contents
            print("iFrame:%d" % ForcePlatesData.iFrame)
            for iForcePlate in range(ForcePlatesData.nForcePlates):
                print("Fxyz:[%f,%f,%f] xyz:[%f,%f,%f] MFree:[%f]" % (
                    ForcePlatesData.ForcePlates[iForcePlate].Fxyz[0],
                    ForcePlatesData.ForcePlates[iForcePlate].Fxyz[1],
                    ForcePlatesData.ForcePlates[iForcePlate].Fxyz[2],
                    ForcePlatesData.ForcePlates[iForcePlate].xyz[0],
                    ForcePlatesData.ForcePlates[iForcePlate].xyz[1],
                    ForcePlatesData.ForcePlates[iForcePlate].xyz[2],
                    ForcePlatesData.ForcePlates[iForcePlate].Mfree
                ))
                print('hello world')

    def mainprocess_init(self):
        serverIp = self.serverIp
        self.client = PySDKClient()

        ver =  self.client.PySeekerVersion()
        print('SeekerSDK Sample Client 2.4.0.3142(SeekerSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

        self.client.PySetVerbosityLevel(0)
        self.client.PySetMessageCallback(self.py_msg_func)  # 设置消息和数据的回调函数
        self.client.PySetDataCallback(self.py_data_func, None)

        print("Begin to init the SDK Client")
        ret = self.client.Initialize(bytes(serverIp, encoding="utf8"))

        if ret == 0:
            print("Connect to the Seeker Succeed")
        else:
            print("Connect Failed: [%d]" % ret)
            exit(0)

        # Give 5 seconds to system to init forceplate device
        ret = self.client.PyWaitForForcePlateInit(5000)
        if (ret != 0):
            print("Init ForcePlate Failed[%d]" % ret)
            exit(0)

    def getframedata(self):
        return self.framedata

    def getmarkerset(self):
        return self.markerset


    def mainprocess(self):
        self.client.PySetForcePlateCallback(self.py_forcePlate_func, None)




# global preFrmNo, curFrmNo
# curFrmNo = frameData.iFrame
# if curFrmNo == preFrmNo:
#     return
#     preFrmNo = curFrmNo


# print("FrameNo: %d\tTimeStamp:%Ld" % (frameData.iFrame, frameData.iTimeStamp))
# print("nMarkerset = %d" % frameData.nMarkerSets)


#     print("Markerset%d: %s [nMarkers Count=%d]\n" % (iMarkerSet + 1, markerset.szName, markerset.nMarkers))
#     print("{\n")
#
#     for iMarker in range(markerset.nMarkers):
#         print("\tMarker%d: %3.2f,%3.2f,%3.2f\n" % (
#             iMarker,
#             markerset.Markers[iMarker][0],
#             markerset.Markers[iMarker][1],
#             markerset.Markers[iMarker][2]))
#     print("}\n")





