# from email.utils import collapse_rfc2231_value
from ast import Return
import cv2
import numpy as np
import argparse
import os
import datetime
import shutil
from io import BytesIO

DX_RIGHT = 0
DX_UP = 1
DX_LEFT = 2
DX_DOWN = 3

class Point(object):
    x = 0
    y = 0
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def GetLinePara(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y


def GetCrossPoint(l1,l2):
    GetLinePara(l1)
    GetLinePara(l2)
    d = l1.a * l2.b - l2.a * l1.b
    p = Point()
    p.x = (l1.b * l2.c - l2.b * l1.c)*1.0 / d
    p.y = (l1.c * l2.a - l2.c * l1.a)*1.0 / d
    return p

def GetCrossAngle(l1, l2):
    arr_0 = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])
    arr_1 = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))   # 注意转成浮点数运算
    return np.arccos(cos_value) * (180/np.pi)

# angle = GetCrossAngle(line1, line2)
class MeteorCollector(object):
    """
    匹配
    """

    def __init__(self):
        self.drop_list = []
        self.activite_list = []
        self.waiting_list = []
        self.active_list = []


    def update_contours(self, contours, frame_idx, t_mat, img):
        if contours is None or len(contours)==0:
            return
        for contour in contours:
            in_droplist = False
            in_waitlist = False
            (x,y,w,h)=cv2.boundingRect(contour)
            for dp in self.drop_list:
                if abs(dp.x+dp.w-x) <= 2*dp.first_w and abs(dp.y+dp.h-y)<=2*dp.first_h:
                    dp.x = x
                    dp.y = y
                    dp.w = w
                    dp.h = h
                    in_droplist = True
                    break
            if in_droplist:
                continue

            for dp in self.waiting_list:
                # if (abs(dp.x-(x+w)) <= 2*dp.first_w or abs(dp.x+dp.w-x) <= 2*dp.first_w ) and (abs(dp.y-(y+h))<=2*dp.first_h or abs(dp.y+dp.h-y)<=2*dp.first_h):
                if (abs(dp.x-x) <= 2*dp.first_w and abs(dp.y-y) <= 2*dp.first_h):
                    # print(str(x)+","+str(y)+","+str(w)+","+str(h))
                    in_waitlist = True
                    dp.first_frame=False
                    dp.last_active_frame = frame_idx
                    # if dp.w >= w or dp.h >= h:
                    #     self.drop_list.append(dp)
                    #     self.waiting_list.remove(dp)
                    #     area = np.array([[dp.x, dp.y], [dp.x+w, dp.y], [dp.x+w, dp.y+h], [dp.x, dp.y+h]])
                    #     cv2.fillPoly(t_mat,[area], color=(0,0,0))
                        # print("xxx\n")
                    if dp.img is None:
                        dp.img = img.copy()
                    dp.x = x
                    dp.y = y
                    dp.w = w
                    dp.h = h
                    # if dp.h <= h+1 or dp.w <= w+1:
                    #     dp.count_same += 1
                    break

            if not in_waitlist:
                self.waiting_list.append(MeteorSeries(x, y, w, h,frame_idx))
        
        for dp in self.waiting_list:
            if dp.calInvalidFrame():
                self.drop_list.append(dp)
                area = np.array([[dp.x, dp.y], [dp.x+w, dp.y], [dp.x+w, dp.y+h], [dp.x, dp.y+h]])
                cv2.fillPoly(t_mat,[area], color=(0,0,0))
                self.waiting_list.remove(dp)
                continue
            if dp.calEndFrame(frame_idx):
                area = np.array([[dp.x, dp.y], [dp.x+w, dp.y], [dp.x+w, dp.y+h], [dp.x, dp.y+h]])
                cv2.fillPoly(t_mat,[area], color=(0,0,0))
                # self.active_list.append(dp)
                # self.waiting_list.remove(dp)
                return
    
    def checkOneFrame(self, pt):
        for dp in self.waiting_list:
            samearea = 0
            if pt[0] > dp.x and pt[0] < dp.first_x+dp.first_w and pt[1] > dp.y and pt[1] < (dp.y+dp.h):
                samearea += 1
            # print(samearea)
            if samearea > 0:
                self.waiting_list.remove(dp)
                return True
        
        return False

class MeteorSeries(object):
    """
    存储相应有信息
    """

    def __init__(self, x, y, w, h, frame_idx):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.first_x=x
        self.first_y=y
        self.first_w = w
        self.first_h = h
        self.angle = 0
        self.fang = 0
        self.type = 0
        self.first_frame = True
        self.count_same = 0
        self.first_idx = frame_idx
        self.last_active_frame = frame_idx
        self.img = None

    def calEndFrame(self, frame_idx):
        if frame_idx - self.last_active_frame > 10:
            return True
        return False

    def calInvalidFrame(self):
        line_point1 = Point(self.first_x,self.first_y)
        line_point2 = Point(self.first_x+self.first_w, self.first_y)
        line2 = Line(line_point1,line_point2)
        angle = 0
        #左下或右上
        if (self.x<self.first_x and self.y-self.first_y < 2) or (abs(self.x - self.first_x)<2 and self.y < self.first_y):
            if self.first_frame:
                line_rect1 = Point(self.first_x,self.first_y+self.first_h)
                line_rect2 = Point(self.first_x+self.first_w, self.y)
                line1 = Line(line_rect1,line_rect2)
                self.fang = GetCrossAngle(line1, line2)
                # print("fang:"+str(self.fang))
            else:
                self.type = 0
                line_rect1 = Point(self.x,self.y+self.h)
                line_rect2 = Point(self.x+self.w, self.y)
                line1 = Line(line_rect1,line_rect2)
                angle = GetCrossAngle(line1, line2)
                # print("angle:"+str(angle))

        #左上或右下
        elif (self.x<self.first_x and self.first_y<self.y) or (abs(self.x - self.first_x)<2 and abs(self.y - self.first_y)<2):
            if self.first_frame:
                line_rect1 = Point(self.first_x,self.first_y)
                line_rect2 = Point(self.first_x+self.first_w, self.first_y+self.first_h)
                line1 = Line(line_rect1,line_rect2)
                self.fang = GetCrossAngle(line1, line2)
                # print("fang2:"+str(self.fang))
            else:
                self.type=1
                line_rect1 = Point(self.x,self.y)
                line_rect2 = Point(self.x+self.w, self.y+self.h)
                line1 = Line(line_rect1,line_rect2)
                angle = GetCrossAngle(line1, line2)
                # print("angle2:"+str(angle))

        if not self.first_frame and angle is not None and (abs(angle-self.fang)> 20.0 or abs(angle-self.fang)< 0.1):
            # print(abs(angle-self.fang))
            return True

        return False


# init frame data collector

def filter_img(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # blur = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 15, 100, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    return dilated

def frame2ts(frames, framerate):
    # return datetime.datetime.strftime(
    #     datetime.datetime.utcfromtimestamp(frame / fps), "%H:%M:%S.%f")
    return '{0:02d}:{1:02d}:{2:02d}:{3:02d}'.format(int(frames / (3600 * framerate)),
      int(frames / (60 * framerate) % 60),
      int(frames / framerate % 60),
      int(frames % framerate))

def writeImg(path, frame):
    code = cv2.imencode('.jpg',frame)[1]  # 保存图片
    # byte_stream = BytesIO(code.tobytes())
    with open(path,'wb') as p: # 可以保存任意路径
        p.write(code.tobytes())
    
def detech_one_video(video_name, debug_mode):
    # cap = cv2.VideoCapture("/Users/zhangzhan/Desktop/1-3.mp4")
    meteorColler = MeteorCollector()
    cap = cv2.VideoCapture(video_name)
    total_frame, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    print("开始分析视频：%s"%(video_name))
    print("视频总时间为："+(frame2ts(total_frame, fps))+"\n")

    ret, frame1 = cap.read()
    if ret == False:
        print("读取视频失败\n")
        return
    # frame1 = cv2.resize(frame1, (1024, 768))
    ret, frame2 = cap.read()
    if ret == False:
        print("读取视频失败\n")
        return
    # frame2 = cv2.resize(frame2, (1024, 768))
    tmp = frame1.copy()
    tmp[:, :, 0] = np.ones([frame1.shape[0], frame1.shape[1]]) * 0
    tmp[:, :, 1] = np.ones([frame1.shape[0], frame1.shape[1]]) * 0
    tmp[:, :, 2] = np.ones([frame1.shape[0], frame1.shape[1]]) * 0
    cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    frame_count = 1
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)

        mask = filter_img(diff)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 10:
            for contour in contours:
                if cv2.contourArea(contour) < 120:
                    continue
                # pt =[contour[0][0][0],contour[0][0][1]]
                # if meteorColler.checkOneFrame(pt):
                #     continue
                for c in contour:
                    tmp[c[0][1],c[0][0]] = 255
                # tmp[y,x] = 255
                # tmp[contour[0][0][1],contour[0][0][0]] = 255
            # cv2.drawContours(frame1,contours,-1,(0,255,0),2)
        t = tmp.copy()
        t= cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
        t = cv2.dilate(t,None,iterations=4)
        contours,_=cv2.findContours(t,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        meteorColler.update_contours(contours, frame_count, tmp, frame1)
        t=cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

        if debug_mode:
            for pt in meteorColler.waiting_list:
                if not pt.first_frame:
                    cv2.rectangle(frame1,pt1=(pt.x,pt.y),pt2=(pt.x+pt.w,pt.y+pt.h),color=(0,255,0),thickness=2)
                    # print("(%d,%d0,(%d,%d)\n"%(pt.x,pt.y,pt.x+pt.w,pt.y))
                    # print("perhaps meteor: %s" %(frame2ts(pt.first_idx, fps)))
            cv2.imshow("Frame", frame1)
            cv2.imshow("tmp", t)
            if (cv2.waitKey(1) & 0xff == ord("q")):
                break
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        # frame2 = cv2.resize(frame2, (1024, 768))
        frame_count += 1
        

    cap.release()
    cv2.destroyAllWindows()

    print("视频%s发生流星的时间：\n"%(video_name))
    i=1
    resultPath= os.path.splitext(video_name)[0]
    resultFile = resultPath+"/result.txt"
    if os.path.exists(resultPath):
        shutil.rmtree(resultPath)
    os.mkdir(resultPath)
    # if os.path.exists(resultFile):
    #     os.remove(resultFile)
    file = open(resultFile, 'a+')
    for pt in meteorColler.waiting_list:
        # if not pt.first_frame:
        rlog = "第{}颗: 第{}帧，time:{}, area:(x,y)({},{})\n".format(i,pt.first_idx, frame2ts(pt.first_idx, fps), pt.first_x, pt.first_y)
        print(rlog)
        # print("第%d颗: %s, area:(x,y)(%d,%d)" %(i,frame2ts(pt.first_idx, fps), pt.first_x, pt.first_y))
        file.write(rlog)
        if pt.img is not None:
            cv2.rectangle(pt.img,pt1=(pt.x,pt.y),pt2=(pt.x+pt.w,pt.y+pt.h),color=(0,255,0),thickness=2)
            # cv2.imwrite(resultPath+"/第"+str(i)+"颗第"+str(pt.first_idx)+"帧.png", pt.img)
            writeImg(resultPath+"/第"+str(i)+"颗第"+str(pt.first_idx)+"帧.png", pt.img)
        i+=1
    print("\n")
    file.close()
    
def load_video(video_name, dir_name, debug_mode):
    if len(video_name) == 0 and len(dir_name) == 0:
        print("please input --video for one mp4 or --dir for mp4 dir")
        return

    if len(video_name):
        detech_one_video(video_name, debug_mode)
    elif len(dir_name) > 0:
        for root, dirs, files in os.walk(dir_name):
            for f in files:
                if ('.mp4' in f):
                    detech_one_video(os.path.join(root, f), debug_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meteor Detector V1.2')

    parser.add_argument('--video', help="input H264 video.", default="")
    parser.add_argument('--dir', help="input H264 video dir.",default="")
    parser.add_argument('--debug',
                        '-D',
                        action='store_true',
                        help="Apply Debug Mode",
                        default=False)

    args = parser.parse_args()

    video_name = args.video
    dir_name = args.dir
    debug_mode = args.debug
    
    load_video(video_name,
                 dir_name,
                 debug_mode)