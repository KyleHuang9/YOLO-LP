import numpy as np
import os.path as osp
import random
import argparse
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
import math

pro = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新']
alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '警', '学']

def AddSmudginess(img, box, corner, Smu):
    img_h, img_w = np.array(img).shape[:2]
    r_h, r_w = 50.0 / img_h, 50.0 / img_w
    rows = r(Smu.shape[0] - 50)
    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50]
    adder = cv2.resize(adder, (50, 50))
    #adder = cv2.bitwise_not(adder)
    img = cv2.resize(img, (50, 50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)

    box[:, 0] *= r_w
    box[:, 1] *= r_h
    corner[:, 0] *= r_w
    corner[:, 1] *= r_h

    return img, box, corner

def tfactor(img):
    """
    添加饱和度光照的噪声
    """
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def random_envirment(img, img_mask, env):
    """
    添加自然环境的噪声    
    """
    if img_mask is not None:
        bak = (img_mask==0)
        bak = bak.astype(np.uint8)*255
        bak = cv2.resize(bak, (env.shape[1], env.shape[0]))
        inv = cv2.bitwise_and(bak,env)
        img = cv2.bitwise_or(inv,img)
    else:
        bak = (img==0)
        bak = bak.astype(np.uint8)*255
        inv = cv2.bitwise_and(bak,env)
        img = cv2.bitwise_or(inv,img)
    return img

def GenCh(f,val):
    """
    生成中文字符
    """
    img=Image.new("RGB", (45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3),val,(0,0,0),font=f)
    img =  img.resize((23,70))
    A = np.array(img)
    return A

def GenCh1(f,val):
    """
    生成英文字符
    """
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val,(0,0,0),font=f)
    A = np.array(img)
    return A

def AddGauss(img, level):
    """
    添加高斯模糊
    """ 
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
    """
    添加高斯噪声
    """
    diff = 255-single.max()
    noise = np.random.normal(0,1+r(2),single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0])
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1])
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2])
    return img

def check_class(cls):
    assert len(cls) == 8, "class generate length is not right!"
    is_label = True
    for i in range(len(cls)):
        if i == 0 and cls[i] >= 32:
            is_label = False
            break
        if i == 1 and cls[i] >= 25:
            is_label = False
            break
        if i > 1 and cls[i] >= 37:
            is_label = False
            break
    return is_label



class Blue_Gen:
    def __init__(self, size=[272, 72]):
        self.fontC =  ImageFont.truetype("./yolov6/data/generate/font/platech.ttf", 43,0)
        self.fontE =  ImageFont.truetype('./yolov6/data/generate/font/platechar.ttf', 60,0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg  = cv2.resize(cv2.imread("./yolov6/data/generate/images/template.bmp"), (226, 70))
        self.smu = cv2.imread("./yolov6/data/generate/images/smu.jpg")
        self.noplates_path = []
        self.size = size

        # 蓝牌删除皖A
        self.pro = pro[1:31]
        self.alp = alp[1:]
        self.ads = ads[:34]
        for parent, parent_folder,filenames in os.walk("./yolov6/data/generate/NoPlates"):
            for filename in filenames:
                path = parent+"/"+filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset= 2 
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1])
        for i in range(5):
            base = offset+8+23+6+23+17 +i*23 + i*6 
            self.img[0:70, base  : base+23]= GenCh1(self.fontE,val[i+2])
        return self.img

    def genPlateString(self):
        '''
        生成车牌String
        生成车牌标签list
        '''
        plateStr = ""
        plateList=[]
        box = [0,0,0,0,0,0,0,0]
        for unit, cpos in zip(box,range(len(box))):
            if cpos == 0:
                pro_idx = r(len(self.pro))
                plateStr += self.pro[pro_idx]
                plateList.append(pro_idx+1)
            elif cpos == 1:
                alp_idx = r(len(self.alp))
                plateStr += self.alp[alp_idx]
                plateList.append(alp_idx+1)
            elif cpos == 7:
                plateList.append(36)
            else:
                ads_idx = r(len(self.ads))
                plateStr += self.ads[ads_idx]
                plateList.append(ads_idx)
        return plateStr, plateList

    def generate(self):
        text, cls = self.genPlateString()
        is_label = check_class(cls)
        assert is_label == True, "class generate error!"
        assert len(text) == 7, "[Generate] string length is invalid!"
        fg = self.draw(text)
        fg = cv2.bitwise_not(fg)
        box = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        corner = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        com = cv2.bitwise_or(fg, self.bg)
        com = tfactor(com)
        com = AddGauss(com, 1 + r(2))
        com = addNoise(com)
        #com, box, corner = AddSmudginess(com, box, corner, self.smu)

        # resize img
        r_w = self.size[0] / np.array(com).shape[1]
        r_h = self.size[1] / np.array(com).shape[0]
        com = cv2.resize(com, self.size, interpolation=cv2.INTER_LINEAR)
        box[:, 0] *= r_w
        box[:, 1] *= r_h
        corner[:, 0] *= r_w
        corner[:, 1] *= r_h
        
        cls = np.array(cls).reshape(8, 1).T

        # create new box
        box = box.reshape(1, 8)
        x = box[:, [0, 2, 4, 6]]
        y = box[:, [1, 3, 5, 7]]
        new_box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

        corner = corner.reshape(1, 8)
        
        com = np.array(com)
        label = np.concatenate([cls, new_box, corner], axis=1)
        return com, label, None

class Green_S_Gen:
    def __init__(self, size=[272, 72]):
        self.fontC =  ImageFont.truetype("./yolov6/data/generate/font/platech.ttf", 43,0)
        self.fontE =  ImageFont.truetype('./yolov6/data/generate/font/platechar.ttf', 60,0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg  = cv2.resize(cv2.imread("./yolov6/data/generate/images/g1.jpg"), (226, 70))
        self.smu = cv2.imread("./yolov6/data/generate/images/smu.jpg")
        self.noplates_path = []
        self.size = size

        self.pro = pro
        self.alp = alp
        self.ads = ads
        for parent, parent_folder,filenames in os.walk("./yolov6/data/generate/NoPlates"):
            for filename in filenames:
                path = parent+"/"+filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset= 2 
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1])
        for i in range(6):
            base = offset+8+23+6+23+17 +i*23
            self.img[0:70, base : base+23] = GenCh1(self.fontE, val[i+2])
        return self.img

    def genPlateString(self):
        '''
        生成车牌String
        生成车牌标签list
        '''
        plateStr = ""
        plateList=[]
        box = [0,0,0,0,0,0,0,0]
        for unit, cpos in zip(box,range(len(box))):
            if cpos == 0:
                pro_idx = r(len(self.pro))
                plateStr += self.pro[pro_idx]
                plateList.append(pro_idx)
            elif cpos == 1:
                alp_idx = r(len(self.alp))
                plateStr += self.alp[alp_idx]
                plateList.append(alp_idx)
            elif cpos == 2:
                # 新能源小车第三位特殊规则
                t_idx = r(2)
                if t_idx == 0:
                    plateStr += 'D'
                    plateList.append(3)
                else:
                    plateStr += 'F'
                    plateList.append(5)
            elif cpos == 3:
                ads_idx = r(len(self.ads) - 2)
                plateStr += self.ads[ads_idx]
                plateList.append(ads_idx)
            else:
                ads_idx = r(10) + 24
                plateStr += self.ads[ads_idx]
                plateList.append(ads_idx)
        return plateStr, plateList

    def generate(self):
        text, cls = self.genPlateString()
        is_label = check_class(cls)
        assert is_label == True, "class generate error!"
        assert len(text) == 8, "[Generate] string length is invalid!"
        fg = self.draw(text)
        box = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        corner = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        com = cv2.bitwise_and(fg, self.bg)
        fg = cv2.bitwise_not(fg)
        env_mask = cv2.bitwise_or(fg, self.bg)
        com = tfactor(com)
        env_mask = tfactor(env_mask)
        com = AddGauss(com, 1 + r(2))
        com = addNoise(com)
        #com, box, corner = AddSmudginess(com, box, corner, self.smu)

        # resize img
        r_w = self.size[0] / np.array(com).shape[1]
        r_h = self.size[1] / np.array(com).shape[0]
        com = cv2.resize(com, self.size, interpolation=cv2.INTER_LINEAR)
        env_mask = cv2.resize(env_mask, self.size, interpolation=cv2.INTER_LINEAR)
        box[:, 0] *= r_w
        box[:, 1] *= r_h
        corner[:, 0] *= r_w
        corner[:, 1] *= r_h
        
        cls = np.array(cls).reshape(8, 1).T

        # create new box
        box = box.reshape(1, 8)
        x = box[:, [0, 2, 4, 6]]
        y = box[:, [1, 3, 5, 7]]
        new_box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

        corner = corner.reshape(1, 8)
        
        com = np.array(com)
        label = np.concatenate([cls, new_box, corner], axis=1)
        return com, label, env_mask

class Green_B_Gen:
    def __init__(self, size=[272, 72]):
        self.fontC =  ImageFont.truetype("./yolov6/data/generate/font/platech.ttf", 43,0)
        self.fontE =  ImageFont.truetype('./yolov6/data/generate/font/platechar.ttf', 60,0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg  = cv2.resize(cv2.imread("./yolov6/data/generate/images/g2.jpg"), (226, 70))
        self.smu = cv2.imread("./yolov6/data/generate/images/smu.jpg")
        self.noplates_path = []
        self.size = size

        self.pro = pro
        self.alp = alp
        self.ads = ads
        for parent, parent_folder,filenames in os.walk("./yolov6/data/generate/NoPlates"):
            for filename in filenames:
                path = parent+"/"+filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset= 2 
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1])
        for i in range(6):
            base = offset+8+23+6+23+17 +i*23
            self.img[0:70, base : base+23] = GenCh1(self.fontE, val[i+2])
        return self.img

    def genPlateString(self):
        '''
        生成车牌String
        生成车牌标签list
        '''
        plateStr = ""
        plateList=[]
        box = [0,0,0,0,0,0,0,0]
        for unit, cpos in zip(box,range(len(box))):
            if cpos == 0:
                pro_idx = r(len(self.pro))
                plateStr += self.pro[pro_idx]
                plateList.append(pro_idx)
            elif cpos == 1:
                alp_idx = r(len(self.alp))
                plateStr += self.alp[alp_idx]
                plateList.append(alp_idx)
            elif cpos == 7:
                # 新能源小车第三位特殊规则
                t_idx = r(2)
                if t_idx == 0:
                    plateStr += 'D'
                    plateList.append(3)
                else:
                    plateStr += 'F'
                    plateList.append(5)
            else:
                ads_idx = r(10) + 24
                plateStr += self.ads[ads_idx]
                plateList.append(ads_idx)
        return plateStr, plateList

    def generate(self):
        text, cls = self.genPlateString()
        assert len(text) == 8, "[Generate] string length is invalid!"
        fg = self.draw(text)
        box = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        corner = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        com = cv2.bitwise_and(fg, self.bg)
        fg = cv2.bitwise_not(fg)
        env_mask = cv2.bitwise_or(fg, self.bg)
        com = tfactor(com)
        env_mask = tfactor(env_mask)
        com = AddGauss(com, 1 + r(2))
        com = addNoise(com)
        #com, box, corner = AddSmudginess(com, box, corner, self.smu)

        # resize img
        r_w = self.size[0] / np.array(com).shape[1]
        r_h = self.size[1] / np.array(com).shape[0]
        com = cv2.resize(com, self.size, interpolation=cv2.INTER_LINEAR)
        env_mask = cv2.resize(env_mask, self.size, interpolation=cv2.INTER_LINEAR)
        box[:, 0] *= r_w
        box[:, 1] *= r_h
        corner[:, 0] *= r_w
        corner[:, 1] *= r_h
        
        cls = np.array(cls).reshape(8, 1).T

        # create new box
        box = box.reshape(1, 8)
        x = box[:, [0, 2, 4, 6]]
        y = box[:, [1, 3, 5, 7]]
        new_box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

        corner = corner.reshape(1, 8)
        
        com = np.array(com)
        label = np.concatenate([cls, new_box, corner], axis=1)
        return com, label, env_mask

class Yel_S_Gen:
    def __init__(self, size=[272, 72]):
        self.fontC =  ImageFont.truetype("./yolov6/data/generate/font/platech.ttf", 43,0)
        self.fontE =  ImageFont.truetype('./yolov6/data/generate/font/platechar.ttf', 60,0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg  = cv2.resize(cv2.imread("./yolov6/data/generate/images/y1.bmp"), (226, 70))
        self.smu = cv2.imread("./yolov6/data/generate/images/smu.jpg")
        self.noplates_path = []
        self.size = size

        self.pro = pro
        self.alp = alp
        self.ads = ads
        for parent, parent_folder,filenames in os.walk("./yolov6/data/generate/NoPlates"):
            for filename in filenames:
                path = parent+"/"+filename
                self.noplates_path.append(path)

    def draw(self, val):
        offset= 2 
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1])
        for i in range(5):
            if val[i+2] != self.ads[35]:
                base = offset+8+23+6+23+17 +i*23 + i*6 
                self.img[0:70, base  : base+23]= GenCh1(self.fontE,val[i+2])
            else:
                base = offset+8+23+6+23+17 +i*23 + i*6 
                self.img[0:70, base  : base+23]= GenCh(self.fontC,val[i+2])
        return self.img

    def genPlateString(self):
        '''
        生成车牌String
        生成车牌标签list
        '''
        plateStr = ""
        plateList=[]
        box = [0,0,0,0,0,0,0,0]
        for unit, cpos in zip(box,range(len(box))):
            if cpos == 0:
                pro_idx = r(len(self.pro))
                plateStr += self.pro[pro_idx]
                plateList.append(pro_idx)
            elif cpos == 1:
                alp_idx = r(len(self.alp))
                plateStr += self.alp[alp_idx]
                plateList.append(alp_idx)
            elif cpos == 7:
                plateList.append(36)
            elif cpos == 6:
                # 最后一位可能为“学”
                ads_idx = r(len(self.ads))
                while ads_idx == 34:
                    ads_idx = r(len(self.ads))
                plateStr += self.ads[ads_idx]
                plateList.append(ads_idx)
            else:
                ads_idx = r(len(self.ads) - 2)
                plateStr += self.ads[ads_idx]
                plateList.append(ads_idx)
        return plateStr, plateList

    def generate(self):
        text, cls = self.genPlateString()
        is_label = check_class(cls)
        assert is_label == True, "class generate error!"
        assert len(text) == 7, "[Generate] string length is invalid!"
        fg = self.draw(text)
        box = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        corner = np.array([[0.0, 0.0], [0.0, fg.shape[0]], [fg.shape[1], fg.shape[0]], [fg.shape[1], 0.0]])
        com = cv2.bitwise_and(fg, self.bg)
        fg = cv2.bitwise_not(fg)
        env_mask = cv2.bitwise_or(fg, self.bg)
        com = tfactor(com)
        env_mask = tfactor(env_mask)
        com = AddGauss(com, 1 + r(2))
        com = addNoise(com)
        #com, box, corner = AddSmudginess(com, box, corner, self.smu)

        # resize img
        r_w = self.size[0] / np.array(com).shape[1]
        r_h = self.size[1] / np.array(com).shape[0]
        com = cv2.resize(com, self.size, interpolation=cv2.INTER_LINEAR)
        env_mask = cv2.resize(env_mask, self.size, interpolation=cv2.INTER_LINEAR)
        box[:, 0] *= r_w
        box[:, 1] *= r_h
        corner[:, 0] *= r_w
        corner[:, 1] *= r_h
        
        cls = np.array(cls).reshape(8, 1).T

        # create new box
        box = box.reshape(1, 8)
        x = box[:, [0, 2, 4, 6]]
        y = box[:, [1, 3, 5, 7]]
        new_box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

        corner = corner.reshape(1, 8)
        
        com = np.array(com)
        label = np.concatenate([cls, new_box, corner], axis=1)
        return com, label, env_mask

class generate:
    def __init__(self, max_num=2):
        self.max_mum = max_num
        self.blue = Blue_Gen()
        self.gs = Green_S_Gen()
        self.gb = Green_B_Gen()
        self.yel = Yel_S_Gen()

    def generate_one(self, img, label):
        assert len(label.shape) == 2, "Label is invalid!"

        num = random.randint(0, len(label))
        for idx in range(num):
            corner = label[idx, 12:]
            ltx = int(corner[0])
            lty = int(corner[1])
            lbx = int(corner[2])
            lby = int(corner[3])
            rbx = int(corner[4])
            rby = int(corner[5])
            rtx = int(corner[6])
            rty = int(corner[7])

            x_min = min(ltx, lbx, rbx, rtx)
            y_min = min(lty, lby, rby, rty)
            x_max = max(ltx, lbx, rbx, rtx)
            y_max = max(lty, lby, rby, rty)

            ltx = ltx - x_min
            lty = lty - y_min
            lbx = lbx - x_min
            lby = lby - y_min
            rbx = rbx - x_min
            rby = rby - y_min
            rtx = rtx - x_min
            rty = rty - y_min

            ori_pts = np.float32([float(ltx), float(lty), float(lbx), float(lby), float(rbx), float(rby), float(rtx), float(rty)]).reshape(4, 2)

            background = img[y_min:y_max, x_min:x_max]

            t = random.random()
            if t <= 0.48:
                plate, p_label, mask = self.blue.generate()
            elif t > 0.48 and t <= 0.8:
                plate, p_label, mask = self.gs.generate()
            elif t > 0.8 and t <= 0.92:
                plate, p_label, mask = self.yel.generate()
            else:
                plate, p_label, mask = self.gb.generate()

            plate_pts = np.float32(p_label[0, 12:]).reshape(4, 2)
            shape = background.shape
            size = [shape[1], shape[0]]

            # transform
            M  = cv2.getPerspectiveTransform(plate_pts, ori_pts)
            dst = cv2.warpPerspective(plate, M, size)
            if mask is not None:
                mask = cv2.warpPerspective(mask, M, size)
            plate_pts = plate_pts.reshape(-1, 1, 2)
            plate_pts = cv2.perspectiveTransform(plate_pts, M).reshape(4, 2).reshape(1, 8)
            p_label[:, 12:] = plate_pts

            dst = random_envirment(dst, mask, background)
            img[y_min:y_max, x_min:x_max] = dst

            label[idx, :8] = p_label[:, :8]

        return img, label