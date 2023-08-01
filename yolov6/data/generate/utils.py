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

def rot(img, box, corner, angel, shape, max_angel):
    """ 
        添加放射畸变
        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸
    """
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * math.cos((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(math.sin((float(angel) / 180) * 3.14) * shape[0]))
    pts1 = np.float32([[0, 0],[0, size_o[1]],[size_o[0], 0],[size_o[0], size_o[1]]])
    if(angel > 0):
        pts2 = np.float32([[interval, 0],[0, size[1]], [size[0], 0], [size[0] - interval,size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    M  = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    
    box = box.reshape(-1, 1, 2)
    corner = corner.reshape(-1, 1, 2)

    box = cv2.perspectiveTransform(box, M).reshape(4, 2)
    corner = cv2.perspectiveTransform(corner, M).reshape(4, 2)

    return dst, box, corner

def rotRandrom(img, box, corner, factor, size):
    """
    添加透视畸变
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    box = box.reshape(-1, 1, 2)
    corner = corner.reshape(-1, 1, 2)

    box = cv2.perspectiveTransform(box, M).reshape(4, 2)
    corner = cv2.perspectiveTransform(corner, M).reshape(4, 2)

    return dst, box, corner\
    
def rot_mask(img, mask,  box, corner, angel, shape, max_angel):
    """ 
        添加放射畸变
        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸
    """
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * math.cos((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(math.sin((float(angel) / 180) * 3.14) * shape[0]))
    pts1 = np.float32([[0, 0],[0, size_o[1]],[size_o[0], 0],[size_o[0], size_o[1]]])
    if(angel > 0):
        pts2 = np.float32([[interval, 0],[0, size[1]], [size[0], 0], [size[0] - interval,size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    M  = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    mask = cv2.warpPerspective(mask, M, size)
    
    box = box.reshape(-1, 1, 2)
    corner = corner.reshape(-1, 1, 2)

    box = cv2.perspectiveTransform(box, M).reshape(4, 2)
    corner = cv2.perspectiveTransform(corner, M).reshape(4, 2)

    return dst, mask, box, corner

def rotRandrom_mask(img, mask, box, corner, factor, size):
    """
    添加透视畸变
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    mask = cv2.warpPerspective(mask, M, size)

    box = box.reshape(-1, 1, 2)
    corner = corner.reshape(-1, 1, 2)

    box = cv2.perspectiveTransform(box, M).reshape(4, 2)
    corner = cv2.perspectiveTransform(corner, M).reshape(4, 2)

    return dst, mask, box, corner

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

def random_envirment(img, data_set):
    """
    添加自然环境的噪声    
    """
    index=r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env,(img.shape[1],img.shape[0]))
    bak = (img==0)
    bak = bak.astype(np.uint8)*255
    inv = cv2.bitwise_and(bak,env)
    img = cv2.bitwise_or(inv,img)
    return img

def random_envirment_mask(img, img_mask, data_set):
    """
    添加自然环境的噪声    
    """
    index=r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env,(img.shape[1],img.shape[0]))
    bak = (img_mask==0)
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
    noise = np.random.normal(0,1+r(6),single.shape)
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
        if i == 0 and cls[i] >= 31:
            is_label = False
            break
        if i == 1 and cls[i] >= 24:
            is_label = False
            break
        if i > 1 and cls[i] >= 37:
            is_label = False
            break
    return is_label