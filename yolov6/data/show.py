import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

pro = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', 'O']
alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '警', '学', 'O']
color = [tuple(np.random.choice(range(256), size=3)) for _ in range(len(pro) * len(alp))]
path = "/home/hyl/home/LP_Detection/runs/image/"

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "/home/hyl/home/simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def show(img, label):
    idx = random.randint(0, 10000)
    print("img_shape: ", img.shape)
    print("label" + str(idx) + ": ", label)
    if len(label):
        for i in range(len(label)):
            box = label[i, 8:12]
            corners = label[i, 12:]
            pro_id = int(label[i, 0])
            alp_id = int(label[i, 1])
            ad0_id = int(label[i, 2])
            ad1_id = int(label[i, 3])
            ad2_id = int(label[i, 4])
            ad3_id = int(label[i, 5])
            ad4_id = int(label[i, 6])
            ad5_id = int(label[i, 7])
            img = cv2AddChineseText(img, f"{pro[pro_id]}{alp[alp_id]}{ads[ad0_id]}{ads[ad1_id]}{ads[ad2_id]}{ads[ad3_id]}{ads[ad4_id]}{ads[ad5_id]}",
                         (int(box[0]), int(box[1]) - 20), (255, 0, 0), 20)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
            cv2.line(img, (int(corners[0]), int(corners[1])), (int(corners[2]), int(corners[3])), (0, 255, 255), 2)
            cv2.line(img, (int(corners[2]), int(corners[3])), (int(corners[4]), int(corners[5])), (0, 255, 255), 2)
            cv2.line(img, (int(corners[4]), int(corners[5])), (int(corners[6]), int(corners[7])), (0, 255, 255), 2)
            cv2.line(img, (int(corners[6]), int(corners[7])), (int(corners[0]), int(corners[1])), (0, 255, 255), 2)
        cv2.namedWindow("img" + str(idx) + "_" + str(img.shape[0]) + "x" + str(img.shape[1]), cv2.WINDOW_NORMAL)
        cv2.imshow("img" + str(idx) + "_" + str(img.shape[0]) + "x" + str(img.shape[1]), img)
        cv2.waitKey()