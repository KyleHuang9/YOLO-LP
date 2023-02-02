from yolov6.data.generate.utils import *

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
        com, env_mask, box, corner = rot_mask(com, env_mask,  box, corner, r(60)-30, com.shape, 30)
        com, env_mask, box, corner = rotRandrom_mask(com, env_mask, box, corner, 10, (com.shape[1],com.shape[0]))
        com = tfactor(com)
        env_mask = tfactor(env_mask)
        com = random_envirment_mask(com, env_mask, self.noplates_path)
        com = AddGauss(com, 1 + r(4))
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
        return com, label