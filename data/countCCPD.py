import matplotlib.pyplot as plt
import os
import os.path as osp
import argparse
from tqdm import tqdm

from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

pros = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', 'O']  # class names
alps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
adses = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '警', '学', 'O']

def getLabel(txt):
    assert osp.exists(txt), f"{txt}: txt path is invalid!"

    w = 720.
    h = 1160.

    file = open(txt, 'r')
    lines = file.readlines()
    labels = []
    for line in lines:
        line = line.strip("\n")
        label = line.split(' ')
        pro = int(label[0])
        alp = int(label[1])
        ad0 = int(label[2])
        ad1 = int(label[3])
        ad2 = int(label[4])
        ad3 = int(label[5])
        ad4 = int(label[6])
        box_x = float(label[7]) * w
        box_y = float(label[8]) * h
        box_w = float(label[9]) * w
        box_h = float(label[10]) * h
        cor_x1 = float(label[11]) * w
        cor_y1 = float(label[12]) * h
        cor_x2 = float(label[13]) * w
        cor_y2 = float(label[14]) * h
        cor_x3 = float(label[15]) * w
        cor_y3 = float(label[16]) * h
        cor_x4 = float(label[17]) * w
        cor_y4 = float(label[18]) * h
        label = [pro, alp, ad0, ad1, ad2, ad3, ad4, box_x, box_y, box_w, box_h, cor_x1, cor_y1, cor_x2, cor_y2, cor_x3, cor_y3, cor_x4, cor_y4]
        labels.append(label)
    return labels

def draw_cls(cnt, names, cls='pro', path='./data/config/train', color='blue'):
    assert len(cnt) == len(names), "names is not match with counts!"

    if not osp.exists(path):
        os.mkdir(path)
    fig, ax = plt.subplots()

    bar_labels = [color]
    bar_colors = ['tab:' + color]

    ax.bar(names, cnt)

    ax.set_ylabel('counts')
    ax.set_title(cls)

    file_path = osp.join(path, cls + ".jpg")

    plt.savefig(file_path)
    print(f"{cls} finished. {cls} count: {cnt}")

def run(path, output, task="train"):
    assert osp.exists(path), f"{path}: Dataset Path is invalid!"
    assert task == "train" or task == "val" or task == "test", task + " is invalid task name!"
    path = osp.join(osp.join(path, "labels"), task)
    assert osp.exists(path), f"{path} is not in dir!"
    assert osp.exists(osp.dirname(output)), f"{output} is an invalid output path!"

    if not osp.exists(output):
        os.mkdir(output)

    # cls count
    pro_cnt = [0 for i in range(32)]
    alp_cnt = [0 for i in range(25)]
    ads_cnt = [0 for i in range(37)]
    
    # point list
    box_x = []
    box_y = []
    box_w = []
    box_h = []
    cor_cx = []
    cor_cy = []

    txt_list = os.listdir(path)
    for txt in tqdm(txt_list, task):
        assert ".txt" in txt, "There are some files not in txt format!"
        txt_path = osp.join(path, txt)
        label = getLabel(txt_path)
        for l in label:
            pro_cnt[l[0]] += 1
            alp_cnt[l[1]] += 1
            for i in range(2, 7):
                ads_cnt[l[i]] += 1
            box_x.append(int(l[7]))
            box_y.append(int(l[8]))
            box_w.append(int(l[9]))
            box_h.append(int(l[10]))
            corner_cx = 0
            corner_cy = 0
            for i in range(8):
                if i % 2 == 0:
                    corner_cx += l[i + 11]
                else:
                    corner_cy += l[i + 11]
            cor_cx.append(int(corner_cx / 4.0))
            cor_cy.append(int(corner_cy / 4.0))
    print(f"{task}: count finished!")
    
    # save
    t_output = osp.join(output, task)

    # pro
    draw_cls(pro_cnt, pros, cls="pro", path=t_output)
    # alp
    draw_cls(alp_cnt, alps, cls="alp", path=t_output)
    # ads
    draw_cls(ads_cnt, adses, cls="ads", path=t_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source" , "-s", type = str, default = "", help = "path to data")
    parser.add_argument("--output" , "-o", type = str, default = "", help = "output path")
    opt = parser.parse_args()
    input_dir = opt.source
    output_dir = opt.output

    run(input_dir, output_dir, task="train")
    run(input_dir, output_dir, task="val")
    run(input_dir, output_dir, task="test")

if __name__ == "__main__":
    main()