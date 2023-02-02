import os
import os.path as osp
import argparse
from shutil import copy
from tqdm import tqdm

def transdatas_2019(path, output, train_idx=0, val_idx=0, test_idx=0):
    assert osp.exists(path), f"{path} is an invalid path!"
    subdir = "splits"
    sub_path = osp.join(path, subdir)
    assert osp.exists(sub_path), "Dataset format is not right!"
    assert osp.exists(osp.dirname(output)), f"{output} is not an invalid output path!"

    if not osp.exists(output):
        os.mkdir(output)

    train_idx = trans_one(path, output, key="train", idx=train_idx)
    val_idx = trans_one(path, output, key="val", idx=val_idx)
    test_idx = trans_one(path, output, key="test", idx=test_idx)

    print("*" * 15 + "\n    finish!    \n" + "*" * 15)
    return train_idx, val_idx, test_idx

def transdatas_2020(path, output, train_idx, val_idx, test_idx):
    assert osp.exists(path), f"{path} is an invalid path!"
    subdir = "ccpd_green"
    sub_path = osp.join(path, subdir)
    assert osp.exists(sub_path), "Dataset format is not right!"
    assert osp.exists(osp.dirname(output)), f"{output} is not an invalid output path!"

    if not osp.exists(output):
        os.mkdir(output)
    
    dir = ['train', 'val', 'test']
    index = [train_idx, val_idx, test_idx]
    for i in range(3):
        key = dir[i]
        idx = index[i]
        output_image, output_label = creat_output_dir(output, key)
        t_dir = osp.join(sub_path, key)
        imgs = os.listdir(t_dir)
        for img in tqdm(imgs, key):
            img_path = osp.join(t_dir, img)
            label = getlabel2020(img_path)

            # label
            file_name = key + str(idx) + ".txt"
            output_txt = osp.join(output_label, file_name)
            output_txt = open(output_txt, 'w')
            if label is not None:
                output_txt.write(str(label[0]))
                for j in range(1, len(label)):
                    output_txt.write(" " + str(label[j]))
            output_txt.close()

            # img
            img_name = key + str(idx) + ".jpg"
            output_img = osp.join(output_image, img_name)
            copy(img_path, output_img)

            idx += 1
        index[i] = idx
        print()
    print("*" * 15 + "\n    finish!    \n" + "*" * 15)
    return index[0], index[1], index[2]


def trans_one(root, path, key, idx):
    txt_path = osp.join(root, "splits")
    txt_file = key + ".txt"
    txt_path = osp.join(txt_path, txt_file)
    assert osp.exists(txt_path), "No txt file!"

    output_image, output_label = creat_output_dir(path, key)

    file = open(txt_path, 'r')
    lines = file.readlines()

    for line in tqdm(lines, key):
        img_path = line.strip('\n')
        img_path = osp.join(root, img_path)
        label = getlabel2019(img_path)

        # label
        file_name = key + str(idx) + ".txt"
        output_txt = osp.join(output_label, file_name)
        output_txt = open(output_txt, 'w')
        if label is not None:
            output_txt.write(str(label[0]))
            for i in range(1, len(label)):
                output_txt.write(" " + str(label[i]))
        output_txt.close()

        # img
        img_name = key + str(idx) + ".jpg"
        output_img = osp.join(output_image, img_name)
        copy(img_path, output_img)

        idx += 1
    print()
    return idx

def getlabel2019(img_path):
    """get label from image name"""
    img_w = 720.
    img_h = 1160.
    # labels
    img_name = osp.basename(img_path)
    split_name = img_name.strip('.jpg').split('-')

    # box
    _box = split_name[2].split('_')
    _box_tl = _box[0].split('&')
    _box_br = _box[1].split('&')
    box_w = float(_box_br[0]) - float(_box_tl[0])
    box_h = float(_box_br[1]) - float(_box_tl[1])
    box_cx = (float(float(_box_tl[0]) + 0.5 * box_w)) / img_w
    box_cy = (float(float(_box_tl[1]) + 0.5 * box_h)) / img_h
    box_w = box_w / img_w
    box_h = box_h / img_h
    
    # corners
    _corners = split_name[3].split('_')
    _corners_br = _corners[0].split('&')
    _corners_bl = _corners[1].split('&')
    _corners_tl = _corners[2].split('&')
    _corners_tr = _corners[3].split('&')
    corners = [int(_corners_tl[0]) / img_w, int(_corners_tl[1]) / img_h,
            int(_corners_bl[0]) / img_w, int(_corners_bl[1]) / img_h,
            int(_corners_br[0]) / img_w, int(_corners_br[1]) / img_h,
            int(_corners_tr[0]) / img_w, int(_corners_tr[1]) / img_h]
    
    # no
    _no = split_name[4].split('_')
    no = [int(_no[0]), int(_no[1]), int(_no[2]), int(_no[3]), int(_no[4]), int(_no[5]), int(_no[6]), 36]
    box = [box_cx, box_cy, box_w, box_h]

    is_label = check_label_2019(no, box, corners, img_path)

    if is_label:
        label = [no[0], no[1], no[2], no[3], no[4], no[5], no[6], no[7],
                box_cx, box_cy, box_w, box_h,
                corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6], corners[7]]
    else:
        label = None
    return label

def getlabel2020(img_path):
    """get label from image name"""
    img_w = 720.
    img_h = 1160.
    # labels
    img_name = osp.basename(img_path)
    split_name = img_name.strip('.jpg').split('-')

    # box
    _box = split_name[2].split('_')
    _box_tl = _box[0].split('&')
    _box_br = _box[1].split('&')
    box_w = float(_box_br[0]) - float(_box_tl[0])
    box_h = float(_box_br[1]) - float(_box_tl[1])
    box_cx = (float(float(_box_tl[0]) + 0.5 * box_w)) / img_w
    box_cy = (float(float(_box_tl[1]) + 0.5 * box_h)) / img_h
    box_w = box_w / img_w
    box_h = box_h / img_h
    
    # corners
    _corners = split_name[3].split('_')
    _corners_br = _corners[0].split('&')
    _corners_bl = _corners[1].split('&')
    _corners_tl = _corners[2].split('&')
    _corners_tr = _corners[3].split('&')
    corners = [int(_corners_tl[0]) / img_w, int(_corners_tl[1]) / img_h,
            int(_corners_bl[0]) / img_w, int(_corners_bl[1]) / img_h,
            int(_corners_br[0]) / img_w, int(_corners_br[1]) / img_h,
            int(_corners_tr[0]) / img_w, int(_corners_tr[1]) / img_h]
    
    # no
    _no = split_name[4].split('_')
    no = [int(_no[0]), int(_no[1]), int(_no[2]), int(_no[3]), int(_no[4]), int(_no[5]), int(_no[6]), int(_no[7])]
    box = [box_cx, box_cy, box_w, box_h]

    is_label = check_label_2020(no, box, corners, img_path)

    if is_label:
        label = [no[0], no[1], no[2], no[3], no[4], no[5], no[6], no[7],
                box_cx, box_cy, box_w, box_h,
                corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6], corners[7]]
    else:
        label = None
    return label

def check_label_2019(no, box, corner, img_path):
    assert len(no) == 8, "class length is invalid!"
    assert len(box) == 4, "box length is invalid!"
    assert len(corner) == 8, "corner length is invalid!"

    is_label = True

    # check number
    for i in range(len(no)):
        if i == 0 and no[i] >= 31:
            is_label = False
            print(f"{img_path}: {no[i]} -- province number is invalid!")
        if i == 1 and no[i] >= 24:
            is_label = False
            print(f"{img_path}: {no[i]} -- alphabet number is invalid!")
        if i > 1 and i < 7 and no[i] >= 34:
            is_label = False
            print(f"{img_path}: {no[i]} -- ads number is invalid!")
        if i == 7 and no[i] > 36:
            is_label = False
            print(f"{img_path}: {no[i]} -- last number is invalid!")
    
    return is_label

def check_label_2020(no, box, corner, img_path):
    assert len(no) == 8, "class length is invalid!"
    assert len(box) == 4, "box length is invalid!"
    assert len(corner) == 8, "corner length is invalid!"

    is_label = True

    # check number
    for i in range(len(no)):
        if i == 0 and no[i] >= 31:
            is_label = False
            print(f"{img_path}: {no[i]} -- province number is invalid!")
        if i == 1 and no[i] >= 24:
            is_label = False
            print(f"{img_path}: {no[i]} -- alphabet number is invalid!")
        if i > 1 and no[i] >= 34:
            is_label = False
            print(f"{img_path}: {no[i]} -- ads number is invalid!")
    
    return is_label

def creat_output_dir(path, key):
    img_dir = osp.join(path, "images")
    if not osp.exists(img_dir):
        os.mkdir(img_dir)
    label_dir = osp.join(path, "labels")
    if not osp.exists(label_dir):
        os.mkdir(label_dir)
    
    output_img = osp.join(img_dir, key)
    output_label = osp.join(label_dir, key)

    if not osp.exists(output_img):
        os.mkdir(output_img)
    if not osp.exists(output_label):
        os.mkdir(output_label)
    return output_img, output_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type = str, default = None, help = "path of CCPD2019")
    parser.add_argument("--source2", type = str, default = None, help = "path of CCPD2020")
    parser.add_argument("--output" , "-o", type = str, default = "", help = "output path")
    opt = parser.parse_args()
    input_dir_2019 = opt.source1
    input_dir_2020 = opt.source2
    output_dir = opt.output

    if input_dir_2019 is not None:
        train_idx, val_idx, test_idx = transdatas_2019(input_dir_2019, output_dir)
        print("CCPD2019 total count: ", train_idx, val_idx, test_idx)
    if input_dir_2020 is not None:
        train_idx, val_idx, test_idx = transdatas_2020(input_dir_2020, output_dir, train_idx, val_idx, test_idx)
        print("CCPD2020 total count: ", train_idx, val_idx, test_idx)

if __name__ == "__main__":
    main()
    