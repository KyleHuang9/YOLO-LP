import os
import os.path as osp

def load_CCPD_data(txt):
    assert osp.exists(txt), f"{txt} is an invalid path!"
    root = osp.dirname(osp.dirname(txt))
    img_paths = []
    labels = []
    img_w = 720.
    img_h = 1160.

    file = open(txt, 'r')
    lines = file.readlines()
    for line in lines:
        # img_path
        img_path = line.strip('\n')
        img_path = osp.join(root, img_path)
        img_paths.append(img_path)

        # labels
        img_name = osp.basename(img_path)
        split_name = img_name.strip('.jpg').split('-')

        # box
        _box = split_name[2].split('_')
        _box_tl = _box[0].split('&')
        _box_br = _box[1].split('&')
        box_w = (int(_box_br[0]) - int(_box_tl[0])) / img_w
        box_h = (int(_box_br[1]) - int(_box_tl[1])) / img_h
        box_cx = int(int(_box_tl[0]) + 0.5 * box_w) / img_w
        box_cy = int(int(_box_tl[1]) + 0.5 * box_h) / img_h
        
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
        no = [int(_no[0]), int(_no[1]), int(_no[2]), int(_no[3]), int(_no[4]), int(_no[5]), int(_no[6])]

        label = [no[0], no[1], no[2], no[3], no[4], no[5], no[6],
                box_cx, box_cy, box_w, box_h,
                corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6], corners[7]]
        labels.append(label)
    return img_paths, labels