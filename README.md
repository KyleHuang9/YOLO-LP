# YOLO-LP

## 介绍

这是一个基于YOLOv6框架的全新的基于关键点的车牌检测算法。

目前算法仍在不断完善中，各版本模型将陆续上传。



## 快速开始

- __安装__

~~~shell
git clone https://github.com/KyleHuang9/YOLO-LP
cd YOLO-LP
pip install -r requirements.txt
~~~

- __数据集准备__

首先，自行下载CCPD2019数据集，然后运行下面的命令进行数据集转换。

~~~shell
python data/transCCPD.py --source1 [CCPD2019 Path] --source2 [CCPD2020 Path] -o [Output Path]
~~~

- __训练__

```shell
python tools/train.py --batch 32 --conf configs/yololps.py --data data/dataset.yaml --device 0 
```

- __测试__

~~~shell
python tools/eval.py --data data/dataset.yaml --batch 32 --weights [weighs-dir] --task val
~~~

- __推理__

~~~shell
python tools/infer.py --weights [weights-dir] --source [image/image-dir/video]
~~~

