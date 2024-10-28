# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 15:31
# @Author  : zhoujun

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np
import argparse


def get_datalist(data_dir, data_path, max_len):
    """
    获取训练和验证的数据list
    :param data_dir: 数据集根目录
    :param data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    if isinstance(data_path, list):
        for p in data_path:
            train_data.extend(get_datalist(data_dir, p, max_len))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc=f"load data from {data_path}"):
                line = line.strip("\n").replace(".jpg ", ".jpg\t").replace(".png ", ".png\t").split("\t")
                if len(line) > 1:
                    img_path = os.path.join(data_dir, line[0].strip(" "))
                    label = line[1]
                    if len(label) > max_len:
                        continue
                    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        train_data.append([str(img_path), label])
    return train_data


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list, desc=f"make dataset, save to {outputPath}"):
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                continue

        imageKey = "image-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/valid_data")
    parser.add_argument("--data_list", default="data/valid_data/valid_list_v2.txt")
    parser.add_argument("--max_len", default=800, type=int)
    parser.add_argument("--save_path", default="./train_data/data_lmdb_release/validation/")
    return parser


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    train_data_list = get_datalist(args.data_dir, args.data_list, args.max_len)

    createDataset(train_data_list, args.save_path)


if __name__ == "__main__":
    parser = parser_arg()
    args = parser.parse_args()

    main(args)
