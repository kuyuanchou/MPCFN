import json
import os
import random

import numpy as np
import torch

DATA_PATH = {
    'twitter15': {
        # text data
        'train': 'D:/zyk/HVPNeT(twitter17)/data/twitter15_data/twitter2015/train.txt',
        'dev': 'D:/zyk/HVPNeT(twitter17)/data/twitter15_data/twitter2015/valid.txt',
        'test': 'D:/zyk/HVPNeT(twitter17)/data/twitter15_data/twitter2015/test.txt',
        # {data_id : object_crop_img_path}
        'train_auximgs': 'D:/zyk/HVPNeT(twitter17)/data/twitter15_data/twitter2015/twitter2015_train_dict.pth',
        'dev_auximgs': 'D:/zyk/HVPNeT(twitter17)/data/twitter15_data/twitter2015/twitter2015_val_dict.pth',
        'test_auximgs': 'D:/zyk/HVPNeT(twitter17)/data/twitter15_data/twitter2015/twitter2015_test_dict.pth'
    },

    'twitter17': {
        # text data
        'train': 'D:/zyk/HVPNeT(twitter17)/data/twitter17/twitter2017/train.txt',
        'dev': 'D:/zyk/HVPNeT(twitter17)/data/twitter17/twitter2017/valid.txt',
        'test': 'D:/zyk/HVPNeT(twitter17)/data/twitter17/twitter2017/test.txt',
        # {data_id : object_crop_img_path}
        'train_auximgs': 'D:/zyk/HVPNeT(twitter17)/data/twitter17/twitter2017/twitter2017_train_dict.pth',
        'dev_auximgs': 'D:/zyk/HVPNeT(twitter17)/data/twitter17/twitter2017/twitter2017_val_dict.pth',
        'test_auximgs': 'D:/zyk/HVPNeT(twitter17)/data/twitter17/twitter2017/twitter2017_test_dict.pth'
    },

    'multi_geo': {
        # text data
        'train': './data/shuffle_multi_dataset/dataset_03/json/train.json',
        'dev': './data/shuffle_multi_dataset/dataset_03/json/dev.json',
        'test': './data/shuffle_multi_dataset/dataset_03/json/test.json',
        # {data_id : object_crop_img_path}
        'train_auximgs': None,
        'dev_auximgs': None,
        'test_auximgs': None
    },
    'wukong': {
        # text data
        'train': '../data/wukong/wukong-mner/wukong-text/json/train.json',
        'dev': '../data/wukong/wukong-mner/wukong-text/json/dev.json',
        'test': '../data/wukong/wukong-mner/wukong-text/json/test.json',
        # {data_id : object_crop_img_path}
        'train_auximgs': "./data/wukong/wukong-mner/wukong_detect/train.dict",
        'dev_auximgs': "./data/wukong/wukong-mner/wukong_detect/dev.dict",
        'test_auximgs': "./data/wukong/wukong-mner/wukong_detect/test.dict",
        # aug_text
        "aug_text": "./augtext/wukong/word2nparray.pth"
    },

}


def set_seed(seed=1234):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def load_from_file(dataset, sample_ratio, mode="train"):
    """
    Args:
        mode (str, optional): dataset mode. Defaults to "train".
        sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
    """
    data_path = DATA_PATH[dataset]
    load_file = data_path[mode]
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        raw_words, raw_targets = [], []
        raw_word, raw_target = [], []
        imgs = []
        for line in lines:
            if line.startswith("IMGID:"):
                img_line = line.strip()
                imgs.append(img_line)
                continue
            if line != "\n":
                raw_word.append(line.split('\t')[0])
                label = line.split('\t')[1][:-1]
                if 'OTHER' in label:
                    label = label[:2] + 'MISC'
                raw_target.append(label)
            else:
                raw_words.append(raw_word)
                raw_targets.append(raw_target)
                raw_word, raw_target = [], []

    assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                len(imgs))

    # sample data, only for low-resource
    if mode == "train":
        sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words) * sample_ratio))
    else:
        sample_indexes = list(range(len(raw_words)))

    output = ""
    for idx in sample_indexes:
        output += imgs[idx] + "\n"
        for word, label in zip(raw_words[idx], raw_targets[idx]):
            output += f"{word}\t{label}\n"
        output += "\n"
    return output


def load_wukong(dataset, sample_ratio, mode="train"):
    data_path = DATA_PATH[dataset]
    load_file = data_path[mode]
    with open(load_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    """
    words: [['New', 'Post', ':', 'Blackburn', 'Festival', 'of', 'Voice', '2017'],...]
    targets: [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'],...]
    imgs: ['17_06_12483.jpg', '17_06_2932.jpg', 'O_1926.jpg', '16_05_24_704.jpg', '17_06_12486.jpg']
    """
    raw_words, raw_targets = [], []
    imgs = []
    for single_data in data_list:
        raw_words.append(single_data['sentence'])
        raw_targets.append(single_data['label'])
        imgs.append(str(single_data["image_uid"]))

    # sample data, only for low-resource
    if mode == "train":
        sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words) * sample_ratio))
    else:
        sample_indexes = list(range(len(raw_words)))
    output = ""
    for idx in sample_indexes:
        output += "IMGID:" + imgs[idx] + "\n"
        for word, label in zip(raw_words[idx], raw_targets[idx]):
            output += f"{word}\t{label}\n"
        output += "\n"
    return output


if __name__ == '__main__':
    set_seed(5)
    modes = ["train", "dev", "test"]
    datasets = ["twitter15", "twitter17"]
    sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    out_path = "../data/low_resource/"
    # for dataset in datasets:
    #     for sample_ratio in sample_ratios:
    #         fold = '../data/low_resource/' + dataset + "/" + str(sample_ratio)
    #         os.makedirs(fold)
    #         for mode in modes:
    #             out = load_from_file(dataset, sample_ratio, mode)
    #             if mode == "dev":
    #                 mode = "valid"
    #             with open(fold + "/" + mode + ".txt", mode="w", encoding="UTF-8") as f:
    #                 f.write(out)
    for sample_ratio in sample_ratios:
        fold = '../data/low_resource/wukong/' + str(sample_ratio)
        os.makedirs(fold)
        for mode in modes:
            out = load_wukong("wukong", sample_ratio, mode)
            if mode == "dev":
                mode = "valid"
            with open(fold + "/" + mode + ".txt", mode="w", encoding="UTF-8") as f:
                f.write(out)
