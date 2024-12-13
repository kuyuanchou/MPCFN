import random
import os
import torch
import json
import ast
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class TwitterProcessor(object):
    def __init__(self, data_path, bert_name, args) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.args = args

    def load_from_file(self, mode="train"):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
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
        # load aux image
        aux_path = self.data_path[mode + "_auximgs"]
        aux_imgs = torch.load(aux_path)

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs": aux_imgs}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]",
                      "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping


class WukongProcessor(object):
    def __init__(self, data_path, bert_name, args) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.args = args

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.strip().split('\t')[1]
                    # if 'OTHER' in label:
                    #     label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                    len(imgs))
        # load aux image
        if self.data_path[mode + "_auximgs"] is not None:
            with open(self.data_path[mode + "_auximgs"], "r", encoding="utf-8") as f:
                aux_imgs = json.load(f)
            # aux_imgs: {'O_2576.jpg': ['3372_pred_yolo_crop_4836.png'], ...}
            # aux_imgs = torch.load(aux_path)
        else:
            aux_imgs = {}

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs": aux_imgs}

    def get_label_mapping(self):
        LABEL_LIST = ['O', 'B-PER', 'M-PER', 'E-PER', 'B-GPE', 'M-GPE', 'E-GPE', 'B-LOC', 'M-LOC', 'E-LOC', 'B-ORG',
                      'M-ORG', 'E-ORG', 'S-GPE', 'S-PER', 'S-LOC', 'S-ORG', 'S-MOHUGPE', 'S-MOHULOC', 'S-PE', 'B-PE',
                      'M-PE', 'E-PE', 'B-MOHULOC', 'M-MOHULOC', 'E-MOHULOC', "X", "[CLS]", "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping


class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, sample_ratio=1, mode='train',
                 ignore_idx=0, word2nplist_address=None, args=None) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode)
        self.label_mapping = processor.get_label_mapping()
        self.tokenizer = processor.tokenizer
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else None
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.word2nplist_address = word2nplist_address  # add

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], \
            self.data_dict['imgs'][idx]

        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
            encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx] * (
                self.max_seq - len(labels) - 2)

        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, 'inf.jpg')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            # image = torch.zeros((3, 224, 224))

            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                for i in range(3 - len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
                    attention_mask), torch.tensor(labels), image, aux_imgs
            else:
                aux_imgs = []
                for i in range(3):
                    aux_imgs.append(torch.zeros((3, 224, 224)))
                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
                    attention_mask), torch.tensor(labels), image, aux_imgs

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(
            labels)


if __name__ == '__main__':
    pass
